from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import encode_position, encode_positions_batch
from chess_anti_engine.inference import (
    BatchEvaluator,
    LocalModelEvaluator,
    _policy_output,
)
from chess_anti_engine.mcts.sampling import sample_action_with_temperature
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import index_to_move_fast, legal_move_indices
from chess_anti_engine.utils.amp import inference_autocast
from chess_anti_engine.utils.numpy_helpers import softmax_1d


def _softmax_legal(logits: np.ndarray, legal_idx: np.ndarray) -> np.ndarray:
    """Softmax over legal moves only. Returns priors for each legal index."""
    return softmax_1d(logits[legal_idx])


def _value_scalar_from_wdl_logits(wdl_logits: np.ndarray) -> float:
  # Convert (3,) logits into a scalar v in [-1,1] from side-to-move perspective.
  # Pure Python math is faster than numpy for 3-element arrays.
    w, d, l = float(wdl_logits[0]), float(wdl_logits[1]), float(wdl_logits[2])
    mx = max(w, d, l)
    ew = math.exp(w - mx)
    ed = math.exp(d - mx)
    el = math.exp(l - mx)
    s = ew + ed + el
    return (ew - el) / s if s > 0 else 0.0


@dataclass
class MCTSConfig:
    simulations: int = 50
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature: float = 1.0

  # FPU (First Play Urgency): LC0/KataGo-style reduction for unvisited children.
  # Unvisited child Q = parent.Q - fpu * sqrt(sum of visited children's priors).
  # This biases search toward deeply exploring promising lines rather than
  # spreading visits across all children.
    fpu_reduction: float = 1.2  # Non-root nodes (LC0 default)
    fpu_at_root: float = 1.0  # Root node (typically lower — root has Dirichlet noise)

  # Inference AMP: used in selfplay / evaluation for throughput.
  # dtype='auto' => bf16 if supported else fp16.
    use_amp: bool = True
    amp_dtype: str = "auto"


class Node:
    __slots__ = ("_board", "_move", "_action_idx", "parent", "prior", "N", "W", "children", "expanded", "to_play")

    _board: chess.Board | None
    _move: chess.Move | None
    _action_idx: int | None
    parent: Node | None
    prior: float
    N: int
    W: float
    children: dict[int, Node]
    expanded: bool
    to_play: chess.Color

    def __init__(
        self,
        board: chess.Board | None,
        *,
        parent: Node | None,
        prior: float,
        move: chess.Move | None = None,
        action_idx: int | None = None,
    ):
        self._board = board
        self._move = move
        self._action_idx = action_idx
        self.parent = parent
        self.prior = float(prior)
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.expanded = False
        if board is not None:
            self.to_play = board.turn
        elif parent is not None:
            self.to_play = not parent.to_play
        else:
            self.to_play = chess.WHITE

    @property
    def board(self) -> chess.Board:
        if self._board is None:
            assert self.parent is not None
            if self._move is None:
                assert self._action_idx is not None
                self._move = index_to_move_fast(self._action_idx, self.parent.board)
  # Preserve full history so search-time encodings keep LC0 history
  # and repetition planes consistent with python-chess semantics.
            self._board = self.parent.board.copy(stack=True)
            self._board.push(self._move)
        return self._board

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def _select_child(node: Node, *, c_puct: float, fpu_reduction: float) -> tuple[int, Node]:
  # PUCT: Q_eff + c_puct * P * sqrt(N_parent) / (1 + N_child)
  # FPU: unvisited children use parent.Q - fpu_reduction * sqrt(visited_policy)
    c_sqrt_n = c_puct * math.sqrt(max(1, node.N))

  # LC0-style FPU: penalty scales with how much prior mass is already explored
    visited_policy = 0.0
    for ch in node.children.values():
        if ch.N > 0:
            visited_policy += ch.prior
    fpu_value = node.Q - fpu_reduction * math.sqrt(visited_policy)

    best = None
    best_score = -1e30
    for a, ch in node.children.items():
        n = ch.N
  # Child W/Q is stored from the child's side-to-move perspective.
  # Negate visited children so every score is compared in the parent's frame.
        q = (-ch.W / n) if n > 0 else fpu_value
        score = q + c_sqrt_n * ch.prior / (1.0 + n)
        if score > best_score:
            best_score = score
            best = (a, ch)
    assert best is not None
    return best


def _expand_sparse(node: Node, legal_indices: np.ndarray, priors: np.ndarray) -> None:
    """Expand using pre-computed legal indices and their priors (avoids 4672-bool mask)."""
    node.expanded = True
    for i in range(len(legal_indices)):
        a = int(legal_indices[i])
        node.children[a] = Node(None, parent=node, prior=float(priors[i]), action_idx=a)


def _backprop(path: list[Node], value: float) -> None:
  # value is from perspective of the leaf's side-to-move.
    v = float(value)
    for n in reversed(path):
        n.N += 1
        n.W += v
        v = -v  # switch perspective each ply


def _init_root_from_logits(
    board: chess.Board,
    *,
    pol_logits: np.ndarray,
    wdl_logits: np.ndarray,
    rng: np.random.Generator,
    cfg: MCTSConfig,
) -> Node:
    """Create an MCTS root node from pre-computed model logits."""
    root = Node(board.copy(stack=True), parent=None, prior=1.0)

    if root.board.is_game_over():
        root.N = 1
        root.W = _terminal_value(root.board)
        return root

    legal_idx = legal_move_indices(root.board)
    if legal_idx.size > 0:
        pri = _softmax_legal(pol_logits, legal_idx)

        if cfg.dirichlet_eps > 0:
            noise = rng.dirichlet(np.full(int(legal_idx.size), cfg.dirichlet_alpha)).astype(np.float64)
            pri = (1 - cfg.dirichlet_eps) * pri + cfg.dirichlet_eps * noise

        _expand_sparse(root, legal_idx, pri)

    root.N = 1
    root.W = _value_scalar_from_wdl_logits(wdl_logits)
    return root


def _init_root(model: torch.nn.Module, board: chess.Board, *, device: str, rng: np.random.Generator, cfg: MCTSConfig) -> Node:
    """Create root node by running a forward pass then delegating to _init_root_from_logits."""
    x0 = encode_position(board, add_features=True)
    xt = torch.from_numpy(x0[None, ...]).to(device)
    with inference_autocast(device=device, enabled=bool(cfg.use_amp), dtype=str(cfg.amp_dtype)):
        out = model(xt)
    pol_logits = _policy_output(out).detach().float().cpu().numpy().reshape(-1)
    wdl_logits = out["wdl"].detach().float().cpu().numpy().reshape(-1)
    return _init_root_from_logits(board, pol_logits=pol_logits, wdl_logits=wdl_logits, rng=rng, cfg=cfg)


def _terminal_value(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1/2-1/2":
        return 0.0
    if res == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    return 1.0 if board.turn == chess.BLACK else -1.0


def _select_one_leaf(root: Node, *, c_puct: float, fpu_at_root: float,
                     fpu_reduction: float) -> tuple[Node, list[Node]]:
    """Descend the tree from ``root`` to one unexpanded leaf, returning (leaf, path).

    Root selection uses ``fpu_at_root``; deeper selections switch to
    ``fpu_reduction``. Expanded nodes with children are never terminal, so
    the game-over check happens only at the returned leaf in the caller.
    """
    node = root
    path = [node]
    fpu = fpu_at_root
    while node.expanded and node.children:
        _, node = _select_child(node, c_puct=c_puct, fpu_reduction=fpu)
        path.append(node)
        fpu = fpu_reduction
    return node, path


def _expand_and_backprop_leaves(
    leaf_nodes: list[Node], leaf_paths: list[list[Node]],
    *, eval_impl: BatchEvaluator,
) -> None:
    """Run one batched NN eval for the leaves, expand each, and backprop the value."""
    leaf_x = encode_positions_batch([n.board for n in leaf_nodes], add_features=True)
    pol_logits_batch, wdl_logits_batch = eval_impl.evaluate_encoded(leaf_x)
    for node, path, pol_logits, wdl_logits in zip(
        leaf_nodes, leaf_paths, pol_logits_batch, wdl_logits_batch, strict=True,
    ):
        legal_idx = legal_move_indices(node.board)
        if legal_idx.size > 0:
            _expand_sparse(node, legal_idx, _softmax_legal(pol_logits, legal_idx))
        v = _value_scalar_from_wdl_logits(wdl_logits.reshape(-1))
        _backprop(path, v)


def _build_root_outputs(
    root: Node, *, rng: np.random.Generator, temperature: float,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    """Build (probs, action, root_Q, legal_mask) for one finished root."""
    visits_full = np.zeros((POLICY_SIZE,), dtype=np.float32)
    child_actions = np.array(sorted(root.children.keys()), dtype=np.int32)
    child_visits = np.array(
        [float(root.children[int(a)].N) for a in child_actions], dtype=np.float64,
    )
    for a, v in zip(child_actions, child_visits):
        visits_full[a] = v
    s = float(child_visits.sum())
    probs = (visits_full / s) if s > 0 else visits_full

    if child_actions.size == 0:
        action = int(np.argmax(visits_full))
    else:
        action = sample_action_with_temperature(
            rng, child_actions, child_visits, float(temperature),
            argmax_idx=int(np.argmax(child_visits)),
        )

    mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
    mask[child_actions] = True
    return probs.astype(np.float32, copy=False), action, float(root.Q), mask


@torch.no_grad()
def run_mcts_many(
    model: torch.nn.Module | None,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: MCTSConfig,
    evaluator: BatchEvaluator | None = None,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray]]:
    """Run PUCT MCTS for multiple root boards, batching leaf evaluations.

    This aims to keep the GPU busy by evaluating one leaf per active root per
    simulation, in a single forward pass.

    Returns per-root:
    - policy target probs (POLICY_SIZE,)
    - selected action index
    - root Q value
    - legal move mask (POLICY_SIZE,) bool
    """
    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_mcts_many requires model or evaluator")
        eval_impl = LocalModelEvaluator(
            model, device=device,
            use_amp=bool(cfg.use_amp), amp_dtype=str(cfg.amp_dtype),
        )

    if pre_pol_logits is not None and pre_wdl_logits is not None:
        roots = [
            _init_root_from_logits(
                b, pol_logits=pre_pol_logits[i], wdl_logits=pre_wdl_logits[i],
                rng=rng, cfg=cfg,
            )
            for i, b in enumerate(boards)
        ]
    else:
        # _init_root runs the model itself — caller must provide it when pre-logits aren't given.
        assert model is not None
        roots = [_init_root(model, b, device=device, rng=rng, cfg=cfg) for b in boards]

    for _ in range(int(cfg.simulations)):
        leaf_nodes: list[Node] = []
        leaf_paths: list[list[Node]] = []
        for root in roots:
            node, path = _select_one_leaf(
                root, c_puct=cfg.c_puct,
                fpu_at_root=cfg.fpu_at_root, fpu_reduction=cfg.fpu_reduction,
            )
            if node.board.is_game_over():
                _backprop(path, _terminal_value(node.board))
                continue
            leaf_nodes.append(node)
            leaf_paths.append(path)

        if leaf_nodes:
            _expand_and_backprop_leaves(leaf_nodes, leaf_paths, eval_impl=eval_impl)

    probs_list: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    legal_masks: list[np.ndarray] = []
    for root in roots:
        probs, action, root_q, mask = _build_root_outputs(
            root, rng=rng, temperature=float(cfg.temperature),
        )
        probs_list.append(probs)
        actions.append(action)
        values.append(root_q)
        legal_masks.append(mask)

    return probs_list, actions, values, legal_masks


@torch.no_grad()
def run_mcts(
    model: torch.nn.Module | None,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: MCTSConfig,
    evaluator: BatchEvaluator | None = None,
) -> tuple[np.ndarray, int, float]:
    probs_list, actions, values, _masks = run_mcts_many(model, [board], device=device, rng=rng, cfg=cfg, evaluator=evaluator)
    return probs_list[0], actions[0], float(values[0])
