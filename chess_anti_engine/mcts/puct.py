from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask, move_to_index, index_to_move
from chess_anti_engine.utils.amp import inference_autocast


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = float(e.sum())
    if s <= 0:
        return np.full_like(x, 1.0 / x.size)
    return e / s


def _value_scalar_from_wdl_logits(wdl_logits: np.ndarray) -> float:
    # Convert (3,) logits into a scalar v in [-1,1] from side-to-move perspective.
    p = _softmax_np(wdl_logits.astype(np.float64))
    return float(p[0] - p[2])


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
    fpu_reduction: float = 1.2   # Non-root nodes (LC0 default)
    fpu_at_root: float = 1.0     # Root node (typically lower — root has Dirichlet noise)

    # Inference AMP: used in selfplay / evaluation for throughput.
    # dtype='auto' => bf16 if supported else fp16.
    use_amp: bool = True
    amp_dtype: str = "auto"


class Node:
    __slots__ = ("board", "parent", "prior", "N", "W", "children", "expanded", "to_play")

    def __init__(self, board: chess.Board, *, parent: Node | None, prior: float):
        self.board = board
        self.parent = parent
        self.prior = float(prior)
        self.N = 0
        self.W = 0.0
        self.children: dict[int, Node] = {}
        self.expanded = False
        self.to_play = board.turn

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def _select_child(node: Node, *, c_puct: float, fpu_reduction: float) -> tuple[int, Node]:
    # PUCT: Q_eff + c_puct * P * sqrt(N_parent) / (1 + N_child)
    # FPU: unvisited children use parent.Q - fpu_reduction * sqrt(visited_policy)
    sqrt_n = np.sqrt(max(1, node.N))

    # LC0-style FPU: penalty scales with how much prior mass is already explored
    visited_policy = sum(ch.prior for ch in node.children.values() if ch.N > 0)
    fpu_value = node.Q - fpu_reduction * np.sqrt(visited_policy)

    best = None
    best_score = -1e30
    for a, ch in node.children.items():
        q = ch.Q if ch.N > 0 else fpu_value
        u = c_puct * ch.prior * sqrt_n / (1.0 + ch.N)
        score = q + u
        if score > best_score:
            best_score = score
            best = (a, ch)
    assert best is not None
    return best


def _expand(node: Node, priors: np.ndarray) -> None:
    # priors is (POLICY_SIZE,), already masked to legal.
    node.expanded = True
    for a_idx in np.nonzero(priors > 0)[0]:
        a = int(a_idx)
        move = index_to_move(a, node.board)
        b2 = node.board.copy(stack=False)
        b2.push(move)
        node.children[a] = Node(b2, parent=node, prior=float(priors[a]))


def _backprop(path: list[Node], value: float) -> None:
    # value is from perspective of the leaf's side-to-move.
    v = float(value)
    for n in reversed(path):
        n.N += 1
        n.W += v
        v = -v  # switch perspective each ply


def _init_root(model: torch.nn.Module, board: chess.Board, *, device: str, rng: np.random.Generator, cfg: MCTSConfig) -> Node:
    root = Node(board.copy(stack=False), parent=None, prior=1.0)

    x0 = encode_position(root.board, add_features=True)
    xt = torch.from_numpy(x0[None, ...]).to(device)
    with inference_autocast(device=device, enabled=bool(cfg.use_amp), dtype=str(cfg.amp_dtype)):
        out = model(xt)
    policy_out = out["policy"] if "policy" in out else out["policy_own"]
    pol_logits = policy_out.detach().float().cpu().numpy().reshape(-1)
    wdl_logits = out["wdl"].detach().float().cpu().numpy().reshape(-1)

    mask = legal_move_mask(root.board)
    pol_logits = pol_logits.astype(np.float64)
    pol_logits[~mask] = -1e9
    pri = _softmax_np(pol_logits)
    pri[~mask] = 0.0

    legal_idxs = np.nonzero(mask)[0]
    if legal_idxs.size > 0 and cfg.dirichlet_eps > 0:
        noise = rng.dirichlet([cfg.dirichlet_alpha] * int(legal_idxs.size)).astype(np.float64)
        pri2 = pri.copy()
        pri2[legal_idxs] = (1 - cfg.dirichlet_eps) * pri2[legal_idxs] + cfg.dirichlet_eps * noise
        pri = pri2

    _expand(root, pri)
    root.N = 1
    root.W = _value_scalar_from_wdl_logits(wdl_logits)
    return root


def _terminal_value(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1/2-1/2":
        return 0.0
    if res == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    return 1.0 if board.turn == chess.BLACK else -1.0


@torch.no_grad()
def run_mcts_many(
    model: torch.nn.Module,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: MCTSConfig,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Run PUCT MCTS for multiple root boards, batching leaf evaluations.

    This aims to keep the GPU busy by evaluating one leaf per active root per
    simulation, in a single forward pass.

    Returns per-root:
    - policy target probs (POLICY_SIZE,)
    - selected action index
    """
    roots = [_init_root(model, b, device=device, rng=rng, cfg=cfg) for b in boards]

    for _ in range(int(cfg.simulations)):
        leaf_nodes: list[Node] = []
        leaf_paths: list[list[Node]] = []
        leaf_x: list[np.ndarray] = []

        # Select one leaf per root
        for root in roots:
            node = root
            path = [node]
            fpu = cfg.fpu_at_root  # First selection from root uses root FPU
            while node.expanded and node.children:
                _, node = _select_child(node, c_puct=cfg.c_puct, fpu_reduction=fpu)
                path.append(node)
                fpu = cfg.fpu_reduction  # Subsequent selections use tree FPU
                if node.board.is_game_over():
                    break

            if node.board.is_game_over():
                _backprop(path, _terminal_value(node.board))
                continue

            leaf_nodes.append(node)
            leaf_paths.append(path)
            leaf_x.append(encode_position(node.board, add_features=True))

        if not leaf_nodes:
            continue

        # Batched eval
        xt = torch.from_numpy(np.stack(leaf_x, axis=0)).to(device)
        with inference_autocast(device=device, enabled=bool(cfg.use_amp), dtype=str(cfg.amp_dtype)):
            out = model(xt)
        policy_out = out["policy"] if "policy" in out else out["policy_own"]
        pol_logits_batch = policy_out.detach().float().cpu().numpy()
        wdl_logits_batch = out["wdl"].detach().float().cpu().numpy()

        for node, path, pol_logits, wdl_logits in zip(leaf_nodes, leaf_paths, pol_logits_batch, wdl_logits_batch, strict=True):
            mask = legal_move_mask(node.board)
            pl = pol_logits.astype(np.float64)
            pl[~mask] = -1e9
            pri = _softmax_np(pl)
            pri[~mask] = 0.0
            _expand(node, pri)
            v = _value_scalar_from_wdl_logits(wdl_logits.reshape(-1))
            _backprop(path, v)

    probs_list: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []

    for root in roots:
        visits = np.zeros((POLICY_SIZE,), dtype=np.float32)
        for a, ch in root.children.items():
            visits[a] = float(ch.N)
        s = float(visits.sum())
        probs = (visits / s) if s > 0 else visits

        if cfg.temperature <= 0:
            action = int(np.argmax(visits))
        else:
            # numpy.random.Generator.choice is strict about probabilities summing to 1.
            # "probs" is float32 and can drift far enough from 1.0 across ~4.6k entries
            # to trigger ValueError, so always renormalize in float64.
            p = probs.astype(np.float64, copy=True)
            if cfg.temperature != 1.0:
                p = np.power(p, 1.0 / float(cfg.temperature))

            ps = float(p.sum())
            if not np.isfinite(ps) or ps <= 0:
                action = int(np.argmax(visits))
            else:
                p /= ps
                # If any numerical weirdness causes small negative entries, clip & renormalize.
                if np.any(p < 0):
                    p = np.clip(p, 0.0, None)
                    ps2 = float(p.sum())
                    if ps2 <= 0:
                        action = int(np.argmax(visits))
                    else:
                        p /= ps2
                        action = int(rng.choice(POLICY_SIZE, p=p))
                else:
                    action = int(rng.choice(POLICY_SIZE, p=p))

        probs_list.append(probs.astype(np.float32, copy=False))
        actions.append(action)
        values.append(float(root.Q))

    return probs_list, actions, values


@torch.no_grad()
def run_mcts(
    model: torch.nn.Module,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: MCTSConfig,
) -> tuple[np.ndarray, int, float]:
    probs_list, actions, values = run_mcts_many(model, [board], device=device, rng=rng, cfg=cfg)
    return probs_list[0], actions[0], float(values[0])
