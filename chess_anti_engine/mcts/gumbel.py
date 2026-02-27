from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import chess
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask, index_to_move
from chess_anti_engine.utils.amp import inference_autocast


def _gumbel(rng: np.random.Generator, size: int) -> np.ndarray:
    # Sample Gumbel(0,1): -log(-log(U))
    u = rng.random(size=size)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = float(e.sum())
    return e / s if s > 0 else np.full_like(x, 1.0 / x.size)


def _wdl_to_q(wdl_logits: np.ndarray) -> float:
    """WDL logits → scalar Q ∈ [-1, 1] from side-to-move perspective (W-L)."""
    p = _softmax(wdl_logits.astype(np.float64))
    return float(p[0] - p[2])


@dataclass
class GumbelConfig:
    simulations: int = 50
    topk: int = 16
    child_sims: int = 8  # kept for API compat, no longer used
    temperature: float = 1.0


@torch.no_grad()
def run_gumbel_root_many(
    model: torch.nn.Module,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Gumbel MCTS with sequential halving, fully batched.

    Algorithm (Danihelka et al., ICLR 2022 — simplified Phase-1 version):
      1. Batch-evaluate all root positions → priors + root Q values.
         (Skipped if pre_pol_logits / pre_wdl_logits are supplied — callers
         that already evaluated the root can pass them in to avoid a second
         forward pass.)
      2. Per root: apply Gumbel noise + log-prior, select top-k candidates.
      3. Sequential halving: each round, batch-evaluate ALL remaining candidate
         child boards in one GPU forward pass; use the value head (negated,
         since child is opponent's turn) as Q(a).
      4. Improved policy: π'(a) ∝ π(a) * exp(clip(Q(a), -5, 5)).
      5. Action selected from improved policy (with temperature).

    Returns (probs_list, actions, root_values).
    """
    n_boards = len(boards)
    if n_boards == 0:
        return [], [], []

    sim_budget = max(1, int(cfg.simulations))

    # ── 1. Batch root evaluation ─────────────────────────────────────────────
    if pre_pol_logits is not None and pre_wdl_logits is not None:
        # Reuse logits computed by the caller (saves one forward pass per ply).
        pol_logits_batch = np.asarray(pre_pol_logits, dtype=np.float32)   # (B, POLICY_SIZE)
        wdl_logits_batch = np.asarray(pre_wdl_logits, dtype=np.float32)   # (B, 3)
    else:
        xs = [encode_position(b, add_features=True) for b in boards]
        xt = torch.from_numpy(np.stack(xs, axis=0)).to(device)
        with inference_autocast(device=device, enabled=True, dtype="auto"):
            root_out = model(xt)
        policy_out = root_out["policy"] if "policy" in root_out else root_out["policy_own"]
        pol_logits_batch = policy_out.detach().float().cpu().numpy()  # (B, POLICY_SIZE)
        wdl_logits_batch = root_out["wdl"].detach().float().cpu().numpy()  # (B, 3)

    root_qs = [_wdl_to_q(wdl_logits_batch[i]) for i in range(n_boards)]

    # ── 2. Per-board prior + Gumbel candidate selection ──────────────────────
    probs_out: list[np.ndarray | None] = [None] * n_boards
    actions_out: list[int | None] = [None] * n_boards
    values_out: list[float] = list(root_qs)

    priors: list[np.ndarray | None] = [None] * n_boards
    candidates_per_board: list[list[int] | None] = [None] * n_boards
    q_map_per_board: list[dict[int, float] | None] = [None] * n_boards
    remaining_per_board: list[list[int] | None] = [None] * n_boards

    for i, b in enumerate(boards):
        mask = legal_move_mask(b)
        legal = np.nonzero(mask)[0]

        if legal.size == 0:
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            continue

        # Masked prior
        pl = pol_logits_batch[i].astype(np.float64)
        pl[~mask] = -1e9
        pri = _softmax(pl)
        pri[~mask] = 0.0
        priors[i] = pri

        if legal.size == 1:
            a0 = int(legal[0])
            p = np.zeros((POLICY_SIZE,), dtype=np.float32)
            p[a0] = 1.0
            probs_out[i] = p
            actions_out[i] = a0
            continue

        # Gumbel noise → select top-k
        log_pri = np.log(np.maximum(pri[legal], 1e-12))
        g = _gumbel(rng, legal.size)
        score = g + log_pri

        k_cap = max(2, sim_budget // 2)
        k = int(min(int(cfg.topk), int(legal.size), int(k_cap)))
        k = max(2, k)

        kth = min(k - 1, int(score.size) - 1)
        top_idx = np.argpartition(-score, kth)[:k]
        cands = legal[top_idx].astype(int).tolist()

        candidates_per_board[i] = cands
        # Initialise Q with root value (will be overwritten by child evals)
        q_map_per_board[i] = {a: float(root_qs[i]) for a in cands}
        remaining_per_board[i] = list(cands)

    # ── 3. Sequential halving (batched child evaluation each round) ───────────
    # Track which (board_idx, action) pairs have already been GPU-evaluated.
    # The network is deterministic at inference time, so re-evaluating the same
    # child position would return identical Q values. Caching avoids the redundant
    # encode + forward-pass overhead in halving rounds 2+.
    evaluated_per_board: list[set[int]] = [set() for _ in range(n_boards)]

    def _active_boards() -> list[int]:
        return [
            i for i in range(n_boards)
            if remaining_per_board[i] is not None and len(remaining_per_board[i]) > 1
        ]

    while True:
        active = _active_boards()
        if not active:
            break

        # Only collect children not yet evaluated — survivors from earlier rounds
        # already have correct Q values cached in q_map_per_board.
        child_entries: list[tuple[int, int]] = []  # (board_idx, action)
        child_boards_list: list[chess.Board] = []
        for bi in active:
            for a in remaining_per_board[bi]:
                if int(a) not in evaluated_per_board[bi]:
                    mv = index_to_move(int(a), boards[bi])
                    b2 = boards[bi].copy(stack=False)
                    b2.push(mv)
                    child_entries.append((bi, a))
                    child_boards_list.append(b2)

        if child_boards_list:
            child_xs = [encode_position(b2, add_features=True) for b2 in child_boards_list]
            child_xt = torch.from_numpy(np.stack(child_xs, axis=0)).to(device)
            with inference_autocast(device=device, enabled=True, dtype="auto"):
                child_out = model(child_xt)
            child_wdl = child_out["wdl"].detach().float().cpu().numpy()  # (N, 3)

            for j, (bi, a) in enumerate(child_entries):
                # Child is opponent's turn → negate for root perspective
                q_map_per_board[bi][int(a)] = -_wdl_to_q(child_wdl[j])
                evaluated_per_board[bi].add(int(a))

        # Eliminate bottom half of remaining candidates per board
        for bi in active:
            rem = remaining_per_board[bi]
            rem.sort(key=lambda a: q_map_per_board[bi].get(int(a), 0.0), reverse=True)
            remaining_per_board[bi] = rem[: max(1, len(rem) // 2)]

    # ── 4. Build improved policies ────────────────────────────────────────────
    for i in range(n_boards):
        if probs_out[i] is not None:
            continue  # handled (empty / single-legal cases)

        cands = candidates_per_board[i]
        pri = priors[i]
        q_map = q_map_per_board[i]
        remaining = remaining_per_board[i]

        q_vals = np.array([q_map.get(int(a), 0.0) for a in cands], dtype=np.float64)
        pri_c = np.array([pri[int(a)] for a in cands], dtype=np.float64)

        # Improved policy: π'(a) ∝ π(a) · exp(Q(a))   (Gumbel eq. 10, simplified)
        imp = pri_c * np.exp(np.clip(q_vals, -5.0, 5.0))
        s = float(imp.sum())
        imp = imp / s if s > 0 else np.full_like(imp, 1.0 / imp.size)

        probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
        probs[cands] = imp.astype(np.float32)

        # Action from sequential halving winner; re-sample with temperature if requested
        best_a = int(remaining[0])
        if cfg.temperature <= 0:
            action = best_a
        else:
            p = imp.copy()
            if cfg.temperature != 1.0:
                p = np.power(np.maximum(p, 0.0), 1.0 / float(cfg.temperature))
            ps = float(p.sum())
            if ps > 0:
                p /= ps
                action = int(rng.choice(cands, p=p))
            else:
                action = best_a

        probs_out[i] = probs
        actions_out[i] = action

    return probs_out, actions_out, values_out


@torch.no_grad()
def run_gumbel_root(
    model: torch.nn.Module,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, int, float]:
    probs, acts, vals = run_gumbel_root_many(model, [board], device=device, rng=rng, cfg=cfg)
    return probs[0], acts[0], float(vals[0])
