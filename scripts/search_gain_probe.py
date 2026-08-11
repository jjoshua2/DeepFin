#!/usr/bin/env python3
"""Does Gumbel search change the network's move, and does it change it for the better?

A per-move instrument for the sims->strength question, built because an Elo arena
cannot answer it: the loop gains ~0.02 Elo/iteration and a 200-game paired arena
resolves ~+/-42 Elo, so a sims ladder that costs a day of GPU returns a CI wide
enough to contain both "search is worthless" and "search is healthy". This script
scores thousands of POSITIONS instead of hundreds of games, against the frozen
deep-SF audit set, and decomposes the question the way it has to be decomposed:

  1. CHANGE RATE   - how often does search play something other than argmax(prior)?
  2. CHANGE QUALITY- on those positions, is the new move better by deep SF?
  3. SIGMA HEADROOM- can the value transform re-rank the prior *at all*? The
     improved policy is softmax(log_prior + sigma(Qbar)) with sigma spanning
     exactly `q_scale` nats across the legal moves, so a prior whose top-1/top-2
     logit gap exceeds `q_scale` is UNREACHABLE by any value estimate, however
     good. That is a structural bound, not a tuning observation.
  4. Q QUALITY     - does more search produce a BETTER Qbar (rank correlation
     against deep SF), or just a more confident one?

A blended metric cannot separate 1 from 2: a low change rate and a high change
rate with neutral quality are different diseases with the same symptom.

CONTROLS (a statistic that cannot fail is not an instrument). They are ALWAYS
on -- there is no `--controls` flag, because a control an operator can forget
is a control that will be forgotten. Every arm is scored through the identical
code path as the search arm and printed in section 5:
  * `prior`   - the prior's own move, but with its regret looked up by UCI
                STRING through `move_regrets` instead of by array position.
                Must reproduce `regret_prior` to 0.00; a nonzero max deviation
                means `ucis` and the regret vector are misaligned and every cp
                number in the table is being read off the wrong move.
  * `rank2`   - the prior's SECOND move. Must score clearly POSITIVE; it
                calibrates what one rank of prior is worth in cp.
  * `random`  - a uniformly random legal move. Must score STRONGLY positive.
  * `shuffled`- ⚑ THE NEGATIVE CONTROL. The SAME search and the SAME prior,
                scored against a WITHIN-POSITION PERMUTATION of that position's
                deep-SF labels. It destroys the move->quality association while
                preserving the legal set, the coverage, and the marginal
                distribution of regrets exactly, so it MUST collapse to ~0 with
                a CI covering 0. If it does not, the quality metric is reading
                something other than move quality and no number above it means
                anything. (An earlier revision permuted the search REGRETS
                across positions while subtracting the same prior vector; that
                leaves the mean algebraically unchanged and could not fail.
                Permuting the LABELS is not the same operation and does fail.)
  * `p(sign)` - a paired sign-flip permutation test on the search delta. This
                is a null-hypothesis test on the real data, NOT a negative
                control; `shuffled` is the negative control.
If `random` does not come out worse than `search`, or `shuffled` does not
collapse, the quality metric is broken and no number in the table means
anything.

SEARCH SHAPE IS NOT OPTIONAL AND HAS NO SAFE DEFAULT. `configs/pbt2_small.yaml`
on the live branch and on `origin/main` disagree on every Gumbel knob that
matters, and the root value-transform has a THIRD setting used only by
play/UCI. `--shape` is required and its realized fields are printed in the
header and stored in every output row.

⚑⚑ THE BUILT-IN SHAPES ARE DATED SNAPSHOTS, NOT LIVE READINGS, and their names
say so. `live_selfplay_20260809` is what the live yaml said on 2026-08-09; the
live yaml is not in this repo and moves without touching this file. It has
ALREADY moved once (`gumbel_c_scale` 0.025 -> 0.1 and `gumbel_policy_temp`
1.0 -> 1.5 shipped on 2026-08-09/10), which is exactly the failure this naming
exists to stop -- a fixture called `live_selfplay` that quietly stops being
live is [[same_name_different_population]]. To measure TODAY's shape, point
`--shape-yaml` at the actual live config and the shape is read from it:

    PYTHONPATH=. python3 scripts/search_gain_probe.py \\
        --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --shape live_selfplay_20260809 \\
        --shape-yaml /home/josh/projects/chess/configs/pbt2_small.yaml \\
        --sims 1,8,32,64,128,256 --positions 512 --out scratchpad/probe.jsonl

Usage (a dated snapshot, when reproducing a banked number rather than measuring
today's loop):

    PYTHONPATH=. python3 scripts/search_gain_probe.py \\
        --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --shape live_selfplay_20260809 --sims 1,8,32,64,128,256 \\
        --positions 512 --out scratchpad/probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.eval.audit import (
    AuditPosition,
    decode_board_from_planes,
    legal_full_indices,
    load_audit_set,
    move_regrets,
)
from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, POLICY_SIZE, policy_batch_to_full_if_needed
from chess_anti_engine.utils.git_meta import git_sha

# `finalize._build_sf_p0_regret_vector` divides by this before storing, so the
# stored per-move value is a FRACTION of it, not a cp figure.
SF_OWN_REGRET_CAP_CP = 1000.0

# Production's own `data_policy_entropy` range over the live run (banked-shard
# median ~0.924). It is the fidelity gate's threshold: a rig claiming to
# reproduce the production search at the production shape must land in it, and
# one that does not may quote only DELTAS between its own arms.
DATA_POLICY_ENTROPY_BAND: tuple[float, float] = (0.889, 0.979)


# ---------------------------------------------------------------------------
# Search shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchShape:
    """Every GumbelConfig field a Gumbel search can act on, named explicitly.

    No field falls back to a dataclass default: the whole point of this script
    is that the sentinels (`c_scale_root=-1`, `q_visit_exp_root=99`) silently
    select a DIFFERENT root value-transform than the one the +Elo result was
    measured on, and a shape that inherits them without saying so reproduces
    the defect it is trying to measure.
    """

    label: str
    topk: int
    c_scale: float
    c_visit: float
    c_visit_root: float
    c_scale_root: float
    q_visit_exp: float
    q_visit_exp_root: float
    halving_div: int
    policy_temp: float
    vloss_weight: int
    target_batch: int
    provenance: str

    def root_q_scale(self, max_visit: float) -> float:
        """Realized root sigma(q) span in nats, resolving the sentinels.

        Mirrors `gumbel._root_sigma_scale` and `_mcts_tree.c gss_score_and_halve`.
        The improved policy's sigma term spans exactly this many nats from the
        worst to the best completed-Q, because the transform min-max normalizes
        the completed values to [0, 1] before scaling.
        """
        cvr = self.c_visit_root if self.c_visit_root >= 0.0 else self.c_visit
        csr = self.c_scale_root if self.c_scale_root >= 0.0 else self.c_scale
        qer = self.q_visit_exp_root if self.q_visit_exp_root < 90.0 else self.q_visit_exp
        if qer < 0.0:
            return csr * math.log1p(cvr + max_visit)
        mv_term = max_visit if qer == 1.0 else max_visit**qer
        return csr * (cvr + mv_term)

    def descent_q_scale(self, max_visit: float) -> float:
        """Realized DESCENT sigma(q) span. The root and descent sites use
        different constants, so they have to be reported separately."""
        return self.c_scale * (self.c_visit + max_visit)

    def root_transform_name(self) -> str:
        qer = self.q_visit_exp_root if self.q_visit_exp_root < 90.0 else self.q_visit_exp
        if qer < 0.0:
            return "LOG"
        if qer == 1.0:
            return "LINEAR"
        return f"POWER({qer})"


# ⚑ A DATED SNAPSHOT OF THE LIVE SELFPLAY SHAPE, read off
# `configs/pbt2_small.yaml` on branch `ops/live-20260725` ON 2026-08-09. It is
# NOT a live reading and it has already gone stale once: the search-authority
# bundle shipped `gumbel_c_scale` 0.025 -> 0.1 and `gumbel_policy_temp`
# 1.0 -> 1.5 within a day of it being written. Use `--shape-yaml` to read the
# shape off the actual live config; keep this entry only to reproduce numbers
# banked against it.
#
# `selfplay/network_turn.py` builds its GumbelConfig from an explicit keyword
# list that OMITS c_visit / c_visit_root / c_scale_root / q_visit_exp_root /
# halving_div -- those take the GumbelConfig dataclass default, and for the root
# trio that default is a SENTINEL meaning "fall back to the descent knob", so
# live selfplay runs a LINEAR root. ⚑ `policy_temp` is NOT in that omitted set
# any more: PR #385 plumbed `gumbel_policy_temp` through to selfplay, and
# `test_selfplay_plumbs_policy_temp_but_not_the_root_transform` pins which side
# of the line each field is on so this comment cannot drift again.
SHAPES: dict[str, SearchShape] = {
    "live_selfplay_20260809": SearchShape(
        label="LIVE selfplay SNAPSHOT 2026-08-09 (ops/live-20260725)",
        topk=32, c_scale=0.025, c_visit=50.0,
        c_visit_root=-1.0, c_scale_root=-1.0,
        q_visit_exp=1.0, q_visit_exp_root=99.0,
        halving_div=2, policy_temp=1.0,
        vloss_weight=1, target_batch=0,
        provenance="SNAPSHOT of the live yaml as of 2026-08-09: gumbel_c_scale "
                   "0.025 / gumbel_topk 32 / gumbel_vloss_weight 1 / "
                   "gumbel_policy_temp absent -> 1.0; root trio unset -> "
                   "sentinels -> linear root. THE LIVE YAML HAS SINCE MOVED "
                   "(c_scale 0.1, policy_temp 1.5) -- use --shape-yaml to read it",
    ),
    "main_selfplay": SearchShape(
        label="origin/main selfplay defaults",
        topk=16, c_scale=0.1, c_visit=50.0,
        c_visit_root=-1.0, c_scale_root=-1.0,
        q_visit_exp=1.0, q_visit_exp_root=99.0,
        halving_div=2, policy_temp=1.0,
        vloss_weight=1, target_batch=0,
        provenance="origin/main configs/pbt2_small.yaml + GumbelConfig defaults",
    ),
    "play": SearchShape(
        label="PLAY/UCI (PLAY_SEARCH_DEFAULTS)",
        topk=32, c_scale=0.025, c_visit=50.0,
        c_visit_root=900.0, c_scale_root=7.0,
        q_visit_exp=1.0, q_visit_exp_root=-1.0,
        halving_div=2, policy_temp=1.0,
        vloss_weight=3, target_batch=0,
        provenance="mcts.gumbel.PLAY_SEARCH_DEFAULTS + PLAY_SEARCH_VLOSS_WEIGHT",
    ),
    # The live selfplay shape with ONLY the root transform swapped to the play
    # shape's log root. Isolates "which root transform" from every other knob,
    # which the play-vs-selfplay contrast confounds with topk and vloss_weight.
    "live_selfplay_20260809_logroot": SearchShape(
        label="LIVE selfplay SNAPSHOT 2026-08-09 + LOG root (isolation arm)",
        topk=32, c_scale=0.025, c_visit=50.0,
        c_visit_root=900.0, c_scale_root=7.0,
        q_visit_exp=1.0, q_visit_exp_root=-1.0,
        halving_div=2, policy_temp=1.0,
        vloss_weight=1, target_batch=0,
        provenance="live selfplay with c_visit_root/c_scale_root/q_visit_exp_root "
                   "set to PLAY_SEARCH_DEFAULTS; everything else live",
    ),
    # sqrt-N root. The estimator argument for a sublinear exponent is that the
    # standard error of Qbar shrinks like 1/sqrt(N), so trust should grow like
    # sqrt(N); weighting it linearly in N over-trusts by ~sqrt(N). (The Gumbel
    # MuZero paper's own sigma is the LINEAR form -- "log is more principled" is
    # a claim about estimator reliability, not about the paper.) sqrt sits
    # between the two shapes production actually runs, and nothing has ever
    # measured it. c_scale_root is set so the 256-sim root q_scale lands in the
    # same band as the log root's; see the report's inversion table.
    "live_selfplay_20260809_sqrtroot": SearchShape(
        label="LIVE selfplay SNAPSHOT 2026-08-09 + sqrt(N) root (untested middle)",
        topk=32, c_scale=0.025, c_visit=50.0,
        c_visit_root=0.0, c_scale_root=6.5,
        q_visit_exp=1.0, q_visit_exp_root=0.5,
        halving_div=2, policy_temp=1.0,
        vloss_weight=1, target_batch=0,
        provenance="live selfplay, q_visit_exp_root=0.5 (sqrt-N), c_scale_root "
                   "chosen to match the log root's realized 256-sim q_scale",
    ),
}


# yaml key -> SearchShape field, for the knobs `network_turn.py` actually passes
# through to GumbelConfig. Everything NOT in this map is omitted by the
# production call site and therefore takes the GumbelConfig dataclass default,
# which is what the snapshot shapes above encode.
_YAML_TO_SHAPE: dict[str, str] = {
    "gumbel_topk": "topk",
    "gumbel_c_scale": "c_scale",
    "gumbel_policy_temp": "policy_temp",
    "gumbel_vloss_weight": "vloss_weight",
}


def _find_scalar_keys(node: object, wanted: set[str]) -> dict[str, list[float]]:
    """Every NUMERIC value any of `wanted` takes in a nested yaml document.

    Collected as a LIST per key rather than a single value: `pbt2_small.yaml`
    carries `curriculum_gumbel_scale` alongside `gumbel_scale` and a naive
    "first hit wins" reader would silently pick whichever the walk reached
    first. A key found twice with different values is an error, not a choice.

    A non-numeric value raises rather than being skipped: skipping it would make
    the key read as ABSENT, and "absent" is handled by keeping the base shape's
    value -- so a typo'd yaml would silently measure the snapshot again.
    """
    found: dict[str, list[float]] = {k: [] for k in wanted}
    stack: list[object] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            items: list[tuple[object, object]] = list(cur.items())
            for k, v in items:
                if isinstance(k, str) and k in wanted:
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise SystemExit(
                            f"yaml key {k!r} has non-numeric value {v!r}; this "
                            "reader cannot turn it into a search-shape field"
                        )
                    found[k].append(float(v))
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return found


def shape_from_yaml(path: Path, *, base: SearchShape) -> SearchShape:
    """Build a SearchShape from a real config file instead of a hardcoded copy.

    THE POINT. The live yaml is not in this repo, is re-read every iteration,
    and moves without touching this script -- so any hardcoded "live" fixture is
    a claim with a shelf life. Reading it makes the shape a MEASUREMENT of the
    config the operator names, and the resolved values are echoed into the
    header and the dump's provenance so a number can never be read without them.

    Only the knobs `network_turn.py` passes are taken from the yaml; the rest
    keep `base`'s values, which are the GumbelConfig defaults the production
    call site leaves alone. `test_selfplay_plumbs_policy_temp_but_not_the_root_transform`
    fails the day that split changes, so this reader cannot silently under-read.
    """
    import dataclasses

    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    found = _find_scalar_keys(doc, set(_YAML_TO_SHAPE))
    changes: dict[str, int | float] = {}
    resolved: list[str] = []
    for ykey, field_name in sorted(_YAML_TO_SHAPE.items()):
        vals = found[ykey]
        uniq = {repr(v) for v in vals}
        if len(uniq) > 1:
            raise SystemExit(
                f"--shape-yaml {path}: {ykey!r} appears with conflicting values "
                f"{sorted(uniq)}; refusing to guess which one selfplay reads"
            )
        if not vals:
            # Absent means the SelfPlayConfig default applies, which is NOT
            # necessarily this shape's value -- say so rather than inheriting.
            resolved.append(f"{ykey}=ABSENT->keeping {getattr(base, field_name)!r}")
            continue
        cur = getattr(base, field_name)
        changes[field_name] = int(vals[0]) if isinstance(cur, int) else float(vals[0])
        resolved.append(f"{ykey}={changes[field_name]!r}")
    return dataclasses.replace(
        base,
        label=f"{base.label} [READ FROM {path}]",
        provenance=f"read from {path}: " + ", ".join(resolved)
                   + "; every other field keeps the GumbelConfig default that "
                     "network_turn.py leaves unset",
        **changes,
    )


def apply_overrides(shape: SearchShape, spec: str | None) -> SearchShape:
    """`--override c_scale_root=3.0,topk=16` -> a new SearchShape.

    Every override is echoed in the header and stored in the dump, so a number
    can never be read without the shape that produced it.
    """
    import dataclasses

    if not spec:
        return shape
    changes: dict[str, object] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in {f.name for f in dataclasses.fields(SearchShape)} or k in {"label", "provenance"}:
            raise SystemExit(f"--override: unknown or non-overridable field {k!r}")
        cur = getattr(shape, k)
        changes[k] = int(v) if isinstance(cur, int) else float(v)
    return dataclasses.replace(
        shape,
        label=f"{shape.label} [override {spec}]",
        provenance=f"{shape.provenance}; OVERRIDDEN {spec}",
        **changes,
    )


# ---------------------------------------------------------------------------
# Position sources
# ---------------------------------------------------------------------------


@dataclass
class ProbePosition:
    """A position to search, plus whatever ground truth came with it.

    `move_cp` is the deep-SF label map when the source is the frozen audit set,
    and the production SF label when the source is a banked shard. `stored_target`
    is the improved policy production ACTUALLY stored for this row -- present only
    for shard rows, and used as the harness-fidelity gate.
    """

    fen: str
    board: chess.Board
    phase: int
    audit: AuditPosition | None
    stored_target: np.ndarray | None = None      # full-4672 probs, or None
    stored_regret_cp: np.ndarray | None = None   # full-4672 cp regret, or None
    ply_index: int = -1
    priority_policy_kl: float = float("nan")
    extra: dict[str, float] = field(default_factory=dict)


def audit_positions(path: Path, n: int, rng: np.random.Generator) -> list[ProbePosition]:
    all_pos = load_audit_set(path)
    if n and n < len(all_pos):
        sel = sorted(rng.choice(len(all_pos), size=int(n), replace=False).tolist())
        all_pos = [all_pos[int(i)] for i in sel]
    return [
        ProbePosition(fen=p.fen, board=chess.Board(p.fen), phase=int(p.phase), audit=p)
        for p in all_pos
    ]


def _checkpoint_history_encoding(checkpoint: str) -> str:
    """The history encoding the checkpoint declares, without building the model.

    `decode_board_from_planes` reads different plane offsets per encoding, so
    guessing this would silently decode the wrong squares out of every stored
    row and produce a set of plausible-looking but wrong positions.
    """
    import torch

    from chess_anti_engine.uci.model_loader import _resolve_trainer_pt

    path = _resolve_trainer_pt(Path(checkpoint))
    try:
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch") if isinstance(ckpt, dict) else None
    if not isinstance(arch, dict) or "input_history_encoding" not in arch:
        raise SystemExit(
            f"{path} carries no arch.input_history_encoding; shard decoding "
            "cannot proceed without it"
        )
    return str(arch["input_history_encoding"])


def shard_positions(
    glob_pat: str, n: int, rng: np.random.Generator, *, hist: str, n_shards: int,
) -> list[ProbePosition]:
    """Real production positions, decoded back out of the banked replay shards.

    ⚑ The decoded board has an EMPTY move stack, so re-encoding it loses the
    history planes production searched with. That is a property of the storage
    format (no FEN, no move list is stored), not a choice here, and it means the
    fidelity gate below measures harness error AND history loss together -- so a
    HIGH agreement is strong evidence and a low one is ambiguous.
    """
    import glob as _glob

    import zarr

    paths = sorted(_glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"no shards matched {glob_pat!r}")
    paths = paths[-int(n_shards):]
    out: list[ProbePosition] = []
    per = max(1, int(n) // len(paths) + 1)
    for p in paths:
        z = zarr.open(p, mode="r")
        keep = (
            np.asarray(z["has_policy"][:], dtype=bool)
            & np.asarray(z["is_selfplay"][:], dtype=bool)
            & np.asarray(z["is_network_turn"][:], dtype=bool)
        )
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(per, idx.size), replace=False)
        xs = z["x"][:]
        pol = z["policy_target"][:]
        has_reg = np.asarray(z["has_sf_p0_regret"][:], dtype=bool)
        reg = z["sf_p0_regret"][:]
        plies = np.asarray(z["ply_index"][:])
        kls = np.asarray(z["priority_policy_kl"][:], dtype=np.float64)
        for i in sorted(take.tolist()):
            b = decode_board_from_planes(
                np.asarray(xs[i], dtype=np.float32), input_history_encoding=hist,
            )
            if b is None or b.is_game_over():
                continue
            tgt = np.zeros(POLICY_SIZE, dtype=np.float64)
            tgt[COMPACT_TO_FULL_POLICY] = np.asarray(pol[i], dtype=np.float64)
            rcp = None
            if bool(has_reg[i]):
                rcp = np.full(POLICY_SIZE, np.nan, dtype=np.float64)
                rcp[COMPACT_TO_FULL_POLICY] = (
                    np.asarray(reg[i], dtype=np.float64) * SF_OWN_REGRET_CAP_CP
                )
            out.append(ProbePosition(
                fen=b.fen(), board=b, phase=-1, audit=None,
                stored_target=tgt, stored_regret_cp=rcp,
                ply_index=int(plies[i]), priority_policy_kl=float(kls[i]),
            ))
    rng.shuffle(out)  # pyright: ignore[reportArgumentType]
    return out[: int(n)] if n else out


# ---------------------------------------------------------------------------
# Per-position measurement record
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    fen: str
    phase: int
    sims: int
    n_legal: int
    n_cands: int
    prior_move: str
    search_move: str
    improved_move: str
    prior_rank_of_search: int
    prior_rank_of_improved: int
    prior_top1_p: float
    prior_gap_1_2: float
    prior_gap_1_3: float
    root_max_visit: int
    root_q_scale: float
    sigma_span_observed: float
    q_scale_needed: float
    n_zeroed_legal: int
    n_visited: int
    regret_prior: float
    regret_search: float
    regret_improved: float
    regret_rank2: float
    regret_random: float
    # Controls. `regret_prior_bymove` re-derives the prior move's regret by UCI
    # STRING rather than by array position, so it can DISAGREE with
    # `regret_prior` and thereby catch an index misalignment; the previous
    # revision printed a hardcoded 0.00 in that column, which is a control that
    # cannot fail. `*_shuffled` are the same two moves scored against a
    # within-position permutation of the SF labels -- the negative control.
    regret_prior_bymove: float
    regret_prior_shuffled: float
    regret_search_shuffled: float
    covered_prior: bool
    covered_search: bool
    # False when the position source carries no per-move coverage information at
    # all (banked shards). Without it the report cannot tell "0% of moves were
    # inside SF's MultiPV" from "coverage was never tracked", and it printed the
    # former for the latter.
    coverage_known: bool
    q_spearman: float
    prior_spearman: float
    q_spearman_n: int
    # Harness-fidelity gate, shard rows only (NaN / "" elsewhere).
    # Shape of the improved policy -- the stored TRAINING TARGET. `kl_target_prior`
    # is the axis that says how much search moved the distribution at all;
    # entropy is reported beside it because raising sigma necessarily lowers
    # entropy, so a fall there is the mechanism working, not a regression.
    improved_entropy: float = float("nan")
    prior_entropy: float = float("nan")
    kl_target_prior: float = float("nan")
    stored_top1: str = ""
    stored_kl: float = float("nan")
    stored_entropy: float = float("nan")
    ply_index: int = -1
    priority_policy_kl: float = float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks; NaN when either side is constant."""
    n = a.size
    if n < 3:
        return float("nan")
    ra = _avg_rank(a)
    rb = _avg_rank(b)
    sa = ra.std()
    sb = rb.std()
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _avg_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    # Average ties so a flat block (e.g. every unvisited child sharing the
    # mixed value) cannot manufacture an ordering the data does not contain.
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


class _RowSink:
    """Append-only JSONL dump, flushed after every sims rung.

    Banked incrementally on purpose: the first version of this script wrote the
    dump only after the last rung, so an OOM on the 512-sim rung destroyed six
    completed rungs of finished work. Bank the dump, not just the number.
    """

    def __init__(self, path: Path, meta: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", encoding="utf-8")
        self._f.write(json.dumps({"_meta": True, **meta}) + "\n")
        self._f.flush()
        self.n = 0

    def write(self, row: RowResult) -> None:
        self._f.write(json.dumps(asdict(row)) + "\n")
        self.n += 1

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def run_probe(
    positions: list[ProbePosition],
    *,
    checkpoint: str,
    shape: SearchShape,
    sims_list: list[int],
    device: str,
    batch_size: int,
    seed: int,
    add_noise: bool,
    gumbel_scale: float,
    syzygy_path: str | None,
    sink: _RowSink | None = None,
) -> list[RowResult]:
    import torch

    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts.gumbel import GumbelConfig, apply_policy_temp
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    if use_rel:
        raise SystemExit("dynamic-relation checkpoints are out of scope for this probe")
    evaluator = LocalModelEvaluator(model, device=device)

    tb_probe = None
    if syzygy_path:
        from chess_anti_engine.tablebase import SyzygyProbe

        tb_probe = SyzygyProbe(syzygy_path)

    def _cfg(sims: int) -> GumbelConfig:
        return GumbelConfig(
            simulations=int(sims),
            topk=int(shape.topk),
            temperature=0.0,
            policy_temp=float(shape.policy_temp),
            c_scale=float(shape.c_scale),
            c_visit=float(shape.c_visit),
            c_visit_root=float(shape.c_visit_root),
            c_scale_root=float(shape.c_scale_root),
            q_visit_exp=float(shape.q_visit_exp),
            q_visit_exp_root=float(shape.q_visit_exp_root),
            halving_div=int(shape.halving_div),
            add_noise=bool(add_noise),
            gumbel_scale=float(gumbel_scale),
            input_history_encoding=hist,
            input_extra_features=extra,
            policy_encoding=pol_enc,
        )

    boards = [p.board for p in positions]
    rows: list[RowResult] = []
    rng = np.random.default_rng(seed)
    # The controls get their OWN stream. Drawing them from `rng` -- which is the
    # search's Gumbel-noise source -- made the control draws a function of the
    # chunking, and made the search's noise a function of how many controls had
    # been drawn before it. Two independent streams, both seeded.
    ctrl_rng = np.random.default_rng(seed + 991)

    for sims in sims_list:
        cfg = _cfg(sims)
        t0 = time.perf_counter()
        # Leaves per sequential-halving rep grow with the sim budget, so the
        # evaluator batch does too. Shrink the board batch to hold the peak
        # roughly constant -- a 512-sim rung OOMed against the memory cap that
        # 256 sat inside comfortably, and an OOM two hours in costs the run.
        eff_batch = max(4, min(batch_size, 8192 // max(1, sims)))
        if eff_batch != batch_size:
            print(f"[probe] sims={sims}: board batch {batch_size} -> {eff_batch} "
                  f"to bound peak evaluator memory", flush=True)
        for start in range(0, len(boards), eff_batch):
            chunk_pos = positions[start:start + eff_batch]
            chunk = boards[start:start + eff_batch]
            xs = _encode(chunk, hist=hist, extra=extra)
            with torch.no_grad():
                pol_logits, _wdl = evaluator.evaluate_encoded(xs)
            pol_logits = np.asarray(pol_logits, dtype=np.float32)
            # ⚑ TEMPER FIRST, exactly as `gumbel._policy_logits_to_full` does.
            # The search seeds its tree from `logits / policy_temp`, so the
            # improved policy is softmax(log(tempered prior) + sigma(Qbar)).
            # Reading `sigma_span_observed` and `q_scale_needed` off an
            # UNTEMPERED prior silently folds the temperature into both: at
            # T=1.5 the prior logit gaps shrink by 1.5x, so section 3's
            # "P(gap < q_scale)" was overstated and section 6's inversion
            # solved the wrong equation the moment `gumbel_policy_temp` left 1.0
            # -- which it did, on the live yaml, on 2026-08-09.
            pol_logits = np.asarray(
                apply_policy_temp(pol_logits, cfg=cfg), dtype=np.float32,
            )
            if pol_logits.shape[1] != POLICY_SIZE:
                pol_logits = policy_batch_to_full_if_needed(
                    pol_logits, policy_encoding=pol_enc, fill_value=-1e9,
                )

            # Named explicitly rather than through a kwargs dict: `vloss_weight`
            # and `target_batch` are NOT GumbelConfig fields, so no
            # `dataclasses.replace` override surface can reach them, and that is
            # exactly how every arena Elo in the ledger came to be measured at
            # `vloss_weight=0` while production ran 1. Passing them at the call
            # site is the only way they are guaranteed to arrive.
            result = run_gumbel_root_many_c(
                model=None, boards=list(chunk), device=device, rng=rng,
                cfg=cfg, evaluator=evaluator, tb_probe=tb_probe,
                vloss_weight=int(shape.vloss_weight),
                target_batch=int(shape.target_batch),
            )
            probs_b, actions_b, _values, _masks, tree, root_ids = result[:6]

            for j, (pos, board) in enumerate(zip(chunk_pos, chunk, strict=True)):
                row = _score_one(
                    pos=pos, board=board, sims=sims, shape=shape,
                    pol_logits_row=pol_logits[j],
                    search_probs=np.asarray(probs_b[j], dtype=np.float64),
                    search_action=int(actions_b[j]),
                    tree=tree, root_id=int(root_ids[j]),
                    ctrl_rng=ctrl_rng,
                )
                rows.append(row)
                if sink is not None:
                    sink.write(row)
        if sink is not None:
            sink.flush()
        dt = time.perf_counter() - t0
        print(
            f"[probe] sims={sims:>4}  {len(boards)} positions in {dt:6.1f}s "
            f"({dt / max(1, len(boards)) * 1000:.1f} ms/pos)",
            flush=True,
        )
    return rows


def _encode(boards: list[chess.Board], *, hist: str, extra: str) -> np.ndarray:
    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard

    return np.stack([
        encode_cboard(
            CBoard.from_board(b), input_history_encoding=hist, input_extra_features=extra,
        )
        for b in boards
    ])


def _score_one(
    *,
    pos: ProbePosition,
    board: chess.Board,
    sims: int,
    shape: SearchShape,
    pol_logits_row: np.ndarray,
    search_probs: np.ndarray,
    search_action: int,
    tree: object,
    root_id: int,
    ctrl_rng: np.random.Generator,
) -> RowResult:
    ucis, idxs = legal_full_indices(board)
    coverage_known = pos.audit is not None
    if pos.audit is not None:
        # Frozen deep-SF ruler: >=1M nodes, MultiPV >=10, capped at 1000cp, with
        # unlisted moves floored at the worst listed line (an optimistic bound,
        # reported as coverage rather than hidden).
        regrets = move_regrets(pos.audit, ucis)
        covered = {u for u in ucis if u in pos.audit.move_cp}
    elif pos.stored_regret_cp is not None:
        # Production SF label for this exact row. Coarser than the audit set
        # (MultiPV 6 at ~700k nodes) and already carries finalize.py's
        # "(worst+1)/2" default for the moves SF never scored, so `covered`
        # cannot be recovered per move -- report it as unknown rather than
        # claiming full coverage.
        regrets = np.asarray(pos.stored_regret_cp[idxs], dtype=np.float64)
        regrets = np.nan_to_num(regrets, nan=SF_OWN_REGRET_CAP_CP)
        best = float(regrets.min()) if regrets.size else 0.0
        regrets = np.clip(regrets - best, 0.0, SF_OWN_REGRET_CAP_CP)
        covered = set()
    else:
        regrets = np.full(len(ucis), np.nan, dtype=np.float64)
        covered = set()

    logits = pol_logits_row[idxs].astype(np.float64)
    logits -= logits.max()
    e = np.exp(logits)
    prior = e / e.sum()
    order = np.argsort(-prior, kind="stable")
    prior_top = int(order[0])
    gap_1_2 = float(logits[order[0]] - logits[order[1]]) if prior.size > 1 else float("inf")
    gap_1_3 = float(logits[order[0]] - logits[order[2]]) if prior.size > 2 else float("inf")

    # Rank of the played move within the prior's ordering (1 = the prior's own
    # top move). The action ids the search returns are full-4672 ids, the same
    # space `legal_full_indices` produces, so this is a direct lookup.
    pos_in_legal = {int(a): i for i, a in enumerate(idxs.tolist())}
    j_search = pos_in_legal.get(int(search_action), -1)
    rank_of = {int(order[r]): r + 1 for r in range(order.size)}

    imp = search_probs[idxs]
    j_improved = int(np.argmax(imp)) if imp.size else 0

    # sigma span, read straight off the returned policy: the improved policy is
    # softmax(log_prior + sigma(Qbar)) over the SAME legal set, so
    # log(improved) - log(prior) recovers sigma(Qbar) up to an additive
    # constant, and its max-min is the realized sigma span in nats. Measuring
    # it this way rather than recomputing the formula means a plumbing defect
    # between the config and the C search shows up as a DISAGREEMENT with
    # `shape.root_q_scale`, instead of being reproduced by the checker.
    with np.errstate(divide="ignore"):
        d = np.log(np.maximum(imp, 1e-300)) - np.log(np.maximum(prior, 1e-300))
    # A legal move can carry EXACTLY zero improved-policy mass: when the root Q
    # is winning, the search deliberately drops immediate-terminal-draw moves
    # from its candidate set (gumbel.py `_init_board_search_state`). Those rows
    # are not part of the softmax the span is a property of, and including them
    # turns the span into ~693 (log 1e-300) and poisons any mean over it.
    valid = np.isfinite(d) & (imp > 0.0) & (prior > 0.0)
    n_zeroed = int(prior.size - int(valid.sum()))
    sigma_span = float("nan")
    q_needed = float("inf")
    if int(valid.sum()) >= 2:
        dv = d[valid]
        sigma_span = float(dv.max() - dv.min())
        # ── the inversion ────────────────────────────────────────────────────
        # The improved policy's argmax leaves the prior's move for move k
        # exactly when  lambda*(d_k - d_top) > log p_top - log p_k. So the
        # smallest sigma span that would re-rank THIS position is
        #   min over k with d_k > d_top of  gap_1k * span / (d_k - d_top).
        # This inverts the transform instead of sweeping the constant: exact,
        # free, and with no sampling error of its own.
        # ⚑ It holds the TREE fixed. A larger c_scale_root would also change
        # which candidates survive sequential halving, so this answers "could a
        # bigger sigma re-rank the search we ran", not "what would a bigger
        # sigma have searched". The swept arms are the check on that.
        if valid[prior_top] and sigma_span > 0.0:
            d_top = float(d[prior_top])
            for k in range(prior.size):
                if k == prior_top or not valid[k] or float(d[k]) <= d_top:
                    continue
                need = float(logits[prior_top] - logits[k]) * sigma_span / (float(d[k]) - d_top)
                q_needed = min(q_needed, need)

    # Root child visits and completed Q from the C tree.
    visits = np.zeros(idxs.size, dtype=np.int64)
    qvals = np.full(idxs.size, np.nan, dtype=np.float64)
    max_visit = 0
    if root_id >= 0:
        ca, cv, cq = tree.get_children_q(root_id, 0.0)  # pyright: ignore[reportAttributeAccessIssue]
        slot = {int(a): k for k, a in enumerate(np.asarray(ca).tolist())}
        for i, a in enumerate(idxs.tolist()):
            k = slot.get(int(a))
            if k is None:
                continue
            visits[i] = int(cv[k])
            if visits[i] > 0:
                qvals[i] = float(cq[k])
        max_visit = int(np.asarray(cv).max(initial=0))

    # Q quality: does the search's own value estimate rank moves the way deep SF
    # does? Restricted to VISITED children -- unvisited ones all carry the same
    # mixed value, so including them would inject a large tied block and pull
    # the correlation toward the prior's ordering for free.
    vis = visits > 0
    if vis.sum() >= 3:
        q_rho = _spearman(qvals[vis], -regrets[vis])
        p_rho = _spearman(prior[vis], -regrets[vis])
    else:
        q_rho = float("nan")
        p_rho = float("nan")

    rank2 = int(order[1]) if order.size > 1 else prior_top
    rnd = int(ctrl_rng.integers(0, prior.size))

    # ── controls ─────────────────────────────────────────────────────────────
    # `prior` scored by UCI STRING through `move_regrets`, i.e. through a
    # different lookup than the array indexing every other number here uses.
    # This is the arm the previous revision printed as a literal 0.00: an
    # identity, and therefore a control that could not fail. Read this way it
    # CAN fail -- it is exactly the check that `ucis[i]` and `regrets[i]`
    # describe the same move.
    if pos.audit is not None:
        regret_prior_bymove = float(move_regrets(pos.audit, [ucis[prior_top]])[0])
    else:
        regret_prior_bymove = float("nan")

    # ⚑ THE NEGATIVE CONTROL. Permute the SF labels WITHIN this position and
    # re-score the same two moves. Within-position is the right scope: it keeps
    # the legal-set size, the coverage and the marginal regret distribution
    # identical and destroys only the move->quality association, so the delta
    # must go to 0. Permuting the derived DELTAS across positions -- what an
    # earlier revision did and then removed as unfalsifiable -- is a different
    # operation: subtracting the same prior vector from a permutation of the
    # search regrets leaves the mean algebraically unchanged. Permuting the
    # LABELS does not. [[shuffle_the_labels_negative_control]]
    perm = ctrl_rng.permutation(regrets.size)
    regrets_shuf = regrets[perm]
    regret_prior_shuffled = float(regrets_shuf[prior_top])
    regret_search_shuffled = (
        float(regrets_shuf[j_search]) if j_search >= 0 else float("nan")
    )

    # Harness-fidelity gate: how close is this re-run to the improved policy
    # production actually banked for this row?
    # Shape of the target this search would store, and how far it moved the
    # prior. Both over the legal set only, which is the support the loss sees.
    pv = prior[prior > 0.0]
    prior_entropy = float(-(pv * np.log(pv)).sum())
    iv = imp[imp > 0.0]
    improved_entropy = float(-(iv * np.log(iv)).sum())
    m_kl = imp > 0.0
    kl_target_prior = float(
        (imp[m_kl] * (np.log(imp[m_kl]) - np.log(np.maximum(prior[m_kl], 1e-300)))).sum()
    )

    stored_top1 = ""
    stored_kl = float("nan")
    stored_entropy = float("nan")
    if pos.stored_target is not None:
        st = np.asarray(pos.stored_target[idxs], dtype=np.float64)
        s = st.sum()
        if s > 0:
            st = st / s
            stored_top1 = ucis[int(np.argmax(st))]
            m = st > 0
            stored_kl = float((st[m] * (np.log(st[m]) - np.log(np.maximum(imp[m], 1e-12)))).sum())
            stored_entropy = float(-(st[m] * np.log(st[m])).sum())

    return RowResult(
        fen=pos.fen,
        phase=int(pos.phase),
        sims=int(sims),
        n_legal=int(prior.size),
        n_cands=int(min(shape.topk, prior.size, max(2, (sims + 1) // 2)) if sims > 1 else 1),
        prior_move=ucis[prior_top],
        search_move=ucis[j_search] if j_search >= 0 else "",
        improved_move=ucis[j_improved] if imp.size else "",
        prior_rank_of_search=rank_of.get(j_search, -1),
        prior_rank_of_improved=rank_of.get(j_improved, -1),
        prior_top1_p=float(prior[prior_top]),
        prior_gap_1_2=gap_1_2,
        prior_gap_1_3=gap_1_3,
        root_max_visit=max_visit,
        root_q_scale=float(shape.root_q_scale(float(max_visit))),
        sigma_span_observed=sigma_span,
        q_scale_needed=q_needed,
        n_zeroed_legal=n_zeroed,
        n_visited=int(vis.sum()),
        regret_prior=float(regrets[prior_top]),
        regret_search=float(regrets[j_search]) if j_search >= 0 else float("nan"),
        regret_improved=float(regrets[j_improved]) if imp.size else float("nan"),
        regret_rank2=float(regrets[rank2]),
        regret_random=float(regrets[rnd]),
        regret_prior_bymove=regret_prior_bymove,
        regret_prior_shuffled=regret_prior_shuffled,
        regret_search_shuffled=regret_search_shuffled,
        covered_prior=ucis[prior_top] in covered,
        covered_search=(ucis[j_search] in covered) if j_search >= 0 else False,
        coverage_known=coverage_known,
        q_spearman=q_rho,
        prior_spearman=p_rho,
        q_spearman_n=int(vis.sum()),
        improved_entropy=improved_entropy,
        prior_entropy=prior_entropy,
        kl_target_prior=kl_target_prior,
        stored_top1=stored_top1,
        stored_kl=stored_kl,
        stored_entropy=stored_entropy,
        ply_index=int(pos.ply_index),
        priority_policy_kl=float(pos.priority_policy_kl),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    """Mean and a 95% normal CI. Returns (mean, lo, hi); NaNs dropped."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if x.size < 2:
        return m, float("nan"), float("nan")
    se = float(x.std(ddof=1) / math.sqrt(x.size))
    return m, m - 1.96 * se, m + 1.96 * se


def _sign_flip_p(deltas: np.ndarray, rng: np.random.Generator, n_perm: int = 10000) -> float:
    """One-sided paired sign-flip permutation p-value for `mean(deltas) < 0`.

    Replaces a shuffle that could not fail. Under "search is no better than the
    prior" each position's delta is sign-symmetric, so flipping signs at random
    is a valid null; the observed mean is compared against that null's left
    tail. Zero deltas (the ties, which are the majority) contribute nothing
    either way, exactly as they should.
    """
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    obs = float(d.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    return float(((null <= obs).sum() + 1) / (n_perm + 1))


def _rate_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def report(rows: list[RowResult], shape: SearchShape, *, rng: np.random.Generator) -> str:
    out: list[str] = []
    sims_vals = sorted({r.sims for r in rows})

    out.append("")
    out.append("=" * 100)
    out.append(f"SEARCH SHAPE: {shape.label}")
    out.append(f"  provenance     : {shape.provenance}")
    out.append(
        f"  topk={shape.topk} c_scale={shape.c_scale} c_visit={shape.c_visit} "
        f"halving_div={shape.halving_div} vloss_weight={shape.vloss_weight}"
    )
    out.append(
        f"  ROOT transform : {shape.root_transform_name()}  "
        f"(c_scale_root={shape.c_scale_root} c_visit_root={shape.c_visit_root} "
        f"q_visit_exp_root={shape.q_visit_exp_root})"
    )
    out.append("=" * 100)

    out.append("")
    out.append("1. CHANGE RATE  (does search move away from argmax(prior)?)")
    out.append(
        f"{'sims':>6} {'n':>5} {'cands':>6} {'played!=prior':>16} {'95% CI':>18} "
        f"{'improved!=prior':>17} {'mean prior rank':>16} {'rank>3':>8}"
    )
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s]
        n = len(sub)
        chg = sum(1 for r in sub if r.search_move != r.prior_move)
        chg_i = sum(1 for r in sub if r.improved_move != r.prior_move)
        p, lo, hi = _rate_ci(chg, n)
        pi, _, _ = _rate_ci(chg_i, n)
        ranks = np.array([r.prior_rank_of_search for r in sub if r.prior_rank_of_search > 0])
        deep = float((ranks > 3).mean()) if ranks.size else float("nan")
        cands = float(np.mean([r.n_cands for r in sub]))
        out.append(
            f"{s:>6} {n:>5} {cands:>6.1f} {p:>15.1%} [{lo:>6.1%},{hi:>6.1%}] "
            f"{pi:>16.1%} {ranks.mean() if ranks.size else float('nan'):>16.2f} {deep:>7.1%}"
        )

    out.append("")
    out.append("1b. TARGET SHAPE  (the improved policy IS the stored training target)")
    out.append("   KL(target||prior) is how far search moved the distribution at all.")
    out.append("   Target entropy FALLS as sigma rises -- that is the mechanism working,")
    out.append("   not a regression, so the two must be read together and never traded.")
    out.append(
        f"{'sims':>6} {'KL(target||prior)':>18} {'95% CI':>20} "
        f"{'H(target)':>10} {'H(prior)':>10} {'dH':>8}"
    )
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s]
        kl = np.array([r.kl_target_prior for r in sub])
        ht = np.array([r.improved_entropy for r in sub])
        hp = np.array([r.prior_entropy for r in sub])
        m, lo, hi = _mean_ci(kl)
        out.append(
            f"{s:>6} {m:>18.4f} [{lo:>8.4f},{hi:>8.4f}] "
            f"{np.nanmedian(ht):>10.4f} {np.nanmedian(hp):>10.4f} "
            f"{np.nanmedian(ht) - np.nanmedian(hp):>8.4f}"
        )

    out.append("")
    out.append("2. CHANGE QUALITY  (deep-SF cp regret; NEGATIVE delta = search is BETTER)")
    out.append("   Scored on the CHANGED positions only. Ties are excluded by construction.")
    out.append(
        f"{'sims':>6} {'n_chg':>6} {'delta cp':>10} {'95% CI':>20} "
        f"{'better':>8} {'worse':>8} {'equal':>8} {'covered':>8}"
    )
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s and r.search_move != r.prior_move]
        if not sub:
            out.append(f"{s:>6} {0:>6}")
            continue
        d = np.array([r.regret_search - r.regret_prior for r in sub])
        m, lo, hi = _mean_ci(d)
        better = float((d < 0).mean())
        worse = float((d > 0).mean())
        equal = float((d == 0).mean())
        # ⚑ Tri-state. Shard rows carry no per-move coverage at all (production's
        # SF label already has finalize.py's fill baked in), so `covered_*` is
        # False there for want of information. Printing that as "0.0%" reads as
        # "no move was inside SF's MultiPV", which is a measurement nobody made.
        known = [r for r in sub if r.coverage_known]
        cov_s = (
            f"{float(np.mean([r.covered_prior and r.covered_search for r in known])):>7.1%}"
            if known else "    n/a"
        )
        out.append(
            f"{s:>6} {len(sub):>6} {m:>10.2f} [{lo:>8.2f},{hi:>8.2f}] "
            f"{better:>7.1%} {worse:>7.1%} {equal:>7.1%} {cov_s}"
        )

    out.append("")
    out.append("   ALL positions (changed + unchanged) -- the number that actually moves play:")
    out.append(f"{'sims':>6} {'mean regret prior':>18} {'mean regret search':>19} {'delta':>10} {'95% CI':>20}")
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s]
        rp = np.array([r.regret_prior for r in sub])
        rs = np.array([r.regret_search for r in sub])
        d = rs - rp
        m, lo, hi = _mean_ci(d)
        out.append(
            f"{s:>6} {np.nanmean(rp):>18.2f} {np.nanmean(rs):>19.2f} {m:>10.2f} "
            f"[{lo:>8.2f},{hi:>8.2f}]"
        )

    out.append("")
    out.append("3. SIGMA HEADROOM  (can the value transform re-rank the prior at all?)")
    out.append("   sigma spans exactly q_scale nats across the legal moves, so a top-1/top-2")
    out.append("   prior LOGIT GAP above q_scale is structurally unreachable by any Qbar.")
    out.append("   `observed` is read off the returned policy and MUST equal `q_scale`;")
    out.append("   a mismatch means the shape printed above is not the search that ran.")
    out.append(
        f"{'sims':>6} {'max_visit':>10} {'q_scale':>9} {'observed':>9} {'maxdev':>8} "
        f"{'med gap12':>10} {'P(gap12<q)':>11} {'med gap13':>10} {'P(gap13<q)':>11}"
    )
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s]
        mv = float(np.mean([r.root_max_visit for r in sub]))
        qs = np.array([r.root_q_scale for r in sub])
        obs = np.array([r.sigma_span_observed for r in sub])
        ok = np.isfinite(obs)
        dev = float(np.abs(obs[ok] - qs[ok]).max()) if ok.any() else float("nan")
        g12 = np.array([r.prior_gap_1_2 for r in sub])
        g13 = np.array([r.prior_gap_1_3 for r in sub])
        f12 = float(np.mean(g12[np.isfinite(g12)] < qs[np.isfinite(g12)]))
        f13 = float(np.mean(g13[np.isfinite(g13)] < qs[np.isfinite(g13)]))
        out.append(
            f"{s:>6} {mv:>10.1f} {qs.mean():>9.3f} {np.median(obs[ok]):>9.3f} {dev:>8.2e} "
            f"{np.median(g12[np.isfinite(g12)]):>10.3f} {f12:>10.1%} "
            f"{np.median(g13[np.isfinite(g13)]):>10.3f} {f13:>10.1%}"
        )
    # ⚑ A VERDICT, not a number to admire. The text above says a mismatch means
    # "the shape printed above is not the search that ran" -- and the previous
    # revision then printed `maxdev` with no threshold and no line saying
    # whether it had been cleared, which is a gate that cannot fail. The
    # comparison is restricted to rows where NO legal move was pruned to zero
    # improved-policy mass: on a pruned row the min-max normalisation's minimum
    # is not in the softmax, so the observed span is legitimately below q_scale.
    clean = [r for r in rows if r.n_zeroed_legal == 0 and math.isfinite(r.sigma_span_observed)]
    if not clean:
        out.append(
            "   SHAPE GATE: NOT EVALUATED -- no row had a full un-pruned softmax, "
            "so the printed q_scale is UNVERIFIED against the search that ran"
        )
    else:
        rel = max(
            abs(r.sigma_span_observed - r.root_q_scale) / max(1e-12, abs(r.root_q_scale))
            for r in clean
        )
        out.append(
            f"   SHAPE GATE: realized sigma span vs the printed q_scale, max relative "
            f"deviation {rel:.2e} over {len(clean)} un-pruned rows -- "
            + ("PASS" if rel <= 1e-3 else
               "FAIL: the search that ran is NOT the shape in the header, so every "
               "q_scale-relative statement in sections 3 and 6 is void")
        )
    zl = sum(r.n_zeroed_legal for r in rows)
    out.append(
        f"   legal moves given exactly zero improved-policy mass: {zl} across "
        f"{len(rows)} rows (immediate-terminal-draw pruning in a won root; by design)"
    )

    out.append("")
    out.append("4. Qbar QUALITY  (Spearman vs deep-SF cp over VISITED root children)")
    out.append("   If more sims does not raise rho, deeper search is averaging noise.")
    out.append(f"{'sims':>6} {'n rows':>7} {'mean visited':>13} {'rho(Q)':>9} {'95% CI':>18} {'rho(prior)':>11}")
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s and math.isfinite(r.q_spearman)]
        if not sub:
            out.append(f"{s:>6} {0:>7}")
            continue
        q = np.array([r.q_spearman for r in sub])
        p = np.array([r.prior_spearman for r in sub])
        m, lo, hi = _mean_ci(q)
        nv = float(np.mean([r.n_visited for r in sub]))
        out.append(
            f"{s:>6} {len(sub):>7} {nv:>13.1f} {m:>9.3f} [{lo:>7.3f},{hi:>7.3f}] "
            f"{np.nanmean(p):>11.3f}"
        )

    out.append("")
    out.append("5. CONTROLS  (mean cp regret vs the prior's move, ALL positions)")
    out.append("   `rank2`  MUST be clearly positive -- it prices ONE rank of prior, i.e.")
    out.append("            the resolution the ruler has for a move-choice change at all.")
    out.append("   `random` MUST be strongly positive -- if a uniformly random legal move")
    out.append("            does not score worse than search, the ruler is not measuring")
    out.append("            move quality and nothing above this line means anything.")
    out.append("   `shuffled` IS THE NEGATIVE CONTROL: the SAME search and the SAME prior")
    out.append("            scored against a within-position permutation of the SF labels.")
    out.append("            It MUST collapse to ~0 with a CI covering 0. A `shuffled` arm")
    out.append("            that reproduces the search's gain means the gain is not coming")
    out.append("            from move quality, and every number above is void.")
    out.append("   `p(sign)` is a paired sign-flip permutation test on the search delta:")
    out.append("            under the null 'search is no better than the prior' the sign of")
    out.append("            each position's delta is exchangeable. 10k flips. It is a NULL")
    out.append("            TEST on the real data, not a negative control.")
    out.append(
        f"{'sims':>6} {'rank2':>19} {'random':>19} {'shuffled':>19} "
        f"{'search':>19} {'p(sign)':>9}"
    )
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s]
        rp = np.array([r.regret_prior for r in sub])
        d_r2 = np.array([r.regret_rank2 for r in sub]) - rp
        d_rn = np.array([r.regret_random for r in sub]) - rp
        d_sh = (
            np.array([r.regret_search_shuffled for r in sub])
            - np.array([r.regret_prior_shuffled for r in sub])
        )
        d_se = np.array([r.regret_search for r in sub]) - rp
        m2, l2, h2 = _mean_ci(d_r2)
        mr, lr, hr = _mean_ci(d_rn)
        mh, lh, hh = _mean_ci(d_sh)
        ms, ls, hs = _mean_ci(d_se)
        out.append(
            f"{s:>6} {m2:>8.2f}[{l2:>5.1f},{h2:>5.1f}] "
            f"{mr:>8.2f}[{lr:>5.1f},{hr:>5.1f}] "
            f"{mh:>8.2f}[{lh:>5.1f},{hh:>5.1f}] "
            f"{ms:>8.2f}[{ls:>5.1f},{hs:>5.1f}] "
            f"{_sign_flip_p(d_se, rng):>9.4f}"
        )
    out.append("")
    # The alignment control. It is a PASS/FAIL line and not a number to admire:
    # the whole cp table is read out of `regrets[i]` while every move is named
    # by `ucis[i]`, and nothing else in this report would notice if those two
    # indexings disagreed.
    bym = np.array([r.regret_prior_bymove for r in rows], dtype=np.float64)
    byi = np.array([r.regret_prior for r in rows], dtype=np.float64)
    ok = np.isfinite(bym) & np.isfinite(byi)
    if not ok.any():
        out.append("   ALIGNMENT control: n/a (no position source carries per-move SF labels)")
    else:
        dev = float(np.abs(bym[ok] - byi[ok]).max())
        out.append(
            f"   ALIGNMENT control: prior regret by UCI string vs by array index, "
            f"max deviation {dev:.6f} over {int(ok.sum())} rows -- "
            + ("PASS" if dev <= 1e-9 else "FAIL: `ucis` and `regrets` are MISALIGNED "
                                          "and every cp number above is off-by-move")
        )
    out.append("")
    out.append("   ⚑ The previous revision printed `prior` as a literal 0.00 in this table.")
    out.append("   That is an identity, not a measurement -- a control that cannot fail. It")
    out.append("   is replaced by the ALIGNMENT control above, which can. Separately, a")
    out.append("   shuffle that permuted the search DELTAS across positions was tried and")
    out.append("   removed: subtracting the same prior vector from a permutation of the")
    out.append("   search regrets leaves the MEAN algebraically unchanged. Permuting the")
    out.append("   LABELS within a position -- the `shuffled` arm above -- is a different")
    out.append("   operation and does collapse.")

    out.append("")
    out.append("6. SIGMA INVERSION  (what q_scale would it take to re-rank the prior?)")
    out.append("   Per position, the smallest sigma span at which the improved policy's")
    out.append("   argmax leaves the prior's move, given THIS search's visits. `frac<=q`")
    out.append("   must reproduce the improved-policy change rate in section 1 -- that is")
    out.append("   the inversion's internal consistency check.")
    out.append(
        f"{'sims':>6} {'q_scale':>9} {'p25':>9} {'median':>9} {'p75':>9} {'frac<=q':>9} "
        f"{'c_scale_root for median':>24}"
    )
    for s in sims_vals:
        sub = [r for r in rows if r.sims == s]
        need = np.array([r.q_scale_needed for r in sub])
        qs = float(np.mean([r.root_q_scale for r in sub]))
        mv = float(np.mean([r.root_max_visit for r in sub]))
        fin = need[np.isfinite(need)]
        frac = float(np.mean(need <= np.array([r.root_q_scale for r in sub])))
        if fin.size == 0:
            out.append(f"{s:>6} {qs:>9.3f}  (no position is reachable at any scale)")
            continue
        med = float(np.median(fin))
        # ⚑ ONLY DEFINED FOR A LINEAR ROOT. The span is
        # c_scale_root*(c_visit+max_visit) there, so the constant that puts the
        # median position on the boundary is a direct division -- but under a
        # LOG or POWER root the same span comes from a different constant
        # entirely, and the previous revision printed this division anyway for
        # all five shapes, three of which are not linear. It also divided by
        # the hardcoded live-snapshot `c_visit` rather than the running shape's.
        if shape.root_transform_name() == "LINEAR":
            implied = f"{med / max(1e-9, (shape.c_visit + mv)):24.4f}"
        else:
            implied = f"{'n/a (' + shape.root_transform_name() + ' root)':>24}"
        out.append(
            f"{s:>6} {qs:>9.3f} {np.percentile(fin, 25):>9.3f} {med:>9.3f} "
            f"{np.percentile(fin, 75):>9.3f} {frac:>8.1%} {implied}"
        )
    out.append("   NOTE: 'median' is the scale that would flip HALF the positions. That is")
    out.append("   a reference point, not a recommendation -- flipping half the moves is a")
    out.append("   different search, and its VALUE has to be measured, not assumed.")

    # ⚑ THE FIDELITY GATE'S ABSENCE IS ITSELF A RESULT. A shard run that scored
    # nothing against the stored target silently dropped this whole section in
    # the previous revision, so the report read exactly like an audit-set run --
    # and the text below calls this the thing "nothing else means anything"
    # without it. Say so instead of vanishing.
    gate = [r for r in rows if r.stored_top1]
    if not gate and any(r.stored_kl == r.stored_kl for r in rows):
        out.append("")
        out.append("0. HARNESS FIDELITY GATE: NOT EVALUATED on any row.")
        out.append("   This run was NOT checked against a target production stored, so no")
        out.append("   absolute number in it may be quoted -- only deltas between its arms.")
    if gate:
        out.append("")
        out.append("0. HARNESS FIDELITY GATE  (shard rows only: this re-run vs the STORED target)")
        out.append("   Nothing else in this table means anything if the re-run is not the")
        out.append("   production path. Disagreement here bounds harness error AND the history")
        out.append("   loss from decoding a board out of the stored planes, TOGETHER.")
        out.append("   The decisive scalar is H(stored): production's own")
        out.append("   `data_policy_entropy` runs 0.889-0.979 and the banked-shard median is")
        out.append("   ~0.924. If H(mine) at the production shape does not land in that band,")
        out.append("   this rig is NOT the production path and only DELTAS between its own")
        out.append("   arms may be quoted -- never an absolute.")
        out.append(
            f"{'sims':>6} {'n':>6} {'top1 agree':>11} {'KL(stored||mine)':>17} "
            f"{'H(stored)':>10} {'H(mine)':>9} {'prior==stored':>14}"
        )
        for s in sims_vals:
            sub = [r for r in gate if r.sims == s]
            if not sub:
                continue
            agree = float(np.mean([r.improved_move == r.stored_top1 for r in sub]))
            pagree = float(np.mean([r.prior_move == r.stored_top1 for r in sub]))
            # An all-NaN column is a real possibility (a stored target that
            # sums to zero on every row), and numpy's nan-aggregates warn and
            # return NaN for it. Reduce over the finite entries explicitly so
            # the cell reads `nan` rather than emitting a RuntimeWarning that
            # nobody sees in a piped run.
            def _agg(vals: list[float], fn: object) -> float:
                arr = np.asarray(vals, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                return float(fn(arr)) if arr.size else float("nan")  # pyright: ignore[reportCallIssue]

            kl_m = _agg([r.stored_kl for r in sub], np.mean)
            hs_m = _agg([r.stored_entropy for r in sub], np.median)
            hm_m = _agg([r.improved_entropy for r in sub], np.median)
            out.append(
                f"{s:>6} {len(sub):>6} {agree:>10.1%} {kl_m:>17.4f} "
                f"{hs_m:>10.4f} {hm_m:>9.4f} {pagree:>13.1%}"
            )
        # ⚑ THE BAND IS A GATE, NOT PROSE. The paragraph above states the rule
        # ("if H(mine) does not land in 0.889-0.979 this rig is NOT the
        # production path") and the previous revision then printed the numbers
        # and left the comparison to the reader -- so the rule could not fail.
        # Evaluated at the LARGEST sims rung, which is the one a production-shape
        # run is claiming to reproduce; smaller rungs are deliberately not
        # production and are exempt.
        top = max(sims_vals)
        at_top = [r for r in gate if r.sims == top]
        if at_top:
            _h = np.asarray([r.improved_entropy for r in at_top], dtype=np.float64)
            _h = _h[np.isfinite(_h)]
            h_mine = float(np.median(_h)) if _h.size else float("nan")
            in_band = DATA_POLICY_ENTROPY_BAND[0] <= h_mine <= DATA_POLICY_ENTROPY_BAND[1]
            out.append(
                f"   FIDELITY GATE @ {top} sims: median H(mine) {h_mine:.4f} vs "
                f"production's data_policy_entropy band "
                f"[{DATA_POLICY_ENTROPY_BAND[0]:.3f},{DATA_POLICY_ENTROPY_BAND[1]:.3f}] -- "
                + ("PASS" if in_band else
                   "FAIL: this rig is NOT reproducing the production path, so ONLY "
                   "deltas between its own arms may be quoted and every absolute "
                   "in this report is void")
            )
            out.append(
                "   ⚑ The band is a property of the PRODUCTION shape. A run at any "
                "other shape is EXPECTED to leave it, and a PASS there means nothing."
            )

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--position-source", choices=("audit", "shards"), default="audit",
                    help="audit = frozen deep-SF set; shards = real production positions "
                         "decoded out of the banked replay shards (adds the fidelity gate)")
    ap.add_argument("--shards-glob", default=None,
                    help="e.g. 'runs/pbt2_small/replay/train_trial_379f6_*/replay_shards/*.zarr'")
    ap.add_argument("--shards-last", type=int, default=40,
                    help="sample from the newest N matching shards")
    ap.add_argument("--shape", required=True, choices=sorted(SHAPES),
                    help="REQUIRED: there is no safe default; live/main/play disagree. "
                         "The live_selfplay_* entries are DATED SNAPSHOTS, not live "
                         "readings -- pair them with --shape-yaml to measure today's shape")
    ap.add_argument("--shape-yaml", type=Path, default=None,
                    help="read gumbel_topk / gumbel_c_scale / gumbel_policy_temp / "
                         "gumbel_vloss_weight off a REAL config file (e.g. the live "
                         "yaml, which is not in this repo) instead of trusting the "
                         "hardcoded snapshot; every other field keeps the "
                         "GumbelConfig default network_turn.py leaves unset")
    ap.add_argument("--sims", default="1,8,32,64,128,256")
    ap.add_argument("--positions", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="keep under 64: gumbel_c pipelines above that and returns "
                         "invalid root ids, which this probe reads the tree through")
    ap.add_argument("--gpu-mem-fraction", type=float, default=None)
    ap.add_argument("--noise", action="store_true", help="root Gumbel noise ON")
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--syzygy-path", default=None)
    ap.add_argument("--override", default=None,
                    help="comma-separated SearchShape field overrides, e.g. c_scale_root=3.0")
    ap.add_argument("--out", type=Path, default=None, help="JSONL dump of every row")
    args = ap.parse_args()

    if args.batch_size >= 64:
        raise SystemExit(
            "--batch-size must stay under 64: run_gumbel_root_many_c switches to "
            "its 2-tree pipeline at n_boards>=64 and returns root_ids of -1, so "
            "the per-child visit/Q readout this probe needs would be silently empty"
        )

    sims_list = [int(s) for s in str(args.sims).split(",") if s.strip()]
    shape = SHAPES[args.shape]
    if args.shape_yaml is not None:
        if not args.shape_yaml.exists():
            raise SystemExit(f"--shape-yaml {args.shape_yaml} does not exist")
        shape = shape_from_yaml(args.shape_yaml, base=shape)
    shape = apply_overrides(shape, args.override)

    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        import torch

        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_mem_fraction), torch.device(args.device).index or 0,
        )

    rng = np.random.default_rng(args.seed)
    if args.position_source == "shards":
        if not args.shards_glob:
            raise SystemExit("--position-source shards requires --shards-glob")
        hist = _checkpoint_history_encoding(args.checkpoint)
        print(f"[probe] shard decode uses input_history_encoding={hist!r} "
              f"(read from the checkpoint, not assumed)")
        positions = shard_positions(
            args.shards_glob, int(args.positions), rng,
            hist=hist, n_shards=int(args.shards_last),
        )
        src = f"{len(positions)} production rows from {args.shards_glob}"
    else:
        positions = audit_positions(args.audit_set, int(args.positions), rng)
        src = f"{len(positions)} audit positions from {args.audit_set}"

    print(f"[probe] git {git_sha()}  checkpoint {args.checkpoint}")
    print(f"[probe] {src}")
    print(f"[probe] shape={args.shape}  noise={'ON' if args.noise else 'OFF'} "
          f"scale={args.gumbel_scale}  sims={sims_list}")

    sink = None
    if args.out:
        sink = _RowSink(args.out, {
            "git": git_sha(), "checkpoint": str(args.checkpoint),
            "shape": asdict(shape), "noise": bool(args.noise),
            "gumbel_scale": float(args.gumbel_scale), "seed": int(args.seed),
            "position_source": args.position_source, "source": src,
            "n_positions": len(positions),
        })
    try:
        rows = run_probe(
            positions,
            checkpoint=args.checkpoint,
            shape=shape,
            sims_list=sims_list,
            device=args.device,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            add_noise=bool(args.noise),
            gumbel_scale=float(args.gumbel_scale),
            syzygy_path=args.syzygy_path,
            sink=sink,
        )
    finally:
        if sink is not None:
            sink.close()
            print(f"[probe] banked {sink.n} rows to {args.out}")

    print(report(rows, shape, rng=np.random.default_rng(args.seed + 1)))


if __name__ == "__main__":
    main()
