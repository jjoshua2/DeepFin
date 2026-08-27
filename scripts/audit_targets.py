#!/usr/bin/env python3
"""Score training-target candidates directly against the frozen audit set.

For every deep-labeled audit position (scripts/build_audit_set.py) this
computes candidate POLICY distributions:

  a) net raw policy — single batched forward of --checkpoint
  b) net + Gumbel search at PLAY (UCI/TCEC) search settings, --sims
  c) the SF MultiPV soft target (--sf-soft-nodes / --sf-soft-multipv,
     low=500k / high=2M via --sf-effort, matching the 500k production
     teacher; default 500k), built with the production sf_policy_temp /
     label-smoothing / cp-logistic params from --config
  d) the production TRAINING target — the RL selfplay search from --config at
     full sims, retempered with the production move-selection temperature
     (policy_t IS the visit distribution at that temperature — see CLAUDE.md
     head table). This is the WHOLE stored policy corpus.
  e) the same search at the playout-capped fast sims, for reference only.
     Playout-capped plies carry NO policy target: finalize.py drops them, and
     with record_fast_ply_value they become value-only rows whose MAIN policy
     head is masked. Never average (e) into (d) — that invents a mixture the
     pipeline does not store.

(b) and (d)/(e) are DIFFERENT SEARCHES and must not be substituted for one
another. RL selfplay keeps `gumbel_c_scale` 0.1 with the legacy LINEAR root
value-transform; UCI/TCEC play uses c_scale 0.025 with the LOG root
(c_scale_root 7.0). Both are deliberate and separately tuned — at the 256-sim
selfplay budget 0.1 measured 0.688 puzzle accuracy against 0.598 for 0.025 —
so one config cannot stand in for the other. Before 2026-07-25 this script
built ONE search from the PLAY defaults and labelled it "production training
target", which put a play-path number next to the SF soft target in the
headline that prices SF's MultiPV CPU bill.

and scores each as expected deep-SF regret (cp) of a move sampled from the
distribution, plus top-1 regret — reported per phase and per source.

  !! THE POLICY TABLE CARRIES THE SAME SAME-MATERIAL BIAS AS THE VALUE TABLE
  BELOW, and for the same reason. The ruler is deep SF; candidate (c) is
  shallow SF. (c) is therefore graded against a DEEPER VERSION OF ITSELF while
  (a)/(b)/(d)/(e) are graded against a different engine family. Wherever SF is
  decisive the two SF objects agree by construction, so (c) starts with a margin
  that has nothing to do with being a better TEACHER.
  Use this table to rank candidates of the SAME material against each other —
  search shape vs search shape, checkpoint vs checkpoint — and to detect a
  candidate that has DRIFTED or BROKEN. Do NOT read "(c) beats (d)" as "training
  on (c) would beat training on (d)": that inference needs a ruler the SF target
  did not help write. This was read the wrong way on 2026-07-27 (the "the
  external teacher is 2x better on top-1" framing) and the correction is what
  this warning exists to prevent.

For VALUE it scores, against the deep-SF native WDL (and separately against
full-strength game outcomes on the positions that have them):

  i)   cp->logistic transform of the shallow SF eval (production slope/width)
  ii)  shallow SF native WDL
  iii) the production blend (sf_wdl_frac / search_wdl_frac from --config;
       the game-outcome component only contributes on outcome-labeled rows)
  iv)  search root WDL — the RL search's root Q from (d), reconstructed the
       way selfplay stores it: the root network's OWN draw mass is preserved
       and only the remaining mass is split around Q (see
       _search_wdl_like_selfplay). Not `1 - |Q|`, which is a different
       distribution and a different target.

  !! THIS VALUE TABLE IS A CALIBRATION RULER, NOT A TARGET-QUALITY RULER. It
  ranks candidates by AGREEMENT WITH DEEP SF, and (ii) shallow SF native WDL
  will normally win it for a reason that has nothing to do with being a good
  teacher: it is the SAME KIND OF OBJECT as the reference. Both are Stockfish,
  so wherever SF is decisive they go one-hot together and the ECE collapses.
  Measured 2026-07-27 on 2000 audit positions: (ii) Brier 0.0348 / ECE 0.0069
  vs the production blend (iii) 0.0484 / 0.0868 — a 12x ECE gap that reads as
  "switch to native WDL" and is NOT that.

  Production deliberately does the opposite (`sf_wdl_use_cp_logistic: true`)
  because SF's UCI_ShowWDL is **~72% one-hot**, and a one-hot value target
  teaches over-confidence — the failure actually observed in play (2026-06-28
  loss: the net evaluated +557 while the position was lost by ~300, an ~860cp
  sign error). The cp-logistic's high ECE **against a deep-SF ruler IS the
  deliberate softness**, not a defect; see CLAUDE.md ("the cp-logistic label is
  deliberately soft; don't chase value sharpness against a deep-SF ruler") and
  the WDL blend section of docs/model_heads.md.

  So: use this table to detect a candidate that has DRIFTED or BROKEN, never to
  pick the value target. Reading it as a target ranking was attempted and
  retracted on 2026-07-27.

as Brier score and expected calibration error.

  !! `--input-encoding` CHANGES THE RULER, AND ONLY FOR ROW (a).
  `fen_only` (DEFAULT, bit-identical to every historical run) builds the net's
  input from `chess.Board(fen)` with an empty move stack, which under the
  production 175-plane encoding leaves 93 planes structurally zero and pins the
  colour flag to the wrong value on ~51% of rows. `stored` is audit-v2: row (a)
  is scored on the real production input recovered by
  scripts/match_audit_rows.py.
  Rows (b)/(d)/(e) are SEARCHES. They build their own encodings inside the C
  tree from the board, so no stored root can reach them, and they stay
  `fen_only` under both settings; row (c) is pure Stockfish and has no encoding
  at all. That is stated on every table row rather than left to be inferred —
  a report whose header said "stored" while four of five rows were FEN-only
  would be exactly the defect this flag exists to remove. The root network WDL
  behind value row (iv) is likewise kept on the SEARCH's encoding, so (iv) is
  never a hybrid of a stored forward and a FEN-only search.
  ⚑ A RULER CHANGE INVALIDATES ITS RECORDS: never put a `stored` (a) next to a
  `fen_only` (a) in one table, trend or threshold.

Shallow SF results are cached to <audit>.shallow_sf.jsonl (append-only,
resumable) so reruns against new checkpoints don't repay the CPU bill.
⚑⚑ THE CACHE IS KEYED BY ENGINE IDENTITY AS WELL AS (nodes, multipv). It used
to be keyed by (nodes, multipv) ALONE, which meant two arms differing only in
`--stockfish` both read the FIRST arm's labels and row (c) came out
byte-identical — a teacher-comparison that structurally could not detect a
teacher change (MEASURED 2026-08-16: two arms, two different Stockfish
binaries, identical `cand.sf_soft`, and Stockfish never launched for the
second). Rows now carry `sf_id` (`id name`) and a run that names an engine
reuses only that engine's rows.
⚑⚑ ENGINE IDENTITY DOES NOT RESCUE A *REPEAT* CONTROL — the two runs share an
`sf_id` BY DESIGN there, so run 2 is served run 1's rows and the measured
run-to-run variance is 0 whatever the engine does (demonstrated with a stub
that returned a different cp on every call: run 2 never launched it). A repeat
must pass `--sf-cache <fresh path>`. Reading `d_obs = 0` off a shared cache
establishes nothing about the pipeline's noise.
GPU use is the batched forwards + search only; --max-positions and
--batch-size bound the run (5k positions / 256 sims fits in <1h on a 5090).

Output: runs/target_audit_<git-sha>.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import collections
import json
import subprocess
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.selfplay.network_turn import SelfplaySearchShape

import chess
import numpy as np

from chess_anti_engine.eval.audit import (
    AuditPosition,
    PHASE_NAMES,
    SOURCE_NAMES,
    criticality_gap,
    expected_and_top1_regret,
    expected_blunder_rates,
    legal_full_indices,
    load_audit_set,
    move_regrets,
    wdl_brier,
    wdl_ece,
)
from chess_anti_engine.eval.audit_cache import (
    audit_set_provenance,
    stamp_summary,
    write_audit_cache,
)
from chess_anti_engine.eval.audit_history import (
    INPUT_ENCODINGS,
    INPUT_ENCODING_DEFAULT,
    STORED_EXTRA_FEATURES,
    STORED_HISTORY_ENCODING,
    MatchedAuditRows,
    default_matched_rows_path,
    normalize_input_encoding,
)
from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, POLICY_SIZE, policy_batch_to_full_if_needed
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.utils.git_meta import git_sha
from chess_anti_engine.selfplay.stockfish_turn import (
    _build_sf_policy_target,
    _pv_cp_score,
    _pv_wdl_score,
)
from chess_anti_engine.selfplay.temperature import apply_policy_temperature
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.stockfish.wdl import cp_to_wdl
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
from chess_anti_engine.eval.production_shape import (
    CONFIG_ABSENT,
    LIVE_CONFIG_ENV,
    FieldDiff,
    assert_matches_production,
    compare_config_values,
    format_shape_table,
    load_live_config,
    load_live_config_or_reason,
    production_search_shape,
    resolve_live_config_path,
    shape_coverage_note,
    shape_field_diff,
)
from scripts.net_source import (
    NetSource,
    add_net_source_args,
    apply_gpu_mem_cap,
    net_source_from_args,
    reject_stored_encoding_for_onnx,
)

_CANDIDATE_NAMES = {
    "raw": "a) net raw policy",
    "search": "b) net + Gumbel search (PLAY settings)",
    "sf_soft": "c) SF MultiPV soft target",
    "train": "d) production training target (full sims)",
    "train_fast": "e) fast-ply search — NOT a policy target in production",
}


@dataclasses.dataclass(frozen=True)
class _SearchProfile:
    """One search shape to score, named for the pipeline stage it belongs to.

    Keeping these separate is the point: the value-transform knobs below are
    tuned per sim-budget and the RL and play budgets disagree, so scoring a
    training target with play settings (or vice versa) reports a number no
    stage of the pipeline actually produces.
    """

    label: str
    sims: int
    topk: int
    c_scale: float
    c_visit: float
    c_visit_root: float
    c_scale_root: float
    q_visit_exp_root: float
  # ⚑ THE THREE FIELDS THIS CLASS USED TO DROP (task #227, found 2026-08-16).
  # `build_selfplay_gumbel_config` -- the production selfplay mapping -- sets
  # all three from the live yaml, and this hand-list carried none of them, so
  # every training row scored GumbelConfig's DEFAULTS instead:
  #
  #     policy_temp             live 1.5   audit used --policy-temp default 1.0
  #     target_max_visit_cap    live 5     audit used 0
  #     target_untempered_prior live True  audit used False
  #
  # The last two are not incidental. In `mcts/gumbel.py` they are the ONLY two
  # adjustments separating the STORED target `imp_store` from the play
  # distribution `imp_all`; with both at their defaults that code takes the
  # `imp_store = imp_all` branch, so the rows headed "production training
  # target" were scoring the PLAY distribution. An audit-first gate measuring
  # the wrong object certifies nothing.
  #
  # They are populated from `production_selfplay_gumbel_config`, never from a
  # fresh `flat.get(...)` call -- a second hand-list would drift the same way.
    policy_temp: float = 1.0
    target_max_visit_cap: int = 0
    target_untempered_prior: bool = False
  # NOTE (play-path audit 2026-08-03, F2): c_puct / cpuct_factor / cpuct_base /
  # fpu_reduction used to live here on the premise that "they act on tree
  # descent, so omitting them leaves a hybrid search". That premise is FALSE
  # for a Gumbel search -- the PUCT descent they drive is unreachable while
  # GumbelConfig.full_tree is True, which it always is. Carrying them made the
  # PLAY and training profiles look more different than they are.
  # Volatility-aware search. Both default OFF; when either is non-zero the
  # mechanism exists only on the PYTHON search path, and selfplay drops to it.
    volatility_q_scale: float = 0.0
    volatility_fpu: float = 0.0
    volatility_anchor: float | None = None
  # Free-form `--gumbel k=v` overrides. A tuple of pairs, not a dict, so the
  # profile stays hashable/frozen and the ORDER an operator typed survives into
  # the report header.
  #
  # These are applied by `dataclasses.replace` on the built GumbelConfig rather
  # than by adding fields here, and that is the whole point: the field list
  # above is a FIXED subset, so an override for anything outside it -- say
  # `halving_div` -- would be accepted by the CLI, printed in the header, and
  # then dropped on the floor by `_build`. That is this repo's signature defect
  # and precisely the null a `--gumbel policy_temp=2.2` sweep would produce.
  # `_assert_overrides_realized` closes it at the dispatch.
    overrides: tuple[tuple[str, float], ...] = ()
  # ⚑ THE TWO RUNNER ARGUMENTS THAT ARE NOT GumbelConfig FIELDS (F1,
  # 2026-08-16). `gumbel_vloss_weight` / `gumbel_target_batch` are read off the
  # selfplay SearchConfig and handed to the C runner as keyword arguments, so
  # the previous revision's "field-complete over GumbelConfig" comparison was
  # blind to them BY CONSTRUCTION: a --config differing from live only on
  # `gumbel_vloss_weight` was not refused and was stamped authoritative, while
  # `_net_candidates` fed the CLI default 0 against production's 1. This
  # script's own C17 note says why that matters -- at weight 0 the C search
  # re-walks 29-76% duplicate leaves, those visits inflate `max_visit`, and
  # `max_visit` sets the root q_scale that sharpens the stored TRAINING
  # TARGET. Same argument the fix already made for `target_max_visit_cap`.
  #
  # They live on the profile so the runner reads them off the SAME object the
  # guard checks, exactly like every GumbelConfig field.
    vloss_weight: int = 0
    target_batch: int = 0
  # The complete search shape production selfplay would hand its runner for
  # THIS row, straight from production's own builder. Present on the training
  # rows, None on the PLAY row (whose shape is intentionally
  # PLAY_SEARCH_DEFAULTS, not the selfplay shape). `build_profile_search_shape`
  # starts from this object rather than reassembling a config field by field,
  # so a knob added to the production mapping arrives here automatically
  # instead of needing a new column above.
    production_base: SelfplaySearchShape | None = None


# Fields on which the audit's TRAINING rows deliberately differ from live
# selfplay. A dict, not a set: there is nowhere to record a deviation without
# also recording why it is one, so an undocumented deviation cannot be added.
# Anything NOT in here must match production exactly or the run is refused.
TRAIN_SHAPE_DEVIATIONS: dict[str, str] = {
    "simulations": (
        "per-row by design — the full-sims and playout-capped fast rows are "
        "different budgets, and --rl-sims re-prices the full row for the "
        "node-matched readout"
    ),
    "add_noise": (
        "root Gumbel noise perturbs the stored visit distribution, so the "
        "training rows measure the noise-free SHAPE of the target rather than "
        "one noisy draw of it; the alternative is a non-deterministic ruler"
    ),
    "input_history_encoding": "read off the loaded checkpoint, not the yaml",
    "input_extra_features": "read off the loaded checkpoint, not the yaml",
    "policy_encoding": "read off the loaded checkpoint, not the yaml",
    "compute_relations": "read off the loaded checkpoint, not the yaml",
}

# ⚑ THE HAND-LIST IS GONE FROM HERE TOO. What used to sit at this line was
# `PRODUCTION_SEARCH_KEYS`, ten key NAMES the audit's `--config` had to agree
# with the live yaml on. That is #227 one level up, and it defeats even the
# correctly-configured path: let production gain
# `gumbel_target_min_visit_frac` (a `SearchConfig` field
# `build_selfplay_gumbel_config` consumes and the live yaml sets) and every
# guard passes. `production_base` is built from the audit's stale flat, so it
# picks the field DEFAULT; `assert_matches_production` compares two objects
# built from that SAME stale flat and finds nothing; and a name list that does
# not contain the new name cannot see it either. The report is still headed
# "production training target".
#
# So the comparison is FIELD-COMPLETE instead: build the complete shape
# production's own builder hands its runner from the audit's config and from
# the live config, and diff every field. A knob added to the production mapping
# is covered the day it is added, with no list to update — exactly like the
# training-row guard.
#
# ⚑ AND "FIELD-COMPLETE OVER <SCHEMA>" IS ONLY AS COMPLETE AS <SCHEMA>. The
# first revision diffed `GumbelConfig`, called that exhaustive, and printed the
# affirmative line over a config differing from live on `gumbel_vloss_weight` —
# a knob production hands the C runner that has no `GumbelConfig` field. The
# comparison object is now `SelfplaySearchShape`, the runner's own argument
# set, because the question is what reaches the CONSUMER and not what one
# dataclass happens to declare. Moving to a roomier dataclass would have been
# the same defect with a later expiry date.
CONFIG_COMPARE_EXEMPT: dict[str, str] = {
    "simulations": (
        "pinned to 1 on BOTH sides of this comparison so the shape is compared "
        "independently of the budget; the real budgets are value-compared "
        "through AUDIT_DIRECT_CONFIG_KEYS below"
    ),
}

# EVERY key this script reads STRAIGHT out of the flat config rather than
# through production's builder. A shape diff cannot see any of them, so they
# are value-compared against the live yaml — and the list has to be COMPLETE,
# not merely non-empty, or `config_authority` claims coverage the check does
# not have. That was the state before 2026-08-16: only the two sim budgets were
# compared, so a `--config` differing from live on `sf_policy_temp` produced a
# non-live SF soft target for row (c) under a stamp saying the config was
# proved to be production's.
#
# ⚑ COMPLETENESS IS TESTED, NOT ASSERTED.
# `tests/test_production_shape_guard.py::test_every_direct_config_read_is_checked`
# walks this module's AST for every `flat[...]` / `flat.get(...)` literal and
# fails if one is missing here. Adding a direct read without adding the key
# breaks that test, which is the only reason this list can be trusted to be a
# list of everything rather than a list of whatever someone remembered.
# ⚑ Prefer routing a new read through the builder over lengthening this list.
AUDIT_DIRECT_CONFIG_KEYS: tuple[str, ...] = (
    # sim budgets — passed to the builder as an explicit `simulations=`
    "fast_simulations",
    "mcts_simulations",
    # row (c), the SF MultiPV soft target
    "sf_policy_cp_temp",
    "sf_policy_label_smooth",
    "sf_policy_score_mode",
    "sf_policy_temp",
    "sf_wdl_cp_draw_width",
    "sf_wdl_cp_slope",
    "sf_wdl_use_cp_logistic",
    # value row (iii), the production WDL blend
    "search_wdl_frac",
    "sf_wdl_frac",
    # rows (d)/(e) — stored-target temperature and the full/fast ply mix
    "playout_cap_fraction",
    "temperature",
    # the searches' tablebase probing
    "syzygy_in_search",
    "syzygy_path",
)


@dataclasses.dataclass(frozen=True)
class ConfigAuthority:
    """Was the audited ``--config`` PROVED to be the live production config?

    Carried out of ``load_audit_config`` rather than left as a printed line,
    because a warning that only reaches stdout is not a mitigation: the
    artifact this script leaves behind is a JSONL dump that outlives the
    terminal, gets joined to other dumps months later, and is read by tools
    that never saw the warning. ``stamp()`` is what goes into it.

    ⚑ ``authoritative`` MEANS EXACTLY ONE THING, and ``covers`` says which.
    Until 2026-08-16 it was written onto every dump row while the check behind
    it compared the ``GumbelConfig`` fields plus two sim budgets — so a
    ``--config`` differing from live on ``sf_policy_temp`` (row (c)) or on
    ``gumbel_vloss_weight`` (a runner argument, invisible to that comparison)
    was stamped as proved. A flag whose scope is not stated is read at the
    scope the reader needs, which is always the wider one. The claim now is:
    *every config value this script consumes* — the complete selfplay search
    shape handed to the runner, plus ``AUDIT_DIRECT_CONFIG_KEYS`` — was
    value-compared against the live file and agreed.

    It still does NOT cover, and cannot: the per-ply root-noise schedule (no
    field to compare — see ``production_shape.SHAPE_COVERAGE_NOTE``), and
    anything an operator overrode from the CLI. CLI deviations are recorded
    separately, on the ``search_shape`` stamp, because they are a property of
    the RUN rather than of the config.
    """

    authoritative: bool
    # The file the audited config was compared against, or why there was none.
    reference: str
    # Whether the operator asked for a deliberately non-production reading.
    allow_stale: bool
    # Empty exactly when `authoritative` is True.
    reason: str
    # ⚑ The REALIZED VALUE of every `AUDIT_DIRECT_CONFIG_KEYS` key, off the
    # config this run actually used. Not the names — see `config_values`.
    values: dict[str, object] = dataclasses.field(default_factory=dict)

    def config_values(self) -> dict[str, object]:
        """The target-construction values this run's rows were built from.

        ⚑ VALUES, NOT NAMES, AND THIS IS THE FIX FOR A REAL JOIN.
        ``authoritative`` is a SAME-RUN property: it says this run's config
        agreed with the live file AT THE TIME IT RAN. It cannot see TEMPORAL
        drift, and temporal drift is the normal case here — the live yaml is
        edited between audits by design. Two dumps a week apart, one made under
        ``sf_policy_temp: 2.0`` and one under ``3.0``, each pass their own
        contemporaneous check, each stamp ``authoritative: true``, and
        ``cand.sf_soft.exp`` then joins across two different row-(c) rulers with
        a tight CI that is entirely the ruler change. Banking the key NAMES
        (which is all this stamp used to do) cannot detect that: the names are
        identical in both dumps by construction.

        So the values ride on the dump as their own field and
        ``paired_compare.RULER_FIELDS`` refuses a join whose two sides
        disagree — the same treatment ``search_shape`` gets, for the same
        reason, over the half of the target that the search shape does not
        describe.
        """
        return dict(self.values)

    def stamp(self) -> dict[str, object]:
        """The dump's record of whether its target rows describe production."""
        return {
            "authoritative": bool(self.authoritative),
            "reference": self.reference,
            "reason": self.reason,
          # WHAT the boolean above was proved over. Banked, not just printed:
          # the scope is the half of the claim a later reader cannot
          # reconstruct, and a stamp read at the wrong scope is worse than no
          # stamp.
            "covers": {
                "search_shape": "complete selfplay runner argument set",
                "config_keys": list(AUDIT_DIRECT_CONFIG_KEYS),
              # ⚑ The residual limit of the completeness check behind
              # `config_keys`, banked so the scope claim is bounded rather
              # than overstated. `test_every_direct_config_read_is_checked`
              # regenerates that list by walking this module's AST for reads
              # off `flat` AND off any local bound from it. It does NOT follow
              # a config dict into a helper that renames the parameter, nor
              # through a copy (`dict(flat)`) — route a new read through
              # production's builder rather than testing that boundary.
                "scan": (
                    "AUDIT_DIRECT_CONFIG_KEYS is regenerated from this "
                    "module's AST (reads off `flat` and its name-aliases); "
                    "a dict copied or renamed through a call is not followed"
                ),
              # And the values themselves are a SEPARATE dump field, not part
              # of this stamp, because they are a RULER and this is a
              # provenance verdict. Joining on `reference` (an absolute path)
              # or on `reason` (free text) would refuse legitimate comparisons.
                "config_values": "banked separately as the `target_config` field",
            },
        }


# GumbelConfig fields that come off the loaded CHECKPOINT rather than off the
# config, and so are not part of the search RULER: two dumps from nets with
# different input layouts must still be joinable, and `input_encoding` is
# already a RULER_FIELD in its own right.
_CHECKPOINT_DERIVED_FIELDS: frozenset[str] = frozenset({
    "input_history_encoding",
    "input_extra_features",
    "policy_encoding",
    "compute_relations",
})


# The training rows' shape, as banked on every dump row. ⚑ THIS IS A RULER
# STAMP, not documentation: `scripts/paired_compare.py` lists it in
# `RULER_FIELDS` and refuses a join whose two sides disagree.
#
# Rows (d)/(e) MOVED on 2026-08-16 — before the fix the audit scored the PLAY
# distribution under the "production training target" heading — so a dump
# banked before that date and one banked after report a paired delta with a
# tight CI that is entirely the ruler change. A doc note cannot stop a tool.
#
# ⚑ COMPLETE, NOT "the three fields that were missing". An earlier revision
# stamped exactly `policy_temp` / `target_max_visit_cap` /
# `target_untempered_prior`, which fixes only the one ruler change that had
# already happened: let production move `topk`, `c_scale` or the sim budget
# after both dumps use this code and each run passes its own live-config check
# while emitting the SAME three-field stamp, so `paired_compare` joins them and
# attributes the ruler change to the checkpoints. The stamp is therefore every
# field of the shape the runner was handed, minus the checkpoint-derived ones.
def train_shape_stamp_fields() -> tuple[str, ...]:
    """Every field the ruler stamp carries, derived — never hand-listed."""
    import dataclasses as _dc

    from chess_anti_engine.eval.production_shape import RUNNER_ARG_FIELDS
    from chess_anti_engine.mcts.gumbel import GumbelConfig

    return tuple(sorted(
        [f.name for f in _dc.fields(GumbelConfig)
         if f.name not in _CHECKPOINT_DERIVED_FIELDS]
        + list(RUNNER_ARG_FIELDS),
    ))


def train_shape_stamp(shape: SelfplaySearchShape) -> dict[str, object]:
    """The ruler stamp for the training rows this run scored.

    ⚑ TAKES THE SHAPE THE RUNNER WAS HANDED, not the ``_SearchProfile``. The
    profile is the pre-override description, and reading the stamp off it made
    the stamp LIE precisely when a ruler difference existed: with
    ``--gumbel-training-rows --gumbel policy_temp=3.0,target_max_visit_cap=99``
    the banked stamp read ``{1.5, 5, True}`` while the search ran ``3.0 / 99``,
    so two dumps from a `policy_temp` sweep stamped identically and
    `require_same_ruler` joined them as the same ruler. An anti-ruler-mixing
    mechanism that reproduces ruler mixing is worse than none, because it is
    also reassuring.
    """
    values: dict[str, object] = {}
    for name in train_shape_stamp_fields():
        raw = getattr(shape.cfg, name) if hasattr(shape.cfg, name) else getattr(shape, name)
      # bool before int: `isinstance(True, int)` is True, and a stamp that
      # writes 1 where the config held True compares unequal to a stamp that
      # writes true. json.dumps is the comparison, so the TYPE is load-bearing.
        if isinstance(raw, bool):
            values[name] = bool(raw)
        elif isinstance(raw, (int, float)):
            values[name] = float(raw) if isinstance(raw, float) else int(raw)
        else:
            values[name] = raw
    return values


def dump_ruler_stamps(
    realized_shapes: dict[str, SelfplaySearchShape], authority: ConfigAuthority,
) -> dict[str, object]:
    """Every ruler / provenance stamp a per-position dump row carries.

    Module-level and public for the same reason ``build_profile_search_shape``
    is: as an inline dict literal buried in ``main()`` the ONLY way to check it
    was a source grep, and a source grep is not a check —
    ``test_dump_row_carries_both_stamps`` used to assert a literal line of text,
    which meant it went red on a correct refactor and would have stayed green
    for a stamp built from the wrong object. With this addressable the test
    RUNS it and compares the stamps two different rulers produce.

    ⚑ EVERY REALIZED ROW, keyed by row name. ``realized_shapes`` is what
    ``_net_candidates`` returns, so it covers every profile that ran a search:
    ``search`` (the PLAY row) as well as ``train`` and ``train_fast``.

    Banking only ``train`` — which is what this did — left ``cand.train_fast.*``
    unprovenanced: change ``fast_simulations`` alone and two dumps carry
    byte-identical stamps over rows produced by different search budgets, so
    ``paired_compare`` joins them and charges the ruler change to the
    checkpoints. The PLAY row had the same hole with a different knob: its
    ``policy_temp`` is the operator-settable ``--policy-temp``.
    ``paired_compare.ruler_fields_for`` checks this stamp for every ``cand.``
    row that ran a search, so nothing banked here is banked and then ignored.
    """
    return {
        "search_shape": {
            row: train_shape_stamp(shape)
            for row, shape in sorted(realized_shapes.items())
        },
        "config_authority": authority.stamp(),
        "target_config": authority.config_values(),
    }


def parse_gumbel_overrides(specs: list[str] | None) -> tuple[tuple[str, float], ...]:
    """``["policy_temp=2.2", "topk=8"]`` -> validated (key, value) pairs.

    Same contract as ``scripts/arena_standard.py``'s ``--cand-gumbel`` /
    ``--ref-gumbel``, and deliberately the same key names (raw ``GumbelConfig``
    field names, snake_case) so a shape can be moved between the two by copy
    and paste. The UCI surface CamelCases the same fields; the mapping is
    documented in docs/operations.md.

    Refuses, rather than accepts-and-ignores:
      * a key that is not a GumbelConfig field -- caught here instead of by
        `dataclasses.replace` after the checkpoint has loaded and SF has run;
      * a key in `INERT_GUMBEL_KNOBS` -- the PUCT descent they drive is
        unreachable while `full_tree=True`, so a sweep over one returns a flat,
        perfectly reproducible null that reads as a measurement;
      * a VALUE that lands inside a real field but outside the band where the
        field does anything -- `mcts.gumbel.validate_gumbel_config` decides,
        and today that is `policy_temp`, any non-finite number, and a
        `halving_div` the C would silently raise (see below).

    ⚑ The third rule is the same defect as the second, one level down: refusing
    a dead KNOB and then accepting a dead VALUE of a live knob leaves the exact
    hole the guard exists to close. `--gumbel policy_temp=0` used to parse,
    survive `_assert_overrides_realized` (it really does reach `cfg.policy_temp`)
    and print `--gumbel realized policy_temp=0.0` -- while `apply_policy_temp`
    returned the priors untouched. A gate that cannot fail, wrapped around a
    value that is silently ignored, under a header naming the operator's number.

    REFUSAL rather than an "inert" note is deliberate and matches the two rules
    above: `--gumbel` is a batch audit CLI, so the operator is not present to
    read a warning, and the artifact left behind is a complete, reproducible,
    WRONG number. It also matches the other surface this PR ships -- UCI
    `PolicyTemperature` has range 0.5-5.0 and refuses `0` out loud.
    """
    import dataclasses as _dc

    from chess_anti_engine.mcts.gumbel import INERT_GUMBEL_KNOBS, GumbelConfig

    fields = {f.name: f.type for f in _dc.fields(GumbelConfig)}
    out: list[tuple[str, float]] = []
    for spec in specs or []:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"--gumbel: expected k=v pairs, got {part!r}")
            key, raw = part.split("=", 1)
            key = key.strip()
            if key not in fields:
                raise SystemExit(
                    f"--gumbel: {key!r} is not a GumbelConfig field. Valid "
                    f"keys: {', '.join(sorted(fields))}"
                )
            if key in INERT_GUMBEL_KNOBS:
                raise SystemExit(
                    f"--gumbel: {key!r} cannot affect a Gumbel search and is "
                    "refused. It drives the PUCT descent, which "
                    "GumbelConfig.full_tree=True makes unreachable (play-path "
                    "audit 2026-08-03 F2). A sweep over it would return a flat "
                    "null and read as a measurement."
                )
            try:
                value = float(raw)
            except ValueError:
                raise SystemExit(
                    f"--gumbel: {key}={raw!r} is not a number"
                ) from None
            out.append((key, value))
    _refuse_dead_overrides(out)
    return tuple(out)


def _refuse_dead_overrides(pairs: list[tuple[str, float]]) -> None:
    """Refuse values that reach the config and are then not read.

    Delegates to `mcts.gumbel.validate_gumbel_config` -- the ONE home of the
    bands, shared with `policy_temp_active`, `apply_policy_temp`, both
    `gumbel_c` bf16 gates, the worker's realized-shape log line and
    `arena_standard`'s `--*-gumbel` -- rather than re-deriving anything here. A
    guard has to share the criterion's instrument or it is guarding a different
    question, and a second copy of `0.05 <= T <= 20.0` in this file would drift
    the day the band moves. This function was that second copy until 2026-08-23:
    it checked `policy_temp` only, so `--gumbel c_scale=nan` still parsed.

    Checked on the ASSEMBLED config rather than per key, because that is the
    object `_net_candidates` will `dataclasses.replace` the overrides onto.

    `policy_temp=1.0` is `policy_temp_active(1.0) == False` but is NOT dead: it
    is the shipped default and an explicit "run the untempered prior" is a real
    request, so the validator lets it through. Everything else the predicate
    rejects (0, 0.01, 1e300, nan, inf, negatives) is a no-op `apply_policy_temp`
    will silently swallow.
    """
    import dataclasses as _dc

    from chess_anti_engine.mcts.gumbel import GumbelConfig

    if not pairs:
        return
    _refuse_dead_search_cfg(_dc.replace(GumbelConfig(), **dict(pairs)), where="--gumbel")


def _refuse_dead_search_cfg(cfg: GumbelConfig, *, where: str) -> None:
    """`validate_gumbel_config` with this script's exit style.

    A ``SystemExit`` rather than the ``ValueError`` the validator raises: this
    is a batch audit CLI, so the operator is not present to read a traceback,
    and the artifact left behind by accepting would be a complete, reproducible,
    WRONG number.
    """
    from chess_anti_engine.mcts.gumbel import validate_gumbel_config

    try:
        validate_gumbel_config(cfg, where=where)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}. The audit would score the DEFAULT search and report it "
            "under a header naming your value. Refusing to run — pick a value "
            "the search reads, or drop the override."
        ) from None


def sf_reference_sets(
    move_cp: Mapping[str, float],
) -> tuple[set[str], set[str]]:
    """Deep-SF `(top1, top10)` reference sets for the paired per-position booleans.

    BOTH are built by SCORE, never by slice.

    `top1` is a SET because SF's MultiPV list routinely holds several moves at
    the same cp, and calling a candidate wrong for picking one of the co-best
    would measure tie-breaking, not agreement.

    `top10` follows the same rule for the same reason. A bare `_ranked[:10]`
    picks among the moves sharing the TENTH-ranked cp by the mapping's
    iteration order, so a candidate playing an equally scored 11th-ranked move
    is recorded `out_of_top10=true` and the paired statistic measures MultiPV
    tie ordering rather than move quality.

    Measured on the shipped frozen set (`data/audit_set_v1.jsonl`, 4000 rows)
    the score form changes NOTHING: that set is MultiPV=10 exactly -- 3562 rows
    carry precisely 10 listings and 0 carry an 11th to tie with -- so no banked
    number moves. It closes the hole for the wider sets `build_audit_set.py`
    can generate, where an 11th listing exists and the ordering is arbitrary.

    `top10` is empty (-> the caller emits `None`, not `False`) when the list is
    shorter than 10: a position SF listed 5 moves for cannot support an
    "outside the top 10" claim at all, and emitting `False` there would quietly
    count it as a success.
    """
    ranked = sorted(move_cp.items(), key=lambda kv: -kv[1])
    if not ranked:
        return set(), set()
    top1 = {u for u, cp in ranked if cp >= ranked[0][1] - 1e-9}
    if len(ranked) < 10:
        return top1, set()
    return top1, {u for u, cp in ranked if cp >= ranked[9][1] - 1e-9}


def _coerce_override(cfg_field_default: object, value: float) -> float | int | bool:
    """Match the GumbelConfig field's type so `topk=8.0` does not land an int
    field with a float (which the C signature then rejects mid-search)."""
    if isinstance(cfg_field_default, bool):
        return bool(value)
    if isinstance(cfg_field_default, int):
        return int(value)
    return float(value)


def _assert_overrides_dispatched(
  # `GumbelConfig` is imported inside the functions that need it (module import
  # time is on the critical path for every `--help`), so it is not a name this
  # signature can reference. `Mapping` rather than `dict` because dict's value
  # type is INVARIANT: `dict[str, GumbelConfig]` is not a `dict[str, object]`.
    cfgs: Mapping[str, object],
    profiles: dict[str, _SearchProfile],
    *,
    requested: tuple[tuple[str, float], ...],
) -> None:
    """THE guard. Conditioned on what the OPERATOR ASKED FOR, never on what survived.

    The version this replaced ran ``if p.overrides:`` and nothing else, so it
    was conditioned on the very value that goes missing: deleting
    ``gumbel_overrides=gumbel_overrides`` from ``main()``'s
    ``build_search_profiles`` call left every profile with ``overrides=()``,
    the loop body never executed, and the whole audit ran the DEFAULT search
    shape under a header that said otherwise. That mutant passed 141 tests.
    A guard that can only fire when the value is present cannot detect the
    value going absent -- it is this repo's signature defect wearing the
    costume of the fix for it.

    ``requested`` comes from ``parse_gumbel_overrides``, i.e. straight off the
    command line, and travels to here beside the profiles rather than through
    them. So an override dropped anywhere in between is a MISMATCH here rather
    than a quietly empty loop.
    """
    if requested:
      # `--gumbel` is a PLAY-row flag (`--gumbel-training-rows` adds the two
      # training rows on top); the PLAY row therefore always carries it, and
      # "the PLAY row does not" is exactly the dropped-keyword signature.
        play = profiles.get("search")
        if play is None or tuple(play.overrides) != tuple(requested):
            asked = " ".join(f"{k}={v}" for k, v in requested)
            got = (
                "no PLAY profile at all" if play is None
                else (" ".join(f"{k}={v}" for k, v in play.overrides) or "nothing")
            )
            raise SystemExit(
                f"[audit] --gumbel {asked} was parsed but the PLAY profile "
                f"carries {got}. The override was dropped between the command "
                "line and the search profile — the run would report the "
                "DEFAULT search shape under a header naming your values. "
                "Refusing to run."
            )
    for name, p in profiles.items():
        if p.overrides:
            _assert_overrides_realized(
                cfgs[name], p.overrides, where=_CANDIDATE_NAMES[name],
            )
            print(
                f"[audit] {_CANDIDATE_NAMES[name]}: --gumbel realized "
                + " ".join(f"{k}={getattr(cfgs[name], k)}" for k, _ in p.overrides),
                flush=True,
            )


def _assert_overrides_realized(cfg, overrides, *, where: str) -> None:
    """Fail loudly if an override did not land on the config the search reads.

    THE dispatch guard. The CLI parsing above proves the operator typed a real
    field name; it proves nothing about whether the value survived the trip
    through `_SearchProfile` into the `GumbelConfig` handed to the runner. That
    trip is where a knob gets dropped, and a dropped knob here produces a
    complete, reproducible, WRONG audit -- the exact failure this script is
    supposed to catch in other people's code.
    """
    import dataclasses as _dc

    defaults = {f.name: getattr(type(cfg)(), f.name) for f in _dc.fields(cfg)}
    bad: list[str] = []
    for key, value in overrides:
        want = _coerce_override(defaults[key], value)
        got = getattr(cfg, key)
        if isinstance(want, float):
            ok = abs(float(got) - want) <= 1e-9 * max(1.0, abs(want))
        else:
            ok = got == want
        if not ok:
            bad.append(f"{key}: asked {want!r}, config has {got!r}")
    if bad:
        raise SystemExit(
            f"[audit] {where}: --gumbel override did not reach the search "
            f"config: {'; '.join(bad)}. Refusing to run — the numbers would "
            "look clean and mean nothing."
        )


def build_search_profiles(
    flat: dict[str, object], *, play_sims: int, play_topk: int | None,
    rl_sims_override: int | None = None,
    gumbel_overrides: tuple[tuple[str, float], ...] = (),
    override_training_rows: bool = False,
    vloss_weight: int | None = None,
    target_batch: int | None = None,
) -> dict[str, _SearchProfile]:
    """The search shapes to score: one PLAY, two TRAINING.

    `flat` is the flattened run config, so the training profiles follow the
    live yaml rather than a constant that goes stale the moment a search knob
    is tuned. GumbelConfig's own defaults ARE the RL shape (c_scale 0.1 and the
    legacy LINEAR root via the c_visit_root/c_scale_root/q_visit_exp_root
    sentinels below), which is the deliberate "training/RL stays bit-identical"
    choice from PR #84 — the sentinels are not placeholders.

    ⚑ The training rows' knobs now come from
    ``production_selfplay_gumbel_config`` — production's OWN builder — rather
    than from a hand-list of ``flat.get`` calls. The hand-list was the #227
    defect: it silently omitted `gumbel_policy_temp`,
    `gumbel_target_max_visit_cap` and `gumbel_target_untempered_prior`, which
    between them are what makes a stored TRAINING target differ from a PLAY
    distribution at all. Reading the builder's output means a knob added to
    production selfplay is carried here automatically, and one this script
    overrides shows up in `assert_matches_production` instead of vanishing.

    ``vloss_weight`` / ``target_batch`` are ``None`` for "whatever production
    runs" and an int for the deliberate C17 separating arm. ⚑ ``None`` rather
    than ``0``: the CLI default used to BE 0 while production runs
    ``gumbel_vloss_weight: 1``, so the training rows searched the
    duplicate-leaf shape on every ordinary invocation and the guard, which
    compared ``GumbelConfig`` fields only, could not see it.
    """
    from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS, GumbelConfig

    rl = GumbelConfig()  # the RL/training shape, by construction
    rl_sims = int(flat.get("mcts_simulations", 256))  # pyright: ignore[reportArgumentType]
  # Node-matched arm for the C17 readout. `--target-batch 1` and `--vloss-weight`
  # both buy ~60% more DISTINCT nodes at the same nominal sim count, so any gain
  # they show is partly "more search". Raising the PRODUCTION arm's sims until
  # its distinct-node count matches the fixed arm's separates the two. Training
  # rows only -- the PLAY row keeps --sims.
    if rl_sims_override is not None and rl_sims_override > 0:
        rl_sims = int(rl_sims_override)
    rl_fast_sims = int(flat.get("fast_simulations", 32))  # pyright: ignore[reportArgumentType]

    def _rl(label: str, sims: int) -> _SearchProfile:
      # ONE call to production's builder per row, and every knob below is read
      # off its result. Note what is NOT here any more: `flat.get("gumbel_...")`
      # lookups. Each of those was an independent opportunity to omit a key,
      # and three of them had already been taken.
      # `production_search_shape`, not `production_selfplay_gumbel_config`:
      # the runner's COMPLETE argument set, so `vloss_weight` /
      # `target_batch` are production's by default instead of the CLI's 0.
        shape = production_search_shape(flat, simulations=sims)
        prod = shape.cfg
        return _SearchProfile(
          # Training rows follow the yaml by default; `--gumbel` reaches them
          # only under `--gumbel-training-rows`. Overriding the TARGET's own
          # shape by accident would score a search selfplay never runs and
          # print it in the column headed "production training target" —
          # the same reasoning that already makes `--gumbel-topk` PLAY-only.
            overrides=gumbel_overrides if override_training_rows else (),
            label=label, sims=sims, topk=int(prod.topk), c_scale=float(prod.c_scale),
            c_visit=rl.c_visit, c_visit_root=rl.c_visit_root,
            c_scale_root=rl.c_scale_root, q_visit_exp_root=rl.q_visit_exp_root,
          # The three fields #227 found missing. They reach the search through
          # `production_base` below; they are also mirrored onto the profile so
          # the report header can PRINT the target shape it scored.
            policy_temp=float(prod.policy_temp),
            target_max_visit_cap=int(prod.target_max_visit_cap),
            target_untempered_prior=bool(prod.target_untempered_prior),
          # Volatility search is an open, default-off flag family that the
          # audit-first rule still has to be able to judge. Carrying the
          # values means enabling them in the yaml changes the audited
          # target, instead of the audit quietly scoring the baseline.
            volatility_q_scale=float(prod.volatility_q_scale),
            volatility_fpu=float(prod.volatility_fpu),
            volatility_anchor=float(prod.volatility_anchor),
          # Production's values unless the operator explicitly asked for the
          # C17 separating arm. An explicit value is a DEVIATION and is
          # declared as one in `build_profile_search_shape`, which is what
          # keeps the "matches production" wording off a run that does not.
            vloss_weight=(
                int(shape.vloss_weight) if vloss_weight is None else int(vloss_weight)
            ),
            target_batch=(
                int(shape.target_batch) if target_batch is None else int(target_batch)
            ),
            production_base=shape,
        )

    return {
        "search": _SearchProfile(
            overrides=gumbel_overrides,
            label="PLAY (UCI/TCEC)", sims=int(play_sims),
          # The PLAY row must be the WHOLE play shape: topk differs from the
          # training default and acts on descent, so taking only the
          # root-transform subset left a hybrid neither path runs. (The PUCT
          # knobs that used to be listed here are inert -- audit F2.)
            topk=int(play_topk if play_topk is not None else PLAY_SEARCH_DEFAULTS["topk"]),
            c_scale=float(PLAY_SEARCH_DEFAULTS["c_scale"]),
            c_visit=float(PLAY_SEARCH_DEFAULTS["c_visit"]),
            c_visit_root=float(PLAY_SEARCH_DEFAULTS["c_visit_root"]),
            c_scale_root=float(PLAY_SEARCH_DEFAULTS["c_scale_root"]),
            q_visit_exp_root=float(PLAY_SEARCH_DEFAULTS["q_visit_exp_root"]),
          # ⚑ NOT `PLAY_SEARCH_VLOSS_WEIGHT`, deliberately. Row (b)'s search is
          # a standing ruler with banked readings, so moving it needs its own
          # ledger entry and readout — the finding this file is closing is
          # about rows (d)/(e). The CLI value (default 0) is kept and the
          # divergence from the play shape is PRINTED at startup rather than
          # left for a reader to assume away.
            vloss_weight=0 if vloss_weight is None else int(vloss_weight),
            target_batch=0 if target_batch is None else int(target_batch),
        ),
        "train": _rl("RL selfplay, full sims", rl_sims),
        "train_fast": _rl("RL selfplay, playout-capped fast sims", rl_fast_sims),
    }


_VALUE_NAMES = {
    "cp_logistic": "i) cp-logistic of shallow SF eval",
    "sf_native": "ii) shallow SF native WDL",
    "blend": "iii) production WDL blend",
    "search_root": "iv) search root WDL",
}


def profile_shape_deviations(p: _SearchProfile) -> dict[str, str]:
    """``TRAIN_SHAPE_DEVIATIONS`` plus this RUN's operator-requested ones.

    An override is a deviation from production like any other, and it belongs
    in the exempt map for the same reason the static ones do: the map is the
    only place a deviation can be recorded WITH its reason, so it is the only
    place the shape table can print it as DELIBERATE rather than as DRIFT.
    Building it per profile is what lets the assertion run AFTER the overrides
    are applied — see ``build_profile_search_shape``.
    """
    exempt = dict(TRAIN_SHAPE_DEVIATIONS)
    for key, value in p.overrides:
        exempt[key] = (
            f"operator override: --gumbel {key}={value} with "
            "--gumbel-training-rows. This run does NOT score production's "
            "target for this field; the search_shape ruler stamp records the "
            "realized value."
        )
    if p.production_base is not None:
        if int(p.vloss_weight) != int(p.production_base.vloss_weight):
            exempt["vloss_weight"] = (
                f"operator override: --vloss-weight {p.vloss_weight} "
                f"(production runs {p.production_base.vloss_weight}). The C17 "
                "separating arm — deliberate, and NOT production's target."
            )
        if int(p.target_batch) != int(p.production_base.target_batch):
            exempt["target_batch"] = (
                f"operator override: --target-batch {p.target_batch} "
                f"(production runs {p.production_base.target_batch}). The C17 "
                "separating arm — deliberate, and NOT production's target."
            )
    return exempt


def build_profile_search_shape(
    name: str,
    p: _SearchProfile,
    *,
    hist: str,
    extra: str,
    pol_enc: str,
    use_rel: bool,
    play_policy_temp: float,
) -> SelfplaySearchShape:
    """The COMPLETE search shape this profile's runner will actually be handed.

    Module-level and public for the same reason ``build_selfplay_gumbel_config``
    is: it used to be a closure inside ``_net_candidates``, so nothing outside
    could call it and the only way to check that the production-shape guard
    RUNS was to read the source. An AST test cannot tell a called guard from a
    dead one, and "accepted then silently ignored" is this codebase's signature
    defect. With this addressable, ``tests/test_production_shape_guard.py``
    drives it with a stub profile and watches the guard fire.

    Returns a ``SelfplaySearchShape``, not a ``GumbelConfig``: ``vloss_weight``
    and ``target_batch`` are runner arguments with no ``GumbelConfig`` field,
    and returning only the inner config is exactly how they escaped the guard.

    ⚑ ORDER IS THE POINT. Overrides are applied FIRST, then the assertion and
    the printed table and (through the caller) the ruler stamp all describe the
    object that is handed to the runner. The previous order — assert, print,
    stamp, THEN override — made every one of the three describe a config the
    run did not use: measured, ``--gumbel-training-rows --gumbel
    policy_temp=3.0,target_max_visit_cap=99`` asserted and stamped ``1.5/5``
    and searched ``3.0/99``. An override is not silently accepted here: it is
    declared in ``profile_shape_deviations`` and prints as DELIBERATE, so the
    table shows the operator's value and the reason it is not production's.

    add_noise=False on every profile: root Gumbel noise (`gumbel_scale` 0.75
    selfplay / 0.25 curriculum) DOES perturb the stored visit distribution, so
    the training-target rows measure the noise-free shape of the target rather
    than a single noisy draw of it. That is a deliberate, stated deviation --
    the alternative is a non-deterministic ruler -- and it is the ONE axis on
    which the train profiles still differ from live selfplay.
    """
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.selfplay.network_turn import SelfplaySearchShape

  # TRAINING rows: start from the config production's OWN builder produced for
  # this row and change only the documented deviations. Nothing is re-listed,
  # so nothing can be re-dropped -- the #227 defect is not "these three fields
  # were wrong", it is "a hand-list was the mechanism", and a hand-list that is
  # merely longer fails again at the next knob.
    if p.production_base is not None:
        cfg = dataclasses.replace(
            p.production_base.cfg,
            simulations=int(p.sims),
            add_noise=False,
            input_history_encoding=hist, input_extra_features=extra,
            policy_encoding=pol_enc, compute_relations=use_rel,
        )
      # ⚑ BEFORE the assertion, the table and (through the caller) the ruler
      # stamp. See this function's docstring: the previous order made all three
      # describe a config the run did not use.
        if p.overrides:
            base = GumbelConfig()
            cfg = dataclasses.replace(cfg, **{
                k: _coerce_override(getattr(base, k), v) for k, v in p.overrides
            })
        _refuse_dead_search_cfg(cfg, where=_CANDIDATE_NAMES.get(name, name))
        shape = SelfplaySearchShape(
            cfg=cfg,
            vloss_weight=int(p.vloss_weight),
            target_batch=int(p.target_batch),
        )
        exempt = profile_shape_deviations(p)
      # Every key changed above is in `exempt`; the assertion proves the
      # converse, that nothing OUTSIDE that map drifted -- over the runner's
      # complete argument set, and over the POST-override object.
        assert_matches_production(
            shape, p.production_base,
            exempt=exempt,
            where=_CANDIDATE_NAMES.get(name, name),
        )
      # Printed on the SUCCESS path too. A guard that only speaks when it fails
      # is indistinguishable from a guard that is not running.
        print(
            f"[shape] {_CANDIDATE_NAMES.get(name, name)}: realized vs "
            f"production selfplay\n"
            + format_shape_table(shape, p.production_base, exempt=exempt),
            flush=True,
        )
        return shape
  # PLAY row: intentionally NOT the selfplay shape (PLAY_SEARCH_DEFAULTS), so
  # it has no production_base to check against and keeps the explicit
  # construction. `--policy-temp` applies here and here only.
    cfg = GumbelConfig(
        simulations=int(p.sims), add_noise=False, temperature=0.0,
        input_history_encoding=hist, input_extra_features=extra,
        policy_encoding=pol_enc, compute_relations=use_rel,
        policy_temp=float(play_policy_temp), topk=int(p.topk),
        c_scale=p.c_scale, c_visit=p.c_visit,
        c_visit_root=p.c_visit_root, c_scale_root=p.c_scale_root,
        q_visit_exp_root=p.q_visit_exp_root,
        volatility_q_scale=p.volatility_q_scale,
        volatility_fpu=p.volatility_fpu,
    )
  # `replace` rather than a `**kw` spread: the spread erased every field's type
  # into the dict's value type, so the constructor call above type-checked as
  # though `full_tree`, `halving_div` and the two target knobs were all floats.
    if p.volatility_anchor is not None:
        cfg = dataclasses.replace(cfg, volatility_anchor=float(p.volatility_anchor))
    if p.overrides:
      # `replace` reaches EVERY GumbelConfig field, including the ones
      # `_SearchProfile` has no column for — which is why the overrides ride as
      # a raw pair list rather than as new profile fields.
        base = GumbelConfig()
        cfg = dataclasses.replace(cfg, **{
            k: _coerce_override(getattr(base, k), v) for k, v in p.overrides
        })
  # `--policy-temp` reaches a GumbelConfig HERE and nowhere else, and it does
  # not go through `parse_gumbel_overrides`, so the parse-time refusal does not
  # cover it: `--policy-temp 1e300` used to search untempered under a header
  # printing 1e300 (`policy_temp={prof.policy_temp}`), the same hole
  # `--cand-gumbel policy_temp=1e300` had in arena_standard.
    _refuse_dead_search_cfg(cfg, where=_CANDIDATE_NAMES.get(name, name))
    return SelfplaySearchShape(
        cfg=cfg,
        vloss_weight=int(p.vloss_weight),
        target_batch=int(p.target_batch),
    )


def load_audit_config(
    config_path: str, *, allow_stale: bool,
) -> tuple[dict[str, Any], ConfigAuthority]:
    """Load ``--config`` AND prove it is production's. One function, on purpose.

    These were two statements in ``main()`` until the mutation run: deleting the
    check line left the load intact, the audit ran, and no test noticed —
    "a value that is accepted and then silently ignored", applied to the guard
    itself. Fusing them means the only way to skip the check is to stop calling
    the loader, which is a visible edit rather than a deleted line — and
    ``tests/test_production_shape_guard.py`` now drives ``main()`` with this
    function stubbed to prove the call site is there.

    Returns the flattened config AND the authority verdict, because the verdict
    has to reach the DUMP. A guard whose only output is a line on stdout stops
    existing the moment the operator redirects stdout, which every batch run
    does.
    """
    flat = dict(flatten_run_config_defaults(load_yaml_file(config_path)))
    authority = _assert_config_is_production(
        config_path, flat, allow_stale=allow_stale,
    )
    return flat, authority


def _production_shape_diff(
    flat: dict[str, object], live_flat: dict[str, object],
) -> list[FieldDiff]:
    """Every RUNNER ARGUMENT on which two configs' SELFPLAY shapes differ.

    Field-complete over the consumer's argument set — see
    ``CONFIG_COMPARE_EXEMPT``. Both sides go through
    ``production_search_shape``, i.e. production's own builder, so a knob that
    stops reaching the search in production stops reaching both sides here and
    this cannot certify a wiring broken on both.
    """
    got = production_search_shape(flat, simulations=1)
    want = production_search_shape(live_flat, simulations=1)
    return shape_field_diff(got, want, exempt=CONFIG_COMPARE_EXEMPT)


def _assert_config_is_production(
    config_path: str, flat: dict[str, object], *, allow_stale: bool,
) -> ConfigAuthority:
    """Refuse to audit a config that is not, provably, the LIVE production one.

    The reference is the live file named by ``$CHESS_ANTI_ENGINE_LIVE_CONFIG``,
    NOT the in-tree ``configs/pbt2_small.yaml`` — the in-tree copy is stale by
    construction on every branch except the live one, because the live working
    tree is its only writer and its edits are routinely uncommitted.

    ⚑ FAIL-CLOSED, and this is the second half of the #227 fix. The first
    revision only WARNED when the live config could not be resolved, and then
    compared ``--config`` against the in-tree fallback and printed "all 10
    production search keys match the live config by VALUE". On this machine
    ``$CHESS_ANTI_ENGINE_LIVE_CONFIG`` was unset, is exported nowhere in the
    repo, and ``origin/main:configs/pbt2_small.yaml`` carries 0 of the 3 keys
    the finding is about — so run from the worktree that CLAUDE.md mandates for
    branch work, the fixed script reproduced the exact defect it fixes and said
    the word "live" while doing it. A guard that is disarmed by its own default
    environment is not a guard.

    So: no authoritative reference, or a reference the audited config disagrees
    with, REFUSES. ``--allow-stale-config`` is the deliberate escape for
    foreign nets and offline experiments, and it is not free — it stamps the
    per-position dump as non-authoritative, so the artifact carries the caveat
    even when nobody read stdout.
    """
    live = load_live_config()
    if live is None or not live.authoritative:
        _, reason = load_live_config_or_reason()
        if live is not None:
            reason = (
                f"the only config that resolved is {live.path} "
                f"({live.provenance}) — the in-tree copy, not the live file"
            )
        return _refuse_or_degrade(
            config_path,
            reference=str(live.path) if live is not None else "<none>",
            reason=f"no authoritative live config: {reason}",
            allow_stale=allow_stale,
            values=_realized_config_values(flat),
        )
    print(live.header(), flush=True)
    try:
        diffs = _production_shape_diff(flat, live.flat)
    except Exception as exc:
      # The audited config does not even build a production search. That is a
      # real answer to "is this production's shape", and the answer is no.
        return _refuse_or_degrade(
            config_path,
            reference=str(live.path),
            reason=(
                f"{config_path} does not build a production selfplay search: "
                f"{type(exc).__name__}: {exc}"
            ),
            allow_stale=allow_stale,
            values=_realized_config_values(flat),
        )
    diffs = diffs + compare_config_values(flat, live.flat, AUDIT_DIRECT_CONFIG_KEYS)
    if not diffs:
        print(
            f"[shape] --config {config_path}: every argument production's "
            "selfplay hands its search runner, plus every key this script "
            f"reads directly ({len(AUDIT_DIRECT_CONFIG_KEYS)}: "
            f"{list(AUDIT_DIRECT_CONFIG_KEYS)}), matches the LIVE config "
            f"({live.path}) by VALUE\n"
            + shape_coverage_note(),
            flush=True,
        )
        return ConfigAuthority(
            authoritative=True, reference=str(live.path),
            allow_stale=allow_stale, reason="",
            values=_realized_config_values(flat),
        )
    detail = "\n  ".join(str(d) for d in diffs)
    return _refuse_or_degrade(
        config_path,
        reference=str(live.path),
        reason=f"{config_path} is not the live production search shape:\n  {detail}",
        allow_stale=allow_stale,
        values=_realized_config_values(flat),
    )


def _realized_config_values(flat: dict[str, object]) -> dict[str, object]:
    """The value of every ``AUDIT_DIRECT_CONFIG_KEYS`` key in ``flat``.

    Absent keys land on ``CONFIG_ABSENT`` rather than being dropped: a key
    missing from one dump and set in another is exactly the ruler difference
    this is banked to catch, and a dropped key would make the two stamps
    compare EQUAL on it.

    ⚑ The subscript is a variable, deliberately, so this loop is not itself a
    "direct config read" the completeness scanner has to account for — the
    keys it reads are the very list that scan regenerates.
    """
    return {key: flat.get(key, CONFIG_ABSENT) for key in AUDIT_DIRECT_CONFIG_KEYS}


def _refuse_or_degrade(
    config_path: str, *, reference: str, reason: str, allow_stale: bool,
    values: dict[str, object] | None = None,
) -> ConfigAuthority:
    """Stop, or proceed under an explicit non-authoritative stamp.

    ⚑ The affirmative wording lives in the CALLER, and only on the branch that
    actually proved authority. Nothing on this path may say "live": the whole
    finding is that the reassuring line a reader greps for was printed about a
    reference the code had already decided was not live.
    """
    if not allow_stale:
        raise SystemExit(
            f"[shape] REFUSING to score a production training target: {reason}\n"
            f"  Rows (d)/(e) are headed 'production training target' and would "
            f"not be one. Export ${LIVE_CONFIG_ENV} to name the "
            f"live yaml (scripts/train.sh does this; see docs/operations.md), "
            f"or pass --allow-stale-config if you deliberately mean to score a "
            f"different configuration — that flag stamps the dump "
            f"non-authoritative so the artifact carries the caveat."
        )
    print(
        f"[shape] WARNING (--allow-stale-config): {reason}\n"
        f"  These numbers describe {config_path}, NOT the running trial, and "
        f"the per-position dump is stamped non-authoritative. Do not put them "
        f"in a table with production readings.",
        flush=True,
    )
    return ConfigAuthority(
        authoritative=False, reference=reference,
        allow_stale=allow_stale, reason=reason,
      # ⚑ Banked on the DEGRADED path too. `--allow-stale-config` is exactly
      # when two dumps are most likely to have been built from different
      # target configs, so dropping the values here would remove the ruler
      # field from the runs that need it most and leave `paired_compare`
      # comparing `{}` to `{}` — equal, and therefore joinable.
        values=dict(values or {}),
    )


def profiles_for_audit(
    args, flat: dict[str, object],
) -> tuple[dict[str, _SearchProfile], tuple[tuple[str, float], ...]]:
    """Parse ``--gumbel`` and build the search profiles in ONE place.

    The point is that there is no keyword left for ``main()`` to forget. The
    request and the profiles are derived from the same ``args`` here and handed
    back together, so the "operator asked for it" and "the search got it" halves
    cannot be wired independently -- which is precisely how
    ``gumbel_overrides=gumbel_overrides`` went missing from the
    ``build_search_profiles`` call and stayed invisible to the whole suite.

    Being a real function rather than a stretch of ``main()`` is also what makes
    it testable: ``tests/test_audit_gumbel_override_dispatch.py`` drives it with
    a stub ``args`` and no checkpoint, so the wiring has a test that does not
    need an hour of Stockfish.
    """
    requested = parse_gumbel_overrides(args.gumbel)
    profiles = build_search_profiles(
        flat,
        play_sims=int(args.sims),
        play_topk=(int(args.gumbel_topk) if args.gumbel_topk is not None else None),
        rl_sims_override=(int(args.rl_sims) if args.rl_sims else None),
        gumbel_overrides=requested,
        override_training_rows=bool(args.gumbel_training_rows),
      # ⚑ `None` means "production's value" and reaches the profile; an int is
      # a deliberate C17 arm. Derived HERE, alongside the overrides, for the
      # same reason the overrides are: a runner argument that main() has to
      # remember to forward is a runner argument that gets forgotten, and this
      # pair spent the whole of #443 commit 1 at the CLI default of 0 while
      # production ran 1.
        vloss_weight=(None if args.vloss_weight is None else int(args.vloss_weight)),
        target_batch=(None if args.target_batch is None else int(args.target_batch)),
    )
    return profiles, requested


# The CLI-SOURCED search knobs that are not recoverable from anything else a
# report carries. `--config` does not pin them: `--vloss-weight`, `--vloss-mode`
# and `--target-batch` are CLI-only (the script never reads
# `gumbel_vloss_weight` / `gumbel_target_batch` from the yaml), `--batch-size` /
# `--sims` / `--rl-sims` override or select what the config would have said, and
# `--policy-temp` / `--gumbel-topk` reshape the search with no config source at
# all. So a banked report made without them is not known-wrong, it is UNKNOWN —
# which is the one state a ruler must never be in.
# ⚑ THIS IS NOT THE WHOLE SEARCH. The CONFIG-sourced knobs are outside this set
# and are still unrecorded; `search_param_stamp`'s docstring names each one.
# ⚑ `vloss_weight` / `target_batch` are SPLIT per profile, following the
# `sims`/`rl_sims` and `play_topk`/`rl_topk` convention directly above, because
# the two profiles genuinely disagree now. `--vloss-weight` and `--target-batch`
# default to None = "inherit production", and `build_search_profiles` resolves
# that asymmetrically ON PURPOSE: the RL rows take production's value (1 / the
# yaml's target batch) while the PLAY row stays at 0, because row (b) is a
# standing ruler with banked readings that must not move silently. A single
# stamped field cannot describe both, and the one it would report is wrong for
# whichever profile it is not describing — a value accepted and then silently
# misrecorded, in the artifact every later reading is joined against.
SEARCH_PARAM_FIELDS: tuple[str, ...] = (
    "play_vloss_weight", "rl_vloss_weight", "vloss_mode",
    "play_target_batch", "rl_target_batch", "batch_size",
    "sims", "rl_sims", "fast_sims",
    "play_topk", "rl_topk",
    "play_policy_temp", "rl_policy_temp",
    "gumbel_training_rows",
)


def realized_gumbel_value(
    profile: _SearchProfile, field: str, fallback: float,
) -> float:
    """What ``_build`` will actually put in this profile's ``GumbelConfig``.

    ``--gumbel k=v`` is applied by ``dataclasses.replace`` on the BUILT config,
    i.e. AFTER the profile's own columns, so ``--gumbel simulations=300`` runs
    the search at 300 while ``profile.sims`` still reads 256. Stamping the
    profile column there would print a sim count nothing was searched at and
    look like provenance while being false — the same failure this stamp exists
    to end, one level down.

    Last write wins, because ``_build`` collects the overrides into a dict
    comprehension and a repeated key resolves the same way.
    """
    value = fallback
    for key, override in profile.overrides:
        if key == field:
            value = override
    return float(value)


def search_param_stamp(
    args: argparse.Namespace, *, profiles: dict[str, _SearchProfile],
) -> dict[str, float | bool]:
    """The search-parameter provenance carried by BOTH the report and the dump.

    One function, two consumers, so the header and the dump cannot drift apart
    or from the search the run actually ran — the same reasoning that makes
    ``profiles_for_audit`` a function rather than a stretch of ``main()``.

    Everything sim/topk/temp-shaped is read off the PROFILES, never off
    ``args``, because three separate things move those between the flag and the
    search: ``--rl-sims 0`` is a sentinel meaning "use the config's
    ``mcts_simulations``", ``--gumbel-topk`` defaults to the PLAY table rather
    than to a literal, and ``--gumbel k=v`` rewrites the built config outright.
    A stamp that echoed the flag would be a false record in all three cases.

    ⚑ EVERY KNOB THIS STAMP COVERS IS STAMPED PER PROFILE — and the covered set
    is SMALLER THAN "the search"; see the gaps below. ``--gumbel`` reaches the
    PLAY row only unless ``--gumbel-training-rows`` is passed, so a single
    unqualified column is FALSE for whichever rows the override missed. That is
    not hypothetical: a single ``policy_temp`` column read off the PLAY profile
    gave BYTE-IDENTICAL provenance to two runs whose ``cand.train`` differed by
    −9.66 cp [−18.07, −3.00] under ``paired_compare`` (independent review of PR
    #434, executed on a 6-position audit). ``play_*``/``rl_*`` pairs, and
    ``fast_sims`` for row (e), are what make those two runs distinguishable.

    ⚑⚑ KNOWN GAPS — DO NOT READ THIS STAMP AS "THE SEARCH". The knobs sourced
    from ``--config`` rather than the CLI are profile-varying and are NOT here:

    * ``gumbel_c_scale`` — ``build_search_profiles`` reads it for the RL rows
      while PLAY takes ``PLAY_SEARCH_DEFAULTS``, so the two rows genuinely
      differ on it and NEITHER value is recorded. MEASURED: two runs differing
      only in the config's ``gumbel_c_scale`` produce a differing ``cand.train``
      and byte-identical dump provenance (same review).
    * ``volatility_q_scale`` / ``volatility_fpu`` / ``volatility_anchor`` — same
      shape, and they additionally decide whether a row takes the C runner at
      all, which is what can make ``vloss_weight`` / ``vloss_mode`` /
      ``target_batch`` above inapplicable to that row while still stamped.
    * ``syzygy_in_search`` / ``syzygy_path`` — the audited search probes
      tablebases when the config says so.

    The header records the config PATH, and that is a WEAK record: the yaml is
    mutable and re-read every run, so a path does not pin the values it held.
    Closing these means stamping the REALIZED ``GumbelConfig`` per profile,
    which has to happen next to the cfgs in ``_net_candidates`` rather than
    here; that is a follow-up, and until it lands this list is the boundary.

    ⚑ ``--seed`` is NOT a gap. ``_build`` fixes ``add_noise=False`` and
    ``temperature=0.0``, and ``gumbel.py`` computes ``scale = gumbel_scale if
    add_noise else 0.0``, so the perturbation is identically zero and two seeds
    are bit-identical. Checked rather than assumed, because "the seed is in the
    dump filename" is exactly the kind of thing that gets asserted and is false.

    ⚑ ``gumbel_training_rows`` IS ITSELF A STAMPED FIELD, and it is what closes
    the same hole for every override with no dedicated column. ``--gumbel
    halving_div=4`` with and without the flag are materially different searches
    that serialize the same ``gumbel_overrides``; only the scope flag separates
    them. Adding a knob here without asking "does this vary per profile, and is
    its SCOPE recorded" re-opens the hole this stamp exists to close.
    """
    play, train = profiles["search"], profiles["train"]
    fast = profiles["train_fast"]
    return {
      # Read off the RESOLVED profiles, never off `args`. Both flags default to
      # None ("inherit production"), so `int(args.vloss_weight)` raises
      # TypeError on an ordinary invocation — and coercing the None to 0 to
      # dodge that would stamp 0 onto RL rows that searched production's 1.
      # Same rule the `sims` columns below already follow: stamp what the
      # search ran, not what the operator typed.
        "play_vloss_weight": int(play.vloss_weight),
        "rl_vloss_weight": int(train.vloss_weight),
        "vloss_mode": int(args.vloss_mode),
        "play_target_batch": int(play.target_batch),
        "rl_target_batch": int(train.target_batch),
        "batch_size": int(args.batch_size),
        "sims": int(realized_gumbel_value(play, "simulations", play.sims)),
        "rl_sims": int(realized_gumbel_value(train, "simulations", train.sims)),
        "fast_sims": int(realized_gumbel_value(fast, "simulations", fast.sims)),
      # train and train_fast are built by the same `_rl` closure and take the
      # same overrides, so one column speaks for both; only `simulations`
      # differs between them, which is why `fast_sims` is the lone extra.
        "play_topk": int(realized_gumbel_value(play, "topk", play.topk)),
        "rl_topk": int(realized_gumbel_value(train, "topk", train.topk)),
        "play_policy_temp": realized_gumbel_value(
            play, "policy_temp", args.policy_temp,
        ),
        "rl_policy_temp": realized_gumbel_value(
            train, "policy_temp", args.policy_temp,
        ),
        "gumbel_training_rows": bool(args.gumbel_training_rows),
    }


def format_search_params(
    stamp: Mapping[str, float | bool],
    *,
    gumbel_overrides: tuple[tuple[str, float], ...] = (),
) -> str:
    """One greppable ``k=v`` line for the report header.

    ``gumbel_overrides`` rides on the end rather than being folded in, because
    the two are different objects: the stamp is the fixed set every run has, the
    overrides are whatever else the operator reached into. Folding them in would
    make the line's key set vary run to run.
    """
    line = " ".join(f"{k}={v}" for k, v in stamp.items())
    if gumbel_overrides:
        line += " gumbel_overrides=" + ",".join(
            f"{k}={v}" for k, v in gumbel_overrides
        )
    return line


def _wdl_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of raw WDL logits, as `network_turn.py` does.

    `LocalModelEvaluator.evaluate_encoded` returns the model's RAW ``out["wdl"]``
    logits, not probabilities. Selfplay softmaxes them before reading the draw
    component (`network_turn.py:365-371`); feeding the logits straight into
    `_search_wdl_like_selfplay` would treat an arbitrary real number as a draw
    probability and produce negative or >1 entries that are still finite, so the
    non-finite fallback there would not catch them.
    """
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=-1, keepdims=True)
    return z


def _search_wdl_like_selfplay(q: float, net_wdl: np.ndarray) -> np.ndarray:
    """The search WDL exactly as `network_turn.py` stores it.

    ``net_wdl`` must be PROBABILITIES (see `_wdl_softmax`), not logits.

    Selfplay KEEPS the root network's own draw mass and splits only the
    remaining mass around the searched Q::

        d_raw = net_wdl[1]; rem = 1 - d_raw
        q     = clip(q, -rem, +rem)
        W     = 0.5 * (rem + q);  D = d_raw;  L = rem - W

    `losses._q_to_wdl_probs` -- the game-outcome regret correction, a DIFFERENT
    target -- instead invents ``D = 1 - |q|``, which is a different
    distribution whenever the net predicts a draw mass other than ``1 - |q|``
    -- i.e. almost always. Scoring the production WDL blend with the wrong
    draw mass makes candidates (iii) and (iv) describe a target the pipeline
    never writes, which is the same mislabeling this script was just fixed for
    on the policy side.
    """
    d_raw = float(net_wdl[1])
    rem = max(0.0, 1.0 - d_raw)
    qc = float(max(-rem, min(rem, float(q))))
    win = 0.5 * (rem + qc)
    out = np.array([win, d_raw, rem - win], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Candidate computation
# ---------------------------------------------------------------------------


def _net_candidates(
    boards: list[chess.Board],
    *,
    net: NetSource,
    device: str,
    batch_size: int,
    seed: int,
    profiles: dict[str, _SearchProfile],
  # Required, and deliberately WITHOUT a default: the expectation this function
  # checks its profiles against must not be omissible, or the guard degrades to
  # the `if p.overrides:` no-op it replaced. Callers that pass no `--gumbel`
  # spell it `()`.
    requested_gumbel_overrides: tuple[tuple[str, float], ...],
    policy_temp: float = 1.0,
    syzygy_path: str | None = None,
  # ⚑ `target_batch` / `vloss_weight` are NOT parameters here any more. They
  # were, with defaults of 0, and that is the whole of F1: production runs
  # `gumbel_vloss_weight: 1`, so every ordinary invocation searched the
  # duplicate-leaf shape while the guard — which compared GumbelConfig fields —
  # was structurally unable to notice. They now ride on the profile, come from
  # production's builder by default, and are checked by the same assertion as
  # every other runner argument. `vloss_mode` stays: it selects HOW an
  # in-flight visit is valued and has no production counterpart to default to.
    vloss_mode: int = 0,
    stored_x: np.ndarray | None = None,
    gpu_mem_fraction: float | None = None,
) -> tuple[
    list[np.ndarray],
    dict[str, list[np.ndarray]],
    dict[str, list[float]],
    list[np.ndarray],
    dict[str, SelfplaySearchShape],
]:
    """(raw probs, {profile: visit probs}, {profile: root Q}, root WDL, SHAPES).

    ⚑ The last element is the RULER, and it is returned rather than re-derived
    because the caller banks it on every dump row. Re-deriving it in ``main()``
    is how the stamp came to describe a config the run did not use: the shape
    the runner is handed is built HERE, after the overrides, and there is no
    second construction that could disagree with it.

    Every profile is run over the same batches against the same evaluator, so
    the raw forward and the model load are paid once no matter how many search
    shapes are being priced. Probs are aligned with _legal_full_indices order.

    ``stored_x`` is the audit-v2 stored production row per board, aligned with
    ``boards``. When given, and ONLY then, the raw-policy candidate is read off
    an EXTRA forward over those rows; the searches and the root WDL keep the
    FEN-only encoding they build internally, so no candidate becomes a hybrid
    of the two. When it is None this function is byte-for-byte the pre-audit-v2
    code path."""
    import torch

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts.gumbel import (
        run_gumbel_root_many,
        volatility_search_enabled,
        warn_volatility_python_path,
    )
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    # `net` carries exactly one of a checkpoint or a foreign ONNX spec and
    # raises if that is not true, so this cannot silently score a default net.
    # Every encoding below is read OFF the loaded model, never assumed: an
    # LC0/Ceres net declares lc0_root/v1/az_4672 and the searches and the raw
    # forward then all encode boards the way that net needs.
    # gpu_mem_fraction is carried this far because on the --onnx path the cap
    # is an ORT SESSION-CONSTRUCTION argument (gpu_mem_limit): there is no
    # later point at which it can be applied, and torch's own cap does not
    # bound the ONNX session.
    model = net.load(device=device, gpu_mem_fraction=gpu_mem_fraction, tag="audit")
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    if stored_x is not None:
        # Stored rows are bytes written under one specific layout; feeding them
        # to a model that declares another would produce a number that silently
        # means nothing.
        if (hist, extra) != (STORED_HISTORY_ENCODING, STORED_EXTRA_FEATURES):
            raise SystemExit(
                f"--input-encoding stored requires a checkpoint encoded as "
                f"{STORED_HISTORY_ENCODING}/{STORED_EXTRA_FEATURES}; this one "
                f"declares {hist}/{extra}"
            )
        if use_rel:
            raise SystemExit(
                "--input-encoding stored does not support dynamic-relation "
                "checkpoints: the relation tensor is rebuilt from the bare board "
                "and would contradict the stored history planes"
            )
    evaluator = LocalModelEvaluator(model, device=device)
    rng = np.random.default_rng(seed)
  # add_noise=False on every profile: root Gumbel noise (`gumbel_scale` 0.75
  # selfplay / 0.25 curriculum) DOES perturb the stored visit distribution, so
  # the training-target rows measure the noise-free shape of the target rather
  # than a single noisy draw of it. That is a deliberate, stated deviation --
  # the alternative is a non-deterministic ruler -- and it is the ONE axis on
  # which the train profiles still differ from live selfplay.
    shapes = {
        name: build_profile_search_shape(
            name, p, hist=hist, extra=extra, pol_enc=pol_enc,
            use_rel=use_rel, play_policy_temp=float(policy_temp),
        )
        for name, p in profiles.items()
    }
    cfgs = {name: s.cfg for name, s in shapes.items()}
  # Guard the DISPATCH, not the CLI: these are the objects each runner is about
  # to be handed, so an override that survives to here survives into the search.
    _assert_overrides_dispatched(
        cfgs, profiles, requested=requested_gumbel_overrides,
    )
  # Volatility-aware search exists ONLY on the Python path; selfplay drops to
  # it when either flag is set. Always calling the C path would silently score
  # the baseline search and report it as the configured training target -- the
  # audit-first gate would then be structurally unable to judge the one flag
  # family it was asked about.
  # Production runs `syzygy_in_search: true`, and selfplay hands the probe to
  # the C search so TB-eligible roots and leaves get their WDL overridden
  # (`network_turn.py:762`). Without it the endgame bucket -- a third of the
  # audit set, and the bucket TB probing exists for -- scores a pure-network
  # search rather than the target production stores.
    tb_probe = None
    if syzygy_path:
        from chess_anti_engine.tablebase import SyzygyProbe
        tb_probe = SyzygyProbe(syzygy_path)
        print(f"[audit] syzygy_in_search: probing {syzygy_path}", flush=True)

    runners = {}
    tb_kwargs: dict[str, dict[str, object]] = {}
    for name, cfg in cfgs.items():
        if volatility_search_enabled(cfg):
            warn_volatility_python_path()
            print(
                f"[audit] {_CANDIDATE_NAMES[name]}: volatility search on "
                f"(q_scale={cfg.volatility_q_scale}, fpu={cfg.volatility_fpu}) "
                f"— using the Python search path, as selfplay does",
                flush=True,
            )
            if tb_probe is not None:
              # The Python path takes no probe, so this profile is scored
              # WITHOUT the TB overrides production applies. Say so rather
              # than reporting it as the production target.
                print(
                    f"[audit] WARNING {_CANDIDATE_NAMES[name]}: the Python "
                    "volatility path cannot take a syzygy probe — endgame "
                    "numbers for this row are NOT the production target",
                    flush=True,
                )
            runners[name] = run_gumbel_root_many
            tb_kwargs[name] = {}
        else:
            runners[name] = run_gumbel_root_many_c
            tb_kwargs[name] = {"tb_probe": tb_probe} if tb_probe is not None else {}
            # C17 separating test: production accumulates leaves across halving
            # reps to fill GSS_GPU_BATCH, and with vloss_weight=0 a later rep
            # re-walks an UNCHANGED tree and re-evaluates the SAME leaf --
            # 29-76% duplicates at 256 sims, -34% tree nodes. Those duplicate
            # visits still increment N, which inflates max_visit, which sets the
            # root q_scale that sharpens the improved-policy TRAINING TARGET.
            # `--target-batch 1` flushes per rep, removing the duplication, so
            # running this audit at 0 vs 1 separates "C17 wastes compute" from
            # "C17 corrupts the target". The Python reference path takes no such
            # argument, hence C-runner only.
            #
            # ⚑ READ OFF THE SHAPE, not off a CLI argument, and passed
            # UNCONDITIONALLY — exactly as `network_turn.py` does, which is why
            # `SelfplaySearchShape.runner_kwargs()` is the thing being unpacked
            # on both sides. The shape is the object `assert_matches_production`
            # checked, so the value the runner gets is the value the guard
            # certified, and on the training rows it defaults to PRODUCTION's
            # `gumbel_vloss_weight` / `gumbel_target_batch` rather than to the
            # 0 the CLI used to supply. The other fix for the same defect, and
            # the one that KEEPS the large cross-rep batches, is
            # `--vloss-weight`: an in-flight leaf carries a visit penalty, so a
            # later rep descends somewhere else instead of re-walking to it.
            tb_kwargs[name]["target_batch"] = int(shapes[name].target_batch)
            tb_kwargs[name]["vloss_weight"] = int(shapes[name].vloss_weight)
            if int(shapes[name].vloss_weight) > 0:
                # Mode only means anything when a weight is applied, so it
                # rides along with it rather than being set independently.
                tb_kwargs[name]["vloss_mode"] = int(vloss_mode)

    raw_out: list[np.ndarray] = []
    search_out: dict[str, list[np.ndarray]] = {name: [] for name in profiles}
    root_q: dict[str, list[float]] = {name: [] for name in profiles}
  # The ROOT NETWORK's WDL, needed to rebuild the search WDL the way selfplay
  # does (see _search_wdl_like_selfplay).
    root_wdl_out: list[np.ndarray] = []
    for start in range(0, len(boards), batch_size):
        chunk = boards[start:start + batch_size]
        cbs = [CBoard.from_board(b) for b in chunk]
        xs = np.stack([
            encode_cboard(cb, input_history_encoding=hist, input_extra_features=extra)
            for cb in cbs
        ])
        rels = (
            np.stack([cb.compute_relations() for cb in cbs]) if use_rel else None
        )
        with torch.no_grad():
            if rels is None:
                pol_logits, net_wdl = evaluator.evaluate_encoded(xs)
            else:
                pol_logits, net_wdl = evaluator.evaluate_encoded(xs, relations=rels)
        net_wdl = _wdl_softmax(net_wdl)
        if stored_x is not None:
            # Row (a) only. A second forward rather than replacing `xs` above,
            # so `net_wdl` — and therefore value candidate (iv) — stays on the
            # same encoding as the search whose root Q it is combined with.
            with torch.no_grad():
                pol_logits, _ = evaluator.evaluate_encoded(
                    np.asarray(stored_x[start:start + batch_size], dtype=np.float32),
                )
        pol_logits = np.asarray(pol_logits, dtype=np.float32)
        if pol_logits.shape[1] != POLICY_SIZE:
            pol_logits = policy_batch_to_full_if_needed(pol_logits, policy_encoding=pol_enc, fill_value=-1e9)

        searched = {
            name: runners[name](
                model=None, boards=list(chunk), device=device, rng=rng,
                cfg=cfgs[name], evaluator=evaluator, **tb_kwargs[name],
            )
            for name in cfgs
        }
        for j, board in enumerate(chunk):
            _, idxs = legal_full_indices(board)
            logits = pol_logits[j, idxs].astype(np.float64)
            logits -= logits.max()
            e = np.exp(logits)
            raw_out.append(e / e.sum())
          # The C runner returns 6 elements and the Python one 4; only the
          # leading (probs, actions, values, masks) are common, so index in
          # rather than destructuring a length this code does not control.
            for name, result in searched.items():
                probs_b, values = result[0], result[2]
                visit = np.asarray(probs_b[j], dtype=np.float64)
                if visit.shape[0] != POLICY_SIZE:
                    full = np.zeros(POLICY_SIZE, dtype=np.float64)
                    full[COMPACT_TO_FULL_POLICY] = visit
                    visit = full
                search_out[name].append(visit[idxs])
                root_q[name].append(float(values[j]))
            root_wdl_out.append(net_wdl[j].copy())
        done = min(start + batch_size, len(boards))
        print(f"[net] {done}/{len(boards)} positions")
        # Release the batch's reserved CUDA blocks so the allocator's pool
        # doesn't creep across batches and collide with a concurrent trainer
        # (same fragmentation issue fixed in eval/puzzles.py). Matters most at
        # high sims (256) where per-batch trees are largest.
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return raw_out, search_out, root_q, root_wdl_out, shapes


UNRECORDED_SF_ID = "<unrecorded>"


def engine_identity(path: str, *, timeout_s: float = 60.0) -> str:
    """The engine's own ``id name``, e.g. ``Stockfish dev-20260810-5062aee5``.

    Read by driving the binary directly rather than through ``StockfishUCI``:
    that class discards every line before ``uciok`` and sits on the production
    selfplay path, so widening it for a ruler script is the wrong trade.

    ⚑ The identity comes from the ENGINE, never from the path. `stockfish_path`
    in production is a two-hop symlink whose intermediate name is misleading, so
    a filename-derived key would record the wrong provenance and still look
    plausible.
    """
    proc = subprocess.Popen(
        [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    name = ""
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        deadline = time.monotonic() + timeout_s
        for line in proc.stdout:
            s = line.strip()
            if s.startswith("id name "):
                name = s[len("id name "):].strip()
            if s == "uciok" or time.monotonic() > deadline:
                break
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # The engine already exited. Nothing to ask it to do.
            pass
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Our own child, by handle -- never a name pattern.
            proc.kill()
            proc.wait(timeout=10)
    if not name:
        raise SystemExit(f"{path}: engine did not report an `id name` before uciok")
    return name


#: The default shallow-SF cache's suffix, derived in exactly one place
#: (`resolve_sf_cache_path`). A second site that re-derives the default is how
#: an override silently stops applying on one of the two paths — which is why
#: `tests/test_audit_shallow_sf_cache_provenance.py` proves the override
#: through `main()` by EXECUTION rather than by reading the source.
SHALLOW_SF_CACHE_SUFFIX = ".shallow_sf.jsonl"


def resolve_sf_cache_path(
    audit_set: Path, override: Path | None, dump_per_position: Path | None = None,
) -> Path:
    """Where this run reads and writes shallow-SF labels.

    ⚑ The override exists for the REPEAT control and nothing else is a
    substitute for it. Engine identity keys the cache by WHICH engine wrote a
    row, which fixes the OLD-vs-NEW arm — two binaries, two `sf_id`s, forced
    re-labelling. A repeat runs the SAME binary twice on purpose, so both runs
    share an `sf_id`, run 2 matches every row, and Stockfish is never launched:
    the measured run-to-run variance is 0 by CONSTRUCTION. Point the repeat at
    a fresh path and it labels for real.

    ⚑⚑ AN OVERRIDE THAT ALIASES THE AUDIT SET IS REFUSED, AND THE DAMAGE IT
    PREVENTS IS PERMANENT. `_shallow_sf_records` opens the cache in APPEND mode,
    so `--sf-cache data/audit_set_v1.jsonl` would write label records into the
    FROZEN scoring set — which by protocol never changes after generation, and
    whose digest every stamped dump carries. Every later audit would score a
    silently different population, and there is no undo. The default can never
    collide (it adds a suffix); only an explicit override can, which is exactly
    the flag a hurried operator types a second path into. Compared by
    `realpath`, so a symlink cannot route around it. (Codex review, PR #446 P2.)
    """
    resolved = override if override is not None else audit_set.with_suffix(
        audit_set.suffix + SHALLOW_SF_CACHE_SUFFIX
    )
    if Path(resolved).resolve() == Path(audit_set).resolve():
        raise SystemExit(
            f"--sf-cache {resolved} resolves to the audit set itself "
            f"({audit_set}). The shallow-SF cache is opened in APPEND mode, so "
            "this would write label rows into the frozen scoring set and "
            "permanently change what every later audit scores. Point --sf-cache "
            "at a NEW file (a repeat control wants one per repeat)."
        )
  # ⚑⚑ AND THE COLLISION THE OTHER TWO GUARDS STRUCTURALLY CANNOT SEE: both
  # `--sf-cache` and `--dump-per-position` pointed at the SAME path that does
  # not exist yet. `refuse_if_not_a_shallow_sf_cache` inspects CONTENT and
  # returns early on a missing file, and the audit-set alias check above
  # compares against a different path -- so a fresh collision passes both. The
  # labelling pass then banks an hour of shallow-SF rows there, and
  # `write_audit_cache(..., force=True)` TRUNCATES that same file at the end of
  # the run and replaces it with the per-position dump. The expensive
  # observations are destroyed by the run that made them, with no error.
  # Compared by resolved path so a symlink or a `./` spelling cannot route
  # around it, and checked HERE because at parse time neither file exists yet,
  # which is precisely the case content inspection cannot reach.
  # (Codex review, PR #446 P2.)
    if dump_per_position is not None and (
        Path(resolved).resolve() == Path(dump_per_position).resolve()
    ):
        raise SystemExit(
            f"--sf-cache {resolved} and --dump-per-position "
            f"{dump_per_position} resolve to the SAME path. The cache is "
            "appended to during labelling and the dump is written with "
            "force=True at the end, so this run would spend an hour producing "
            "shallow-SF rows and then truncate them away. Give them separate "
            "paths."
        )
    return resolved


def refuse_if_not_a_shallow_sf_cache(cache_path: Path) -> None:
    """Refuse to APPEND to an existing file that is not a shallow-SF cache.

    ⚑⚑ THE DAMAGE IS PERMANENT AND SILENT, WHICH IS WHY THIS SITS BESIDE THE
    APPEND-OPEN RATHER THAN ONLY IN THE RESOLVER. `_shallow_sf_records` opens
    `cache_path` in mode ``"a"``. Point `--sf-cache` at the frozen audit set and
    it gains label rows; the set FREEZES after generation by protocol, its
    digest rides in every stamped dump, and there is no undo. Point it at a
    `--dump-per-position` output and that dump is corrupted instead.

    An identity check on the audit set NAMED ON THIS COMMAND LINE is not
    enough, and the review of PR #446 demonstrated all three ways past it:

    * ``os.link(audit_set, alias)`` — a HARDLINK. `realpath` compares paths and
      cannot see inodes, so the alias is accepted and appended to.
    * ``--sf-cache data/audit_set_v2.jsonl`` — a DIFFERENT frozen set, which the
      run never mentions and so cannot compare against.
    * ``--sf-cache <the dump path>`` — arguably the likelier typo than aliasing
      the file you just typed two flags earlier.

    So the test is on the CONTENT, which all three share: a shallow-SF row
    carries an integer ``multipv`` (the width the labeller requested), while an
    audit record's ``multipv`` is a LIST of PV dicts and a dump row has no
    ``multipv`` at all. One line is enough — the file is append-only and
    homogeneous.

    A missing file is fine (that is the normal fresh-cache case), and so is an
    empty one.
    """
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                first = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{cache_path}: exists but line 1 is not JSON ({exc}). "
                    "Refusing to append shallow-SF rows to it."
                ) from exc
            break
        else:
            return
    if not isinstance(first, dict) or not isinstance(first.get("multipv"), int):
        raise SystemExit(
            f"{cache_path}: exists and is NOT a shallow-SF cache (line 1 has no "
            f"integer 'multipv'; found {type(first.get('multipv') if isinstance(first, dict) else first).__name__}). "
            "This file is opened in APPEND mode, so continuing would write label "
            "rows into it — if it is the frozen audit set, that permanently "
            "changes what every later audit scores; if it is a dump, it "
            "corrupts the dump. Point --sf-cache at a new or existing "
            "shallow-SF cache."
        )


def _shallow_sf_records(
    positions: list[AuditPosition],
    *,
    cache_path: Path,
    stockfish: str | None,
    nodes: int,
    multipv: int,
    workers: int,
    nice: int,
) -> dict[str, dict]:
    """Shallow (production-strength) SF search per position, JSONL-cached.

    ⚑ Reuse is gated on ENGINE IDENTITY, not just (nodes, multipv) — see the
    module docstring. When ``stockfish`` is given, only rows this exact engine
    wrote are reused; anything else is re-labelled. When it is not given the
    cache is read as-is, but a cache holding MORE THAN ONE engine's rows at
    these settings is refused rather than silently averaged: that is a mixed
    ruler, and a mixed ruler is not a ruler.
    """
    # ⚑ Announced by the function that USES it, from the same parameter it
    # reads and writes. `main` prints its resolution early so a mistyped
    # --sf-cache fails cheaply, but an early print describes an INTENTION: the
    # only line that cannot disagree with the labelling pass is this one.
    print(f"[sf-soft] cache in use {cache_path}")
    refuse_if_not_a_shallow_sf_cache(cache_path)
    sf_id = engine_identity(str(stockfish)) if stockfish is not None else None
    cache: dict[str, dict] = {}
    other_node_counts: set[int] = set()
    foreign: collections.Counter[str] = collections.Counter()
    accepted_ids: collections.Counter[str] = collections.Counter()
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if int(d.get("nodes_requested", 0)) == nodes and int(d.get("multipv", 0)) == multipv:
                        row_id = str(d.get("sf_id") or UNRECORDED_SF_ID)
                        if sf_id is not None and row_id != sf_id:
                            foreign[row_id] += 1
                            continue
                        accepted_ids[row_id] += 1
                        cache[str(d["key"])] = d
                    elif int(d.get("multipv", 0)) == multipv:
                        other_node_counts.add(int(d.get("nodes_requested", 0)))
    if sf_id is not None:
        print(f"[sf-soft] engine `{sf_id}`")
    if len(accepted_ids) > 1:
        raise SystemExit(
            f"{cache_path}: rows at {nodes} nodes / multipv {multipv} come from "
            f"{len(accepted_ids)} different engines ({dict(accepted_ids)}). A "
            "mixed-provenance cache is a mixed ruler. Pass --stockfish to pin "
            "one engine, or point --audit-set at a per-engine copy."
        )
    if foreign:
        print(
            f"[sf-soft] ⚑ ignoring {sum(foreign.values())} cached rows at these "
            f"settings written by a different or unrecorded engine "
            f"({dict(foreign)}); they will be RE-LABELLED by `{sf_id}`. This is "
            "the cost of the cache never having recorded which engine made it."
        )
    todo = [p for p in positions if p.key not in cache]
    if todo and stockfish is None:
        hint = "pass --stockfish to populate the cache"
        if other_node_counts and not cache:
  # The cache is fully populated but at a different node budget — the audit
  # default is now --sf-effort=low (500k) to match production, so an older 50k
  # cache no longer matches. Point the user at the mismatch instead of a bare
  # "pass --stockfish".
            have = ",".join(f"{n:_}" for n in sorted(other_node_counts))
            hint = (
                f"the cache has entries only at {have} nodes, but this run wants "
                f"{nodes:_} (default --sf-effort=low=500k). Re-run with "
                f"--sf-soft-nodes {sorted(other_node_counts)[0]} (or matching "
                "--sf-effort) to reuse it, or pass --stockfish to regenerate"
            )
        raise SystemExit(f"{len(todo)} positions lack shallow-SF cache entries; {hint}")
    if not todo:
        return cache

    print(f"[sf-soft] labeling {len(todo)} positions at {nodes} nodes, multipv {multipv}")
  # ⚑ CREATE THE PARENT BEFORE AN HOUR OF STOCKFISH, NOT AFTER. `open(..., "a")`
  # does NOT create directories, and the default cache lives beside the audit
  # set so its parent always exists -- but an explicit `--sf-cache` names a path
  # the operator chose, and the repeat control's recommended layout
  # (`scratchpad/sf440/...`) is a directory no fresh checkout has. Without this
  # the run raises FileNotFoundError HERE, after the config, the checkpoint, the
  # audit set and the engine pool have all loaded. Fixed in the writer rather
  # than as a `mkdir -p` line in the ledger, because the ledger cannot be the
  # thing that makes a command runnable. Codex review of PR #446 (P1).
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    engines = [
        StockfishUCI(str(stockfish), nodes=nodes, multipv=multipv, nice=nice)
        for _ in range(max(1, workers))
    ]
    lock = threading.Lock()
    work = iter(todo)
    t0 = time.time()
    n_done = 0
    try:
        with open(cache_path, "a", encoding="utf-8") as f:
            def run_worker(wi: int) -> None:
                nonlocal n_done
                eng = engines[wi]
                while True:
                    with lock:
                        pos = next(work, None)
                    if pos is None:
                        return
                    res = eng.search(pos.fen, nodes=nodes)
                    rec = {
                        "key": pos.key,
                        "nodes_requested": nodes,
                        "multipv": multipv,
                        # Provenance. Without it two teachers share one cache
                        # and a teacher comparison reads its own first arm.
                        "sf_id": sf_id,
                        "cp": None if res.cp is None else int(res.cp),
                        "mate": res.mate,
                        "wdl": None if res.wdl is None else [float(v) for v in res.wdl],
                        "pvs": [
                            {"move": pv.move_uci,
                             "cp": None if pv.cp is None else int(pv.cp),
                             "mate": pv.mate,
                             "wdl": None if pv.wdl is None else [float(v) for v in pv.wdl]}
                            for pv in (res.pvs or [])
                        ],
                    }
                    with lock:
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        cache[pos.key] = rec
                        n_done += 1
                        if n_done % 50 == 0:
                            rate = n_done / max(1e-9, time.time() - t0)
                            print(f"[sf-soft] {n_done}/{len(todo)} ({rate:.2f} pos/s)")

            with ThreadPoolExecutor(max_workers=len(engines)) as pool:
                for fut in [pool.submit(run_worker, wi) for wi in range(len(engines))]:
                    fut.result()
    finally:
        for eng in engines:
            eng.close()
    return cache


@dataclasses.dataclass(frozen=True)
class _SfSoftParams:
    sf_policy_temp: float
    sf_policy_label_smooth: float
    sf_wdl_use_cp_logistic: bool
    sf_wdl_cp_slope: float
    sf_wdl_cp_draw_width: float
    # Mirrors GameConfig: "cp" scores candidates as raw effective centipawns
    # at sf_policy_cp_temp. The audit-first ruler must follow the production
    # score mode or it silently audits the OTHER candidate under this name.
    sf_policy_score_mode: str = "wdl"
    sf_policy_cp_temp: float = 16.2


def _sf_soft_params_from_flat(flat: dict) -> _SfSoftParams:
    """Flattened run config → ruler params. One home for the yaml read so the
    score-mode wiring is testable — F1 was exactly this read going missing
    while the ruler kept scoring the OTHER candidate under the same name."""
    return _SfSoftParams(
        sf_policy_temp=float(flat.get("sf_policy_temp", 0.25)),
        sf_policy_label_smooth=float(flat.get("sf_policy_label_smooth", 0.05)),
        sf_wdl_use_cp_logistic=bool(flat.get("sf_wdl_use_cp_logistic", False)),
        sf_wdl_cp_slope=float(flat.get("sf_wdl_cp_slope", 0.010)),
        sf_wdl_cp_draw_width=float(flat.get("sf_wdl_cp_draw_width", 60.0)),
        sf_policy_score_mode=str(flat.get("sf_policy_score_mode", "wdl")),
        sf_policy_cp_temp=float(flat.get("sf_policy_cp_temp", 16.2)),
    )


class _PvLike:
    """Adapter so cached shallow-SF rows feed the live _pv_wdl_score."""

    def __init__(self, d: dict) -> None:
        self.move_uci = str(d["move"])
        self.cp = d.get("cp")
        self.mate = d.get("mate")
        self.wdl = None if d.get("wdl") is None else np.asarray(d["wdl"], dtype=np.float32)


def _sf_soft_distribution(
    rec: dict, legal_idxs: np.ndarray, *, params: _SfSoftParams,
) -> np.ndarray:
    legal_set = {int(i) for i in legal_idxs}
    cand_idxs: list[int] = []
    cand_scores: list[float] = []
    for d in rec.get("pvs", []):
        pv = _PvLike(d)
        a = uci_to_policy_index(pv.move_uci, True)
        if a < 0 or a not in legal_set:
            continue
        if params.sf_policy_score_mode == "cp":
            score = _pv_cp_score(pv)
        else:
            score = _pv_wdl_score(
                pv,
                sf_wdl_use_cp_logistic=params.sf_wdl_use_cp_logistic,
                sf_wdl_cp_slope=params.sf_wdl_cp_slope,
                sf_wdl_cp_draw_width=params.sf_wdl_cp_draw_width,
            )
        if score is None:
            continue
        cand_idxs.append(a)
        cand_scores.append(float(score))
    if not cand_idxs:
        cand_idxs = [int(legal_idxs[0])]
        cand_scores = [0.0]
    full = _build_sf_policy_target(
        cand_idxs, cand_scores, legal_indices=legal_idxs,
        sf_policy_temp=(
            params.sf_policy_cp_temp
            if params.sf_policy_score_mode == "cp"
            else params.sf_policy_temp
        ),
        sf_policy_label_smooth=params.sf_policy_label_smooth,
    )
    return full[legal_idxs].astype(np.float64)


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------


def _aggregate(
    rows: list[dict], key: str,
) -> dict[tuple[str, str], tuple[float, float, int]]:
    """(group, candidate) -> (mean expected regret, mean top1 regret, n)."""
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        for grp in ("overall", PHASE_NAMES[row["phase"]], SOURCE_NAMES[row["source"]]):
            groups.setdefault((grp, row[key]), []).append((row["expected"], row["top1"]))
    return {
        k: (float(np.mean([v[0] for v in vals])),
            float(np.mean([v[1] for v in vals])), len(vals))
        for k, vals in groups.items()
    }


def _shape_stats(probs: list[np.ndarray]) -> tuple[float, float, float, float]:
    """(mean entropy nats, mean top-1, mean exp(entropy), fraction top-1 >= 0.99).

    The visit distribution IS the stored policy target, so its sharpness is a
    property of the training data, not a diagnostic. C17's duplicate leaves
    sharpen it twice over -- they pile visits onto the already-winning path,
    and the inflated max_visit raises the root q_scale that sharpens the Gumbel
    improved policy on top of that. Any fix therefore FLATTENS the target, and
    this is where to read how much.
    """
    ents: list[float] = []
    tops: list[float] = []
    for p in probs:
        q = np.asarray(p, dtype=np.float64)
        q = q[q > 0.0]
        if q.size == 0:
            continue
        ents.append(float(-(q * np.log(q)).sum()))
        tops.append(float(np.max(p)))
    if not ents:
        nan = float("nan")
        return (nan, nan, nan, nan)
    arr = np.asarray(ents)
    tarr = np.asarray(tops)
    return (
        float(arr.mean()), float(tarr.mean()),
        float(np.exp(arr).mean()), float((tarr >= 0.99).mean()),
    )


def candidate_labels(input_encoding: str) -> dict[str, str]:
    """Candidate labels with each row's OWN input encoding baked in.

    Only row (a) reads `--input-encoding`; the searches encode internally from
    the board and Stockfish has no net input at all. A header naming one
    encoding for the whole report would be false for four of five rows, so the
    encoding is carried per row instead.
    """
    labels: dict[str, str] = {}
    for cand, label in _CANDIDATE_NAMES.items():
        if cand == "sf_soft":
            labels[cand] = f"{label} [no net input]"
        elif cand == "raw":
            labels[cand] = f"{label} [enc={input_encoding}]"
        else:
            labels[cand] = f"{label} [enc=fen_only, search-internal]"
    return labels


def _parse_blunder_taus(spec: str | None) -> tuple[float, ...]:
    """"50,100,200" -> (50.0, 100.0, 200.0). None/empty -> () (feature OFF).

    Deduplicated and sorted so the report columns and the dump keys are in a
    stable order no matter how the operator typed them.

    WHOLE CENTIPAWNS ONLY, enforced rather than assumed. A fractional
    threshold would name its dump key `blunder12.5`, and the dot is a
    separator in `paired_compare.py`'s dotted `--field` path — so
    `cand.raw.blunder12.5` resolves to nothing and every joined row is
    unusable. That failure is loud (the reader exits non-zero) but it happens
    only after a full GPU scoring pass, so it is rejected here at parse time
    instead.
    """
    if spec is None:
        return ()
    out: set[float] = set()
    for chunk in spec.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            tau = float(text)
        except ValueError:
            raise SystemExit(
                f"--blunder-taus: {text!r} is not a number (expected cp "
                f"thresholds like '50,100,200')",
            ) from None
        if not np.isfinite(tau) or tau < 0.0:
            raise SystemExit(
                f"--blunder-taus: {tau} is not a valid cp threshold "
                f"(must be finite and >= 0)",
            )
        if tau != round(tau):
            raise SystemExit(
                f"--blunder-taus: {text!r} must be a whole number of "
                f"centipawns. A fractional threshold names the dump key "
                f"'blunder{tau:g}', whose dot paired_compare.py's dotted "
                f"--field path cannot address, so every paired row would be "
                f"unusable.",
            )
        out.add(tau)
    return tuple(sorted(out))


def _blunder_key(tau: float) -> str:
    """Per-position dump key for one threshold: 100.0 -> 'blunder100'.

    Flat (not nested under a 'blunder' dict) because the paired reader
    addresses it as a dotted path: `paired_compare.py --field
    cand.raw.blunder100`. That same reader is why `_parse_blunder_taus`
    admits only whole centipawns: a dot in the key would split the path.
    """
    return f"blunder{tau:g}"


def _aggregate_blunders(
    rows: list[dict], n_taus: int,
) -> dict[tuple[str, str], tuple[tuple[float, ...], int]]:
    """(group, candidate) -> (mean blunder rate per tau, n).

    Separate from ``_aggregate`` on purpose: that function returns the frozen
    3-tuple every existing caller unpacks, and widening it would change the
    default reporting path this flag is required not to touch.
    """
    groups: dict[tuple[str, str], list[tuple[float, ...]]] = {}
    for row in rows:
        for grp in ("overall", PHASE_NAMES[row["phase"]], SOURCE_NAMES[row["source"]]):
            groups.setdefault((grp, row["cand"]), []).append(row["rates"])
    return {
        k: (
            tuple(
                float(np.mean([v[i] for v in vals])) for i in range(n_taus)
            ),
            len(vals),
        )
        for k, vals in groups.items()
    }


def _blunder_section(
    rows: list[dict],
    group_names: list[str],
    labels: dict[str, str],
    taus: tuple[float, ...],
) -> str:
    """The additive report section. EMPTY STRING when the flag was not passed.

    Returning "" rather than an empty table is what keeps the default report
    byte-identical: the caller concatenates unconditionally.
    """
    if not taus or not rows:
        return ""
    agg_b = _aggregate_blunders(rows, len(taus))
    lines = [
        "| candidate | " + " | ".join(
            f"{g} (n)" for g in group_names
        ) + " |",
        "|" + "---|" * (len(group_names) + 1),
    ]
    body: list[str] = []
    for cand, label in labels.items():
        for ti, tau in enumerate(taus):
            cells = []
            for g in group_names:
                v = agg_b.get((g, cand))
                cells.append(
                    "—" if v is None else f"{100.0 * v[0][ti]:.2f}% ({v[1]})",
                )
            body.append(f"| {label} — >{tau:g} cp | " + " | ".join(cells) + " |")
    if not body:
        return ""
    return (
        "\n## Blunder mass: target probability on moves losing more than N cp\n\n"
        "**A CONJUGACY-NEUTRAL ruler, and the reason it is here.** The E[regret] "
        "table above is LINEAR in the per-move cp cost, so among distributions of "
        "equal entropy its minimizer is exactly the Gibbs distribution in that "
        "cost. A candidate built as a softmax over cp therefore wins that table by "
        "construction, and one built over a saturating transform of cp loses it by "
        "construction — measured 2026-08-04 (ledger `b260373c5`), where the same "
        "pair swaps places when re-scored in win-probability units. This statistic "
        "is a 0/1 functional whose fixed-entropy minimizer is two-level, so it is "
        "conjugate to neither shape, and it reads the quantity collapse-avoidance "
        "is about: how much mass sits on moves that lose outright.\n\n"
        "Lower is better. Cells are mean probability mass, in percent.\n\n"
        + "\n".join(lines + body)
        + "\n"
    )


def _policy_table(agg: dict, group_names: list[str], labels: dict[str, str]) -> str:
    lines = ["| candidate | " + " | ".join(f"{g} E[regret] / top-1 (n)" for g in group_names) + " |"]
    lines.append("|" + "---|" * (len(group_names) + 1))
    for cand, label in labels.items():
        cells = []
        for g in group_names:
            v = agg.get((g, cand))
            cells.append("—" if v is None else f"{v[0]:.1f} / {v[1]:.1f} ({v[2]})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    add_net_source_args(
        ap,
        checkpoint_help="one of ours: trainer.pt or checkpoint dir. Mutually "
        "exclusive with --onnx; exactly one is required.",
    )
    ap.add_argument("--config", type=Path, default=None,
                    help="production config for target-construction params. "
                         "Default: the SAME file the shape guard resolves as "
                         f"production (${LIVE_CONFIG_ENV}, else the in-tree "
                         "configs/pbt2_small.yaml). ⚑ The old default was the "
                         "CWD-relative literal 'configs/pbt2_small.yaml' while "
                         "the reference it is checked against is module-"
                         "relative, so running from anywhere but the repo root "
                         "audited one file and named another.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="net forward batch. Raw policy regret is BATCH-SIZE "
                         "DEPENDENT (~0.8 cp between 64 and 256); pin it across "
                         "every arm of a comparison. Echoed in the report header. "
                         "⚑ IT IS ALSO BOARDS-PER-SEARCH-CALL, which is a SEARCH "
                         "SHAPE, not just a throughput knob: it sets how many trees "
                         "share one leaf-accumulation batch (see --target-batch), and "
                         "at >= 64 boards the C runner additionally takes the 2-group "
                         "eval pipeline (`_use_pipeline`, mcts/gumbel_c.py) that "
                         "production distributed selfplay never reaches, because "
                         "SlotInferenceClient has no evaluate_encoded_async. "
                         "Production runs ~1 board per call.")
    ap.add_argument("--input-encoding", choices=INPUT_ENCODINGS,
                    default=INPUT_ENCODING_DEFAULT,
                    help="input encoding for row (a), the net's raw policy. "
                         "'fen_only' (DEFAULT, bit-identical to every historical "
                         "run) rebuilds it from chess.Board(fen), leaving 93 of 175 "
                         "planes zero and the colour flag wrong on ~51%% of rows. "
                         "'stored' is audit-v2: the real production input. Rows "
                         "(b)/(d)/(e) are searches that encode internally and stay "
                         "fen_only either way; every table row says which it used. "
                         "⚑ DIFFERENT RULERS — never mix them in one table.")
    ap.add_argument("--matched-rows", type=Path, default=None,
                    help="matched-rows index for --input-encoding stored "
                         "(default: <audit-set>.matched_rows.npz); built by "
                         "scripts/match_audit_rows.py, not checked in")
    ap.add_argument("--sims", type=int, default=256,
                    help="sim budget for the PLAY row (b) only. The training rows "
                         "(d)/(e) follow the config's mcts_simulations / "
                         "fast_simulations unless --rl-sims overrides them. Recorded "
                         "in the report header and the per-position dump.")
    ap.add_argument("--allow-stale-config", action="store_true",
                    help="proceed when --config disagrees with the live production "
                         "yaml on a search key. For deliberately auditing a historical "
                         "or experimental config; the mismatch is printed either way.")
    ap.add_argument("--policy-temp", type=float, default=1.0,
                    help="prior temperature on policy logits before gumbel search "
                         "(>1 softens prior, <1 sharpens, 1.0=no-op). Measures search-prior "
                         "calibration on the REAL audit-set distribution (vs puzzle bias). "
                         "PLAY row (b) ONLY: the training rows take gumbel_policy_temp "
                         "from the production config, because a target row scored at a "
                         "temperature selfplay does not use is not a production target.")
    ap.add_argument("--gumbel", action="append", default=None, metavar="k=v",
                    help="override any GumbelConfig field on the PLAY row (b). "
                         "Repeatable, and comma-separated pairs are accepted: "
                         "--gumbel policy_temp=2.2 --gumbel topk=8,halving_div=4. "
                         "Keys are raw GumbelConfig field names, IDENTICAL to "
                         "scripts/arena_standard.py --cand-gumbel/--ref-gumbel, so a "
                         "shape moves between the two by copy-paste (the UCI engine "
                         "CamelCases the same fields: c_scale -> CScale; see "
                         "docs/operations.md). c_puct/cpuct_factor/cpuct_base/"
                         "fpu_reduction are REJECTED: inert in a Gumbel search. "
                         "The override is asserted against the config actually handed "
                         "to the runner, so a knob that fails to plumb aborts the run "
                         "instead of producing a clean null.")
    ap.add_argument("--gumbel-training-rows", action="store_true",
                    help="also apply --gumbel to the TRAINING rows (d)/(e). OFF by "
                         "default: those rows exist to describe the target production "
                         "actually stores, and silently reshaping them would print a "
                         "search selfplay never runs under the heading 'production "
                         "training target'.")
    ap.add_argument("--gumbel-topk", type=int, default=None,
                    help="Override the PLAY row's Gumbel root candidate count. "
                         "Default None = the PLAY default (32). The TRAINING rows "
                         "always take selfplay's value from --config (16) and are "
                         "NOT affected by this flag -- overriding the target's own "
                         "topk would score a search selfplay never runs. At 256 "
                         "sims, ~30 legal moves means topk=32 ≈ all-legal.")
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="cap this process to a fraction of GPU memory so a high-sim "
                         "audit run CONCURRENT with a live trainer fails-fast on its own "
                         "OOM instead of faulting the shared GPU/broker. e.g. 0.4 on a "
                         "32GB card. Applied to BOTH allocators a run can use: the torch "
                         "caching allocator (set_per_process_memory_fraction) and, on "
                         "--onnx, onnxruntime's CUDA arena (gpu_mem_limit, computed "
                         "against the card's total memory). The two are separate -- "
                         "torch's cap does not bound an ORT session.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stockfish", type=str, default=None,
                    help="needed only when the shallow-SF cache is incomplete")
    ap.add_argument("--sf-soft-nodes", type=int, default=None,
                    help="explicit shallow-SF node count; overrides --sf-effort when set")
    ap.add_argument("--sf-effort", choices=("low", "high"), default="low",
                    help="shallow-SF strength tier when --sf-soft-nodes is unset: "
                         "low=500k (matches the production teacher), high=2M (deeper reference). "
                         "The 50k default was retired once production moved to 500k nodes. NOTE: "
                         "the cache is keyed by node count, so switching tiers needs --stockfish to "
                         "(re)label at the new count.")
    ap.add_argument("--sf-cache", type=Path, default=None,
                    help="write/read the shallow-SF label cache HERE instead of "
                         "<audit-set>.shallow_sf.jsonl. ⚑ REQUIRED for a repeat "
                         "control: two runs of the same engine against the "
                         "shared cache do not re-label at all — run 2 is served "
                         "run 1's rows verbatim, so the measured run-to-run "
                         "variance is 0 by construction, not by determinism.")
    ap.add_argument("--sf-soft-multipv", type=int, default=40)
    ap.add_argument("--sf-workers", type=int, default=4)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--target-batch", type=int, default=None,
                    help="C-search leaf-accumulation batch. DEFAULT (unset) = whatever "
                         "gumbel_target_batch the resolved production config sets, so the "
                         "training rows search production's shape. 1 = flush per rep, which "
                         "removes C17's duplicate leaves (29-76%% at 256 sims, -34%% tree nodes). "
                         "Run the audit at production's value and at 1 to separate 'C17 wastes "
                         "compute' from 'C17 corrupts the training target': duplicate visits "
                         "still increment N, inflating max_visit and hence the root q_scale that "
                         "sharpens the improved-policy target. An explicit value is reported as a "
                         "DELIBERATE deviation and lands on the search_shape ruler stamp. "
                         "C-runner only; the Python reference path takes no such argument.")
    ap.add_argument("--vloss-mode", type=int, default=0, choices=(0, 1),
                    help="How an in-flight walker is VALUED when --vloss-weight > 0. "
                         "0 = LEGACY, the parallel-PUCT construct: the pending visit is "
                         "scored as a LOSS, biasing the child down. 0 is also what "
                         "PRODUCTION runs — selfplay never passes vloss_mode, so the C "
                         "runner's LEGACY default stands. 1 = VIRTUAL_MEAN: it "
                         "is valued at the child's existing mean, so the visit count "
                         "moves and the estimate does not. "
                         "*** 1 CURRENTLY RAISES: tree_gumbel_select_child mirrors "
                         "VIRTUAL_MEAN for the CHILD term only, leaving parent_Q -- the "
                         "FPU for every unvisited child -- with legacy pessimism, so the "
                         "mode does not do what this help says (play-path audit "
                         "2026-08-03, F4). A comparison run made through it would be a "
                         "verdict off a broken instrument. Re-enable in the commit that "
                         "mirrors the C parent branch. ***")
    ap.add_argument("--vloss-weight", type=int, default=None,
                    help="C-search virtual-loss weight. DEFAULT (unset) = whatever "
                         "gumbel_vloss_weight the resolved production config sets. ⚑ The old "
                         "default was the LITERAL 0 while production runs 1, so every ordinary "
                         "invocation scored the duplicate-leaf search under a header reading "
                         "'production training target'. 0 = a leaf already awaiting eval in the "
                         "current batch carries no penalty and a later halving rep re-walks "
                         "straight back to it (C17). >0 makes in-flight leaves count as "
                         "penalized visits during descent, removing the duplicates WITHOUT "
                         "giving up the cross-rep batching that --target-batch 1 has to give "
                         "up. An explicit value is reported as a DELIBERATE deviation and lands "
                         "on the search_shape ruler stamp. C-runner only.")
    ap.add_argument("--rl-sims", type=int, default=0,
                    help="override the TRAINING rows' sim budget (default: the config's "
                         "mcts_simulations). The node-matched control for --target-batch / "
                         "--vloss-weight: those buy ~60%% more distinct nodes per nominal "
                         "sim, so run the production arm at the matched node count to "
                         "separate 'less duplication' from 'more search'. Does not touch "
                         "the PLAY row (--sims). 0 is a sentinel meaning 'use the "
                         "config'; the report header and the per-position dump record "
                         "the RESOLVED value, never the sentinel.")
    ap.add_argument("--max-positions", type=int, default=0,
                    help=">0 limits positions (smoke runs)")
    ap.add_argument("--blunder-taus", type=str, default=None,
                    help="comma-separated cp thresholds, e.g. '50,100,200'. "
                         "ADDITIVE: adds a blunder-mass report section and "
                         "per-candidate 'blunderN' keys to the dump. Omitted "
                         "(the default) leaves every existing number and the "
                         "report byte-identical.")
    ap.add_argument("--dump-per-position", type=Path, default=None,
                    help="if set, write one JSONL record per scored position "
                         "(phase, source, criticality gap, per-candidate "
                         "expected/top1 regret, chosen move, and the paired "
                         "per-position booleans top1_agree / out_of_top10) for "
                         "offline slicing and PAIRED statistics")
    ap.add_argument("--dump-distributions", action="store_true",
                    help="with --dump-per-position, also record each candidate's "
                         "full distribution over legal moves as {uci: p} (entries "
                         "below 1e-6 pruned). Roughly 30-40 floats per candidate "
                         "per position, so the dump grows ~20x; needed only when "
                         "the downstream statistic is not one of the booleans "
                         "already emitted.")
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
  # Resolved and PRINTED here, before the checkpoint loads, for the same reason
  # the flag checks below are: the operator of a repeat control needs to see
  # WHICH cache this run will use while there is still time to stop it. Printing
  # it next to the labelling pass would be an hour too late, and printing the
  # flag rather than the resolved path is how a defaulted override reads as an
  # applied one.
    sf_cache_path = resolve_sf_cache_path(
        args.audit_set, args.sf_cache, args.dump_per_position,
    )
    print(f"[sf-soft] cache {sf_cache_path}"
          f"{' (--sf-cache override)' if args.sf_cache else ''}")
  # Same fail-fast reasoning as the flags below, and the most expensive one to
  # get wrong: exactly one of --checkpoint/--onnx, and for --onnx the graph's
  # tensor names, resolved and printed HERE -- before the audit set loads and
  # long before SF spends an hour labelling.
    net = net_source_from_args(args)
    reject_stored_encoding_for_onnx(net, args.input_encoding)
  # Parsed at PARSE time so a malformed threshold list fails before the model
  # is loaded. () when the flag is absent, which is what keeps every code path
  # below identical to the default run.
    blunder_taus = _parse_blunder_taus(args.blunder_taus)
  # Same reasoning: a bad --gumbel key must not surface after the checkpoint
  # has loaded and Stockfish has spent an hour labelling.
  # Fail-fast only: a bad --gumbel key must surface here, not after the
  # checkpoint has loaded and SF has spent an hour labelling. The value is NOT
  # kept -- `profiles_for_audit` re-derives it from `args` below, so there is no
  # variable in `main()` that a later edit can forget to forward.
    parse_gumbel_overrides(args.gumbel)
  # ⚑ THE DEDICATED FLAG GETS THE SAME BAND AS THE OVERRIDE, or this PR creates
  # the false record it exists to remove. `--gumbel policy_temp=X` has been
  # refused outside [POLICY_TEMP_MIN, POLICY_TEMP_MAX] since the override guard
  # landed, for exactly the reason quoted there: `apply_policy_temp` silently
  # swallows an out-of-band value, so the audit scores the DEFAULT prior. The
  # dedicated `--policy-temp` reaches the same `GumbelConfig.policy_temp` and
  # had no such check, which was survivable only while nothing recorded it:
  # `--policy-temp 42.5` and `--policy-temp 1.0` produce dumps identical on
  # every field. Now that the value is STAMPED, accepting it would print 42.5
  # in the header of a report that scored 1.0 — provenance that is false, which
  # this file's own thesis says is worse than provenance that is missing.
  # Same guard, same instrument, so the two cannot drift apart.
    if args.dump_distributions and args.dump_per_position is None:
        raise SystemExit(
            "--dump-distributions needs --dump-per-position (it adds a field "
            "to that dump; on its own it would silently do nothing)"
        )

  # Reject at PARSE time, not deep in the run. `_net_candidates` only forwards
  # vloss_mode when vloss_weight > 0, so `--vloss-mode 1` at the default
  # `--vloss-weight 0` used to be accepted, dropped, and leave no trace -- the
  # exact "value accepted and then silently ignored" pattern, sitting in the flag
  # it had just re-documented. (`search_param_stamp` now records the requested
  # mode in the header and the dump, so it no longer leaves NO trace -- but a
  # recorded value that the search dropped is still the wrong outcome, and this
  # guard is what makes the recorded mode the one the search ran.)
  # With a weight the search DOES raise, but only after the audit set, the
  # checkpoint and the evaluator have loaded. Failing here
  # makes the help text true for every flag combination and the failure cheap.
  # Local import: this module keeps the mcts/C-extension import lazy.
    from chess_anti_engine.mcts.gumbel_c import VLOSS_MODE_VIRTUAL_MEAN

    if int(args.vloss_mode) == VLOSS_MODE_VIRTUAL_MEAN:
        raise SystemExit(
            "--vloss-mode 1 (VIRTUAL_MEAN) is refused: tree_gumbel_select_child "
            "mirrors that mode for the CHILD term only, leaving parent_Q -- the FPU "
            "for every unvisited child -- with legacy virtual-loss pessimism, so the "
            "mode does not do what --vloss-mode's help describes (play-path audit "
            "2026-08-03, F4). Comparing the two constructs through it would be a "
            "verdict off a broken instrument. Re-enable in the commit that mirrors "
            "the C parent branch."
        )

  # Same shape, and it only became a defect once the value was recorded:
  # `_net_candidates` forwards the weight under `if vloss_weight > 0`, so a
  # NEGATIVE weight is dropped exactly like `--vloss-mode 1` at weight 0 was --
  # and the stamp would now print it in the header of a report whose search ran
  # at 0. Refuse rather than stamp a value the search discarded.
  # ⚑ `is not None` FIRST. `--vloss-weight` defaults to None ("inherit whatever
  # `gumbel_vloss_weight` the resolved production config sets"), not to the
  # literal 0 it used to, so a bare `int(args.vloss_weight)` raises TypeError on
  # EVERY ordinary invocation. The None path is not unguarded: it never reaches
  # the runner as an operator value at all — `build_search_profiles` resolves it
  # from the production shape and `_assert_production_shape` compares the result.
    if args.vloss_weight is not None and int(args.vloss_weight) < 0:
        raise SystemExit(
            f"--vloss-weight {args.vloss_weight} is refused: the C runner is "
            "only handed a weight when it is > 0, so a negative value is "
            "silently dropped and the search runs at 0 — while the report "
            "header and the per-position dump would record your number. Pass 0 "
            "for no virtual loss, or a positive weight."
        )

    if args.sf_soft_nodes is None:
        args.sf_soft_nodes = {"low": 500_000, "high": 2_000_000}[args.sf_effort]

    # Caps the TORCH allocator and says only that. The --onnx session gets its
    # own cap at load time via ORT's gpu_mem_limit; torch's fraction cannot
    # reach it, and printing a bare "GPU memory capped" here claimed it could.
    apply_gpu_mem_cap(
        net=net,
        device=str(args.device),
        gpu_mem_fraction=args.gpu_mem_fraction,
        tag="audit",
    )

  # Resolved here rather than in `add_argument`, so the default is the file the
  # guard below will compare against instead of a CWD-relative literal that
  # only coincides with it when the operator happens to be at the repo root.
    if args.config is None:
        args.config = resolve_live_config_path()[0]
    flat, config_authority = load_audit_config(
        str(args.config), allow_stale=bool(args.allow_stale_config),
    )
    sf_params = _sf_soft_params_from_flat(flat)
    train_temp = float(flat.get("temperature", 1.0))
    sf_wdl_frac = float(flat.get("sf_wdl_frac", 0.0))
    search_wdl_frac = float(flat.get("search_wdl_frac", 0.0))

    encoding = normalize_input_encoding(args.input_encoding)
    positions = load_audit_set(args.audit_set)
    # Digested beside the READ, not at write time: the SF labelling pass below
    # runs for up to an hour, and a digest taken afterwards describes whatever
    # is on disk then rather than what was scored. (Codex inline review, #442.)
    set_provenance = audit_set_provenance(args.audit_set)
    if args.max_positions > 0:
        positions = positions[: args.max_positions]

    stored_x: np.ndarray | None = None
    if encoding == "stored":
        matched = MatchedAuditRows(
            args.matched_rows or default_matched_rows_path(args.audit_set)
        )
        n_before = len(positions)
        positions = [p for p in positions if p.key in matched]
        if not positions:
            raise SystemExit(
                "no audit position has a stored row; the matched-rows index does "
                "not cover this audit set"
            )
        # The other two callers reach this through `require_model_compatible`;
        # this script loads its checkpoint deeper in the call stack, so it runs
        # the index-side half here and the model-side half in `_net_candidates`.
        matched.require_index_layout()
        stored_x = np.stack([matched.stored_row(p.key) for p in positions])
        print(f"[audit] input-encoding=stored: {matched.path} covers "
              f"{matched.n_matched}/{matched.n_audit_rows} audit rows; dropped "
              f"{n_before - len(positions)} unmatched ({len(positions)} kept). "
              f"⚑ RULER CHANGE — row (a) is NOT comparable to a fen_only run.")
    boards = [chess.Board(p.fen) for p in positions]
    print(f"[audit] [enc={encoding} b={args.batch_size}] {len(positions)} positions "
          f"from {args.audit_set}")

    shallow = _shallow_sf_records(
        positions,
        cache_path=sf_cache_path,
        stockfish=args.stockfish, nodes=int(args.sf_soft_nodes),
        multipv=int(args.sf_soft_multipv), workers=int(args.sf_workers),
        nice=int(args.nice),
    )

    from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS

    full_share = float(flat.get("playout_cap_fraction", 1.0))
    profiles, gumbel_overrides = profiles_for_audit(args, flat)
    rl_c_scale = profiles["train"].c_scale
  # Built ONCE, here, and handed to both the report header and the per-position
  # dump. Building it twice is how a stamp starts disagreeing with the run.
    search_params = search_param_stamp(args, profiles=profiles)
  # ⚑ READ THE SIM COUNTS OFF THE STAMP, never off `profiles[...].sims` or
  # `args.sims`. `--gumbel simulations=N` lands AFTER the profile columns, so
  # the columns are pre-override: at `--sims 8 --gumbel simulations=17
  # --gumbel-training-rows` all three rows searched at 17 while this header line
  # printed "PLAY 8 sims / RL train 8 full + 32 fast" one line above a stamp
  # reading 17 (independent review of PR #434, executed). A report that
  # contradicts itself is worse than one that omits — and the WRONG line is the
  # one an operator greps.
    play_sims_note = search_params["sims"]
    rl_sims = search_params["rl_sims"]
    rl_fast_sims = search_params["fast_sims"]
    for name, prof in profiles.items():
        print(
            f"[audit] {_CANDIDATE_NAMES[name]}: {prof.label} — "
            f"sims={prof.sims} topk={prof.topk} c_scale={prof.c_scale} "
            f"root={'log' if prof.q_visit_exp_root < 0 else 'linear'} "
          # The three #227 fields, on the header line: a target row that does
          # not print cap/untempered is a row from a build that predates the
          # fix, and the distinction matters because the numbers moved.
            f"policy_temp={prof.policy_temp} "
            f"target_cap={prof.target_max_visit_cap} "
            f"untempered_prior={prof.target_untempered_prior} "
          # The two runner arguments that are not GumbelConfig fields (F1).
          # On the header for the same reason as the three above: a row that
          # does not print them is a row from a build that could not see them.
            f"vloss_weight={prof.vloss_weight} target_batch={prof.target_batch}",
            flush=True,
        )
  # The PLAY row is NOT the play shape on these two. Stated rather than left to
  # be assumed away: `PLAY_SEARCH_VLOSS_WEIGHT` is 3, row (b) runs the CLI value
  # (default 0), and moving it would move a standing ruler with banked
  # readings — which needs its own ledger entry, not a drive-by in this fix.
    from chess_anti_engine.mcts.gumbel import (
        PLAY_SEARCH_TARGET_BATCH,
        PLAY_SEARCH_VLOSS_WEIGHT,
    )
    if (
        profiles["search"].vloss_weight != int(PLAY_SEARCH_VLOSS_WEIGHT)
        or profiles["search"].target_batch != int(PLAY_SEARCH_TARGET_BATCH)
    ):
        print(
            f"[shape] {_CANDIDATE_NAMES['search']}: DELIBERATE deviation from "
            f"the PLAY shape — vloss_weight={profiles['search'].vloss_weight} "
            f"target_batch={profiles['search'].target_batch} vs play's "
            f"{int(PLAY_SEARCH_VLOSS_WEIGHT)}/{int(PLAY_SEARCH_TARGET_BATCH)}. "
            "Row (b) is a standing ruler with banked readings; changing its "
            "search needs its own ledger entry. Read it as 'net + Gumbel "
            "search at the audit's fixed settings', not as UCI play strength.",
            flush=True,
        )
    print(
        "[audit] search params: "
        f"{format_search_params(search_params, gumbel_overrides=gumbel_overrides)}",
        flush=True,
    )

  # Production probes tablebases inside the search; the audited target has to
  # as well or the endgame bucket describes a search production never runs.
    sz_path = str(flat.get("syzygy_path") or "") if flat.get("syzygy_in_search") else ""

    (
        raw_probs, search_by_profile, root_q_by_profile, root_wdl, realized_shapes,
    ) = _net_candidates(
        boards, net=net, device=args.device,
        batch_size=int(args.batch_size), seed=int(args.seed),
        profiles=profiles, requested_gumbel_overrides=gumbel_overrides,
        policy_temp=float(args.policy_temp),
        syzygy_path=sz_path or None,
      # target_batch / vloss_weight are NOT passed: they ride on the profiles,
      # where production's values are the default and the guard can see them.
        vloss_mode=int(args.vloss_mode),
        stored_x=stored_x,
        gpu_mem_fraction=args.gpu_mem_fraction,
    )
    search_probs = search_by_profile["search"]
  # The production WDL blend's search component comes from the RL search, so
  # value candidate (iv) must read the RL root Q, not the play-path one.
    root_q = root_q_by_profile["train"]

    policy_rows: list[dict] = []
    # Populated only when --blunder-taus is passed; an empty list makes
    # `_blunder_section` return "" and the report identical to the default.
    blunder_rows: list[dict] = []
    per_pos_dump: list[dict] = []
  # Distribution shape per candidate, over the SAME rows the regret table
  # scores. Collected from the same `cands` dict so the stored-form transforms
  # (soft-policy temp, legal renormalisation) are already applied.
    shape_probs: dict[str, list[np.ndarray]] = {k: [] for k in _CANDIDATE_NAMES}
    value_rows: dict[str, list[np.ndarray]] = {k: [] for k in _VALUE_NAMES}
    deep_wdls: list[np.ndarray] = []
    # Rows can be skipped (no encodable legal moves); every per-row list below
    # must stay aligned with kept_positions, NOT with the input order.
    kept_positions: list[AuditPosition] = []
    outcome_idx: list[int] = []
    for i, (pos, board) in enumerate(zip(positions, boards, strict=True)):
        legal_ucis, legal_idxs = legal_full_indices(board)
        if not legal_ucis:
            continue
        regrets = move_regrets(pos, legal_ucis)
      # Deep-SF reference sets for the paired per-position booleans below.
      # `top1` is a SET, not a single move: SF's MultiPV list routinely holds
      # several moves at the same cp, and calling a candidate wrong for
      # picking one of the co-best would measure tie-breaking, not agreement.
      # See `sf_reference_sets`.
        sf_top1_set, sf_top10_set = sf_reference_sets(pos.move_cp)
        def _as_stored(probs: np.ndarray) -> np.ndarray:
            # policy_t is the visit distribution at the move-selection
            # temperature; production temperature 0.0 (and 1.0) store the
            # raw visit distribution -- the temperature then only shapes
            # action SAMPLING, not the stored target.
            if train_temp <= 0.0 or train_temp == 1.0:
                return probs
            return apply_policy_temperature(
                probs.astype(np.float32), train_temp,
            ).astype(np.float64)

        cands = {
            "raw": raw_probs[i],
            "search": search_probs[i],
            "train": _as_stored(search_by_profile["train"][i]),
            "train_fast": _as_stored(search_by_profile["train_fast"][i]),
            "sf_soft": _sf_soft_distribution(
                shallow[pos.key], legal_idxs, params=sf_params,
            ),
        }
        per_cand: dict[str, dict] = {}
        for cand, probs in cands.items():
            shape_probs[cand].append(np.asarray(probs, dtype=np.float64))
            exp_r, top1_r = expected_and_top1_regret(probs, regrets)
            policy_rows.append({
                "cand": cand, "phase": pos.phase, "source": pos.source,
                "expected": exp_r, "top1": top1_r,
            })
            top_i = int(np.argmax(probs))
            pv = np.asarray(probs, dtype=np.float64)
            pv = pv / max(1e-12, pv.sum())
            entropy = float(-(pv[pv > 0] * np.log(pv[pv > 0])).sum())
            chosen = legal_ucis[top_i]
            per_cand[cand] = {
                "exp": exp_r, "top1": top1_r,
                "move": chosen, "p": float(probs[top_i]),
                "entropy": entropy,
              # PAIRED per-position booleans. Computed here, once, rather than
              # left to each downstream script: `out_of_top10` in particular
              # depends on how ties and short MultiPV lists are handled, and two
              # callers reimplementing it would silently disagree. Both are
              # per-POSITION and per-CANDIDATE, so `paired_compare.py` can join
              # two dumps on `key` and difference them without re-deriving
              # anything. Note the denominator is a POSITION-level property
              # (the deep-SF list), never conditioned on the candidate's own
              # answer.
                "top1_agree": bool(chosen in sf_top1_set),
                "out_of_top10": (
                    None if not sf_top10_set else bool(chosen not in sf_top10_set)
                ),
            }
            if args.dump_distributions:
                pd = np.asarray(probs, dtype=np.float64)
                per_cand[cand]["probs"] = {
                    u: float(pd[k]) for k, u in enumerate(legal_ucis)
                    if pd[k] >= 1e-6
                }
            if blunder_taus:
                rates = expected_blunder_rates(probs, regrets, blunder_taus)
                per_cand[cand].update({
                    _blunder_key(t): r
                    for t, r in zip(blunder_taus, rates, strict=True)
                })
                blunder_rows.append({
                    "cand": cand, "phase": pos.phase,
                    "source": pos.source, "rates": rates,
                })
        if args.dump_per_position is not None:
            # Criticality = deep-SF gap between the best and 2nd-best listed line
            # (cp). Small gap = quiet position where SF's "best" is near-arbitrary
            # among near-equal moves; large gap = decision-critical. Shared with
            # foreign_net_audit (was bt4_audit, renamed in #414) /
            # audit_compare_buckets so the joined comparison agrees.
            gap = criticality_gap(pos.move_cp)
            per_pos_dump.append({
                "key": pos.key, "phase": pos.phase, "source": pos.source,
                # Which NET produced the row. The report header alone is not
                # enough: dumps outlive reports and get joined to each other,
                # and a checkpoint row and a foreign-ONNX row are otherwise
                # indistinguishable.
                "net": net.label,
                # A dump is a report: carry the ruler it was made with, or a
                # downstream join can silently mix two encodings.
                "input_encoding": {
                    c: (encoding if c == "raw" else
                        None if c == "sf_soft" else "fen_only")
                    for c in cands
                },
                # The SEARCH the row was produced by: --vloss-weight /
                # --vloss-mode / --target-batch are CLI-only and unrecoverable
                # from --config. `batch_size` lives in here rather than being
                # spelled out beside it, so the header and the dump cannot
                # disagree about it.
                **search_params,
                # ⚑ THE RULER STAMPS: rows (d)/(e)'s realized search shape (per
                # TRAINING ROW, read off the shape the runner was ACTUALLY
                # handed and never off the pre-override `_SearchProfile`),
                # whether the config behind them was PROVED to be the live one,
                # and the target-construction VALUES that authority verdict
                # cannot see across time. `dump_ruler_stamps` builds all three
                # so a test can RUN it — see its docstring.
                **dump_ruler_stamps(realized_shapes, config_authority),
                # null (not inf -> non-standard JSON "Infinity") for <2-move positions
                "gap_cp": float(gap) if np.isfinite(gap) else None,
                "n_legal": len(legal_ucis),
                "n_listed": len(pos.move_cp), "best_cp": float(pos.best_cp),
              # The reference the two booleans were judged against, carried so a
              # downstream join can verify it rather than assume it.
                "sf_top1": sorted(sf_top1_set),
                "sf_top10": sorted(sf_top10_set),
                "gumbel_overrides": dict(gumbel_overrides),
                "cand": per_cand,
            })

        rec = shallow[pos.key]
        sf_native = (
            np.asarray(rec["wdl"], dtype=np.float64) if rec.get("wdl") else
            np.array([333.0, 334.0, 333.0])
        )
        sf_native = np.clip(sf_native, 0.0, None)
        sf_native = sf_native / max(1e-9, sf_native.sum())
        if rec.get("cp") is not None or rec.get("mate"):
            cp_log = cp_to_wdl(
                rec.get("cp"), rec.get("mate"),
                slope=sf_params.sf_wdl_cp_slope,
                draw_width_cp=sf_params.sf_wdl_cp_draw_width,
            ).astype(np.float64)
        else:
            cp_log = sf_native
        search_root = _search_wdl_like_selfplay(root_q[i], root_wdl[i])
        # Production blend: outcome component only exists on outcome-labeled
        # rows; elsewhere the sf/search fractions are renormalized (this is
        # the same fallback shape the loss uses when a component is absent).
        w_sf, w_search = sf_wdl_frac, search_wdl_frac
        game_frac = max(0.0, 1.0 - w_sf - w_search)
        if pos.outcome is not None:
            onehot = np.zeros(3)
            onehot[int(pos.outcome)] = 1.0
            blend = game_frac * onehot + w_sf * sf_native + w_search * search_root
        else:
            denom = max(1e-9, w_sf + w_search)
            blend = (w_sf * sf_native + w_search * search_root) / denom
        value_rows["cp_logistic"].append(cp_log)
        value_rows["sf_native"].append(sf_native)
        value_rows["blend"].append(blend / max(1e-9, blend.sum()))
        value_rows["search_root"].append(search_root)
        deep_wdls.append(np.asarray(pos.deep_wdl, dtype=np.float64))
        kept_positions.append(pos)
        if pos.outcome is not None:
            outcome_idx.append(len(deep_wdls) - 1)

    if args.dump_per_position is not None:
        # Provenance-stamped, because this dump is a DERIVED cache in exactly
        # the sense `bt4_audit_cache.jsonl` was: its `gap_cp` and every
        # per-candidate regret come out of eval/audit.py's ruler, and none of
        # that is recoverable from the numbers. `audit_compare_buckets.py`
        # joins it against a BT4 cache and takes the criticality BUCKET for
        # every row from here, so an unstamped one silently sets the row
        # labelling for the other file's numbers too.
        #
        # force=True, unlike foreign_net_audit's --cache-out: this option has
        # no default path, so the operator always names the target explicitly
        # and there is no silent-default clobber to guard against. The hazard
        # that guard exists for is a DEFAULT pointing at a banked file.
        stamp = write_audit_cache(
            args.dump_per_position, per_pos_dump, force=True,
            extra={"producer": "audit_targets.py --dump-per-position",
                   "input_encoding": encoding,
                   **set_provenance},
        )
        print(f"[audit] per-position dump → {args.dump_per_position} "
              f"({len(per_pos_dump)} rows) [{stamp_summary(stamp)}]")

    agg = _aggregate(policy_rows, "cand")
    cand_labels = candidate_labels(encoding)
    group_names = ["overall", *PHASE_NAMES, *SOURCE_NAMES]
    deep = np.stack(deep_wdls)

    value_lines = [
        "| candidate | Brier vs deep WDL | ECE vs deep WDL | Brier vs outcome (n) |",
        "|---|---|---|---|",
    ]
    for key, label in _VALUE_NAMES.items():
        preds = np.stack(value_rows[key])
        brier = float(np.mean([wdl_brier(p, t) for p, t in zip(preds, deep, strict=True)]))
        ece = wdl_ece(preds, deep)
        if outcome_idx:
            oc = [
                wdl_brier(preds[i], np.eye(3)[kept_positions[i].outcome])
                for i in outcome_idx
            ]
            oc_cell = f"{float(np.mean(oc)):.4f} ({len(outcome_idx)})"
        else:
            oc_cell = "— (0)"
        value_lines.append(f"| {label} | {brier:.4f} | {ece:.4f} | {oc_cell} |")

    shape_lines = [
        "| search profile | entropy (nats) | mean top-1 | eff. support | frac top-1 >= 0.99 |",
        "|---|---|---|---|---|",
    ]
    for name, label in cand_labels.items():
        rows = shape_probs.get(name) or []
        if not rows:
            continue
        ent, top1, supp, frac99 = _shape_stats(rows)
        shape_lines.append(
            f"| {label} | {ent:.4f} | {top1:.4f} | {supp:.3f} | {frac99:.3f} |",
        )
    shape_table = "\n".join(shape_lines)

    sha = git_sha(short=True)
    out_path = args.out_dir / f"target_audit_{sha}.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    headline_search = agg.get(("overall", "search"))
    headline_sf = agg.get(("overall", "sf_soft"))
  # The stored POLICY corpus is full-sim rows ONLY -- it is NOT a playout-cap
  # mixture. `finalize.py` drops playout-capped rows outright by default, and
  # with `record_fast_ply_value` they become value-only rows whose MAIN policy
  # head is masked ("Fast plies never get SF label queries either way"). That
  # is KataGo's playout-cap design working as intended: cheap plies buy game
  # length and value coverage, never policy supervision. So the headline is the
  # full-sim row alone -- weighting it by playout_cap_fraction would invent a
  # mixture nothing stores and understate the target by ~9cp.
    headline_full = agg.get(("overall", "train"))
    headline_fast = agg.get(("overall", "train_fast"))
    train_note = "—" if headline_full is None else f"{headline_full[0]:.1f} cp"
    fast_note = (
        "—" if headline_fast is None
        else f"{headline_fast[0]:.1f} cp at {rl_fast_sims} sims"
    )
    report = (
        f"# Target audit @ {sha}\n\n"
        f"- audit set: {args.audit_set} ({len(deep_wdls)} scored positions)\n"
        f"- input encoding: **{encoding}** (row (a) only; batch-size "
        f"{args.batch_size}). Rows (b)/(d)/(e) encode inside the search and are "
        f"always fen_only; row (c) has no net input. ⚑ A RULER CHANGE INVALIDATES "
        f"ITS RECORDS — do not put row (a) from a `stored` run in a table with "
        f"row (a) from a `fen_only` run.\n"
        f"- net: {net.label}\n"
        f"- search: PLAY {play_sims_note} sims / RL train {rl_sims} full + {rl_fast_sims} fast "
        f"(playout_cap_fraction {full_share}); shallow SF: {args.sf_soft_nodes} nodes "
        f"MultiPV {args.sf_soft_multipv}; config: {args.config}\n"
      # The search params, greppable, in one line. They are CLI-only: --config
      # pins none of vloss_weight / vloss_mode / target_batch, so a report
      # without this line cannot be traced to the search that produced it.
      # ⚑ vloss_weight=0 is NOT production (the yaml runs 1) — see the flag's help.
        f"- search params: "
        f"{format_search_params(search_params, gumbel_overrides=gumbel_overrides)} "
        f"(vloss_weight/vloss_mode/target_batch have no --config source at all; "
        f"the sims/topk/policy_temp entries are the REALIZED values, after the "
        f"--rl-sims sentinel and any --gumbel override)\n\n"
        f"## Headline\n\n"
        f"- **production TRAINING target** expected regret (overall): {train_note} vs "
        f"SF-soft-target {'—' if headline_sf is None else f'{headline_sf[0]:.1f} cp'} — "
        f"this is the pair that prices whether {args.sf_soft_nodes}-node "
        f"MultiPV-{args.sf_soft_multipv} labeling is still worth its CPU bill, "
        f"because both sides are targets training actually stores "
        f"(per-phase split below).\n"
        f"- fast-ply (playout-capped) search: {fast_note} — reported for "
        f"reference only. Playout-capped plies carry NO policy target: "
        f"finalize.py drops them, and with record_fast_ply_value they become "
        f"value-only rows with the MAIN policy head masked. Do not average "
        f"this into the training-target number.\n"
        f"- PLAY-path search regret (overall): "
        f"{'—' if headline_search is None else f'{headline_search[0]:.1f} cp'} — "
        f"the UCI/TCEC number. NOT comparable to the SF soft target for the "
        f"labeling decision: it is a different search (c_scale "
        f"{PLAY_SEARCH_DEFAULTS['c_scale']} + log root vs RL's {rl_c_scale} + "
        f"linear root) and no training row is ever built from it.\n"
        f"- production WDL blend calibration vs its best single component: "
        f"see the value table.\n\n"
        f"## Policy: expected / top-1 deep-SF regret (cp)\n\n"
        f"Unlisted legal moves carry the worst-listed-line regret as a "
        f"floor (lower bound; MultiPV >= 10 at >=1M nodes).\n"
        f"**Row (c) is scored against a deeper version of itself** — the ruler is "
        f"deep SF and (c) is shallow SF — so part of its margin over (a)/(b)/(d) "
        f"is definitional, exactly as the value table warns for row (ii). "
        f"Same-material comparisons (search shape vs search shape, checkpoint vs "
        f"checkpoint) are sound; \"the SF target beats the training target\" is a "
        f"calibration reading, NOT a teaching verdict.\n\n"
        f"{_policy_table(agg, group_names, cand_labels)}\n\n"
        f"## Target distribution shape (the stored policy target's sharpness)\n\n"
        f"Row (d) is the distribution the policy head is trained on; row (c) is "
        f"the SF teacher it is blended against. Shape is a property of the "
        f"training DATA, not a diagnostic.\n\n"
        f"{shape_table}\n\n"
        f"## Value: calibration against deep-SF WDL\n\n"
        f"**This is a CALIBRATION ruler, not a target-quality ruler.** Row (ii) "
        f"is shallow SF native WDL and the reference is deep SF native WDL — the "
        f"same kind of object — so (ii) normally wins for a reason that has "
        f"nothing to do with being a better teacher, exactly as the policy table "
        f"warns for row (c). Production deliberately uses the softer cp-logistic "
        f"(`sf_wdl_use_cp_logistic: true`) because `UCI_ShowWDL` is ~72% one-hot "
        f"and a one-hot value target teaches over-confidence. Use this table to "
        f"detect a candidate that has DRIFTED or BROKEN, never to pick the value "
        f"target; reading it as a target ranking was attempted and retracted on "
        f"2026-07-27.\n\n"
        + "\n".join(value_lines)
        + "\n\nOutcome column counts only positions whose game continued at "
        "full strength; the v1 audit set has none (handicapped curriculum), "
        "so the column awaits full-strength continuations.\n"
        + _blunder_section(blunder_rows, group_names, cand_labels, blunder_taus)
    )
    out_path.write_text(report, encoding="utf-8")
    print(f"[audit] report written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
