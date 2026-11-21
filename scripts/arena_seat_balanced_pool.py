#!/usr/bin/env python3
"""Seat-balanced pooling of two ``arena_standard.py`` matches.

A seat-balanced comparison plays the SAME openings twice:

* run A: the arm is the CANDIDATE, the base is the reference;
* run B: the arm is the REFERENCE, the base is the candidate.

Any advantage of the candidate seat (call it delta) enters the two runs with
opposite signs and cancels. Per opening key o (opening_fen + opening_uci; every
pair of a key that repeats within a run joins one cluster):

* e_A(o) = mean over A's pairs of o of (candidate pair score / 2 - 0.5)
           = the arm's per-game excess as candidate  (effect + delta);
* e_B(o) = mean over B's pairs of o of (0.5 - candidate pair score / 2)
           = the arm's per-game excess as reference  (effect - delta);
* d(o)   = (e_A(o) + e_B(o)) / 2                     (effect; delta cancels);
* s(o)   = (e_A(o) - e_B(o)) / 2                     (delta, the seat offset).

The estimate is mean_o d(o), its SE sd_o(d) / sqrt(K) over the K opening
clusters (clustered by opening, because each opening appears in both runs). The
interval uses the convention of ``arena_standard.summarize_pentanomial``: a
normal approximation on the score scale, x +/- z SE (z = 1.96 unless ``--z``; sample variance,
n - 1), with both ends mapped through Elo(x) = -400 log10(1 / (0.5 + x) - 1).
It is unavailable with fewer than two clusters or zero variance. The realised
correlation of e_A(o) and e_B(o) across clusters is reported: a positive value
widens the pooled interval relative to two independent runs.

INPUTS are the two per-game logs (``--games-out`` of each run). Each run's
result row (``--out`` of the arena, a JSONL that may hold many rows) is also
read, for the fields only it carries: it is the row whose
``game_log_fingerprint`` equals the log header's fingerprint, narrowed by
``--a-label`` / ``--b-label`` when several rows share it. The script REFUSES
unless:

* each run is complete: the row's ``games == games_requested``, ``truncated``
  false, ``game_log_agrees`` true; every pair id 0..pairs-1 has both colourings
  in the log (the last row per (pair, half), as the arena's resume reads it);
  the log header's ``settings.games`` equals ``games_requested``; and the
  pentanomial recomputed from the log equals the row's;
* each row records ``buffer_full_flushes == 0``, a non-null
  ``eval_max_batch_realised`` and ``eval_leaf_cap_bound`` false
  (``--allow-legacy-rows`` waives only the first two, for rows written before
  the arena recorded them, and the output says so);
* the arm and base swap seats, and the logs' header settings agree after the
  swap: every ``*_candidate`` key of A equals B's ``*_reference`` key and vice
  versa (``candidate`` <-> ``reference``), every other key is equal, and
  ``volatility_candidate`` is null in both (it has no reference-side twin);
* the result-row fields in ``ROW_MATCH_FIELDS`` are equal;
* the index -> opening maps are identical: pair id p has the same opening key
  in A and in B, for every p, and both colourings of a pair share it;
* ``--arm`` (required) resolves to run A's candidate and to run B's reference,
  so runs passed in the wrong order cannot report the base as the arm;
* the nets are the ones that were played. Result rows written by the current
  arena carry ``candidate_sha256`` / ``reference_sha256`` (and the per-file
  ``candidate_files`` / ``reference_files``, plus ``openings_sha256``). Then:
  A's candidate sha256 must be B's reference sha256 and vice versa; every row
  sha256 must equal the sha256 of the same file AT POOL TIME (a JSON spec seat
  counts as the spec plus every file it names, as the arena hashes it); and the
  two runs' ``openings_sha256`` must be non-null and equal. Each row's
  ``game_log_provenance`` must be ``this_run`` or ``recorded``: ``absent`` means
  a pre-provenance log was resumed, so its earlier games were never checked.
  A hybrid seat (``--<seat>-value-from`` / ``-value-recal``; ``candidate_value`` /
  ``reference_value`` in the row) must be the same hybrid in both runs (value-from
  sha256 and [T, b]), and each value-from file must hash, AT POOL TIME, to what
  the row says was played.
  Rows without the seat shas are legacy: refused unless ``--allow-legacy-rows``,
  under which the older mtime heuristic applies (no net file may be modified
  after the earlier run's header ``created``) and the output marks the check as
  ``legacy_mtime``. A mixed pair (one row with shas, one without) is refused
  outright, waiver or not.

A net MOVED after it was played is pooled only with ``--relocate OLD=NEW``: the
file at NEW is hashed and must equal the sha256 recorded for OLD (the hash is
never skipped); the mapping is recorded under ``relocated`` and on the file
(``recorded_path``).

Provenance: the output records the sha256 of the four input files and of every
net file at pool time, and which net check ran (``row_sha256`` or
``legacy_mtime``).

When several result rows share a fingerprint (and label), as a resumed run can
leave them, the unique complete, non-truncated one is used; anything else is
refused.

LOOKS (a group-sequential match). Repeat ``--a-games`` / ``--b-games`` /
``--a-result`` / ``--b-result`` (and ``--a-label`` / ``--b-label`` if used), one
of each per look, in look order. Every look is checked on its own exactly as
above, and the looks together must be one match: the same arm and base (the same
pool-time net sha256s and seat values), header settings equal except ``seed`` and
``games``, the ``ROW_MATCH_FIELDS`` equal except ``games_requested``, and no
seed or game log used twice (a repeated seed replays the same openings and, on
one commit, the same games: pooling it would count them twice). The estimate
pools every pair of every look: an opening key drawn in more than one look is
ONE cluster, so e_A(o) and e_B(o) average over all of that opening's pairs. The
report keeps each look's own block under ``looks``; ``result`` is the cumulative
estimate the look's boundary is read against (``--z``).

Output: JSON (``--out``) and a one-screen text summary on stdout.
``--verdict-rule lb-gt-zero`` adds the pre-registered reading, decided on the
excess (score) scale so that an Elo end that is undefined at an extreme score
cannot hide a clear result: success when the pooled lower bound of the excess
is above 0 (equivalently, Elo lower bound > 0), kill when the upper bound is below
0, otherwise unread. The bounds are at ``--z`` (default 1.96, 95% two-sided; for
example 2.2414 for a 97.5% two-sided reading, or 2.797 / 1.977 for the two looks
of an O'Brien-Fleming design). ``z`` and its two-sided coverage are recorded in
the output, and every block carries both its ``*_ci`` interval at ``z`` and its
``*_ci95`` interval at 1.96.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chess_anti_engine.eval.model_provenance import ProvenanceError, seat_files

Z95 = 1.96
PAIR_SCORE_BINS = (2.0, 1.5, 1.0, 0.5, 0.0)          # arena_standard.PAIR_SCORES order
PENTA_KEYS = ("WW", "WD_DW", "DD_WL", "LD_DL", "LL")

# Result-row fields (not in the game-log fingerprint) that must be EQUAL.
ROW_MATCH_FIELDS = (
    "config_hash", "config_name", "git_sha", "compile", "eval_hoist", "eval_max_batch",
    "eval_max_batch_realised", "max_concurrent_games", "arena_pool_size", "eval_cuda_graphs", "device",
    "games_requested",
)
LEGACY_OPTIONAL = ("buffer_full_flushes", "eval_max_batch_realised")

Games = dict[tuple[int, int], dict[str, Any]]


class PoolError(SystemExit):
    """A refusal: the inputs cannot be pooled as a seat-balanced pair."""


def _elo(x: float) -> float | None:
    """Elo of a per-game excess x over 0.5 (None outside (-0.5, 0.5))."""
    s = 0.5 + x
    if not 0.0 < s < 1.0:
        return None
    return -400.0 * math.log10(1.0 / s - 1.0) + 0.0


def _mean_se(xs: Sequence[float]) -> tuple[float, float | None]:
    n = len(xs)
    if n == 0:
        raise PoolError("no openings to pool")
    m = sum(xs) / n
    if n < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def two_sided_coverage(z: float) -> float:
    """Two-sided normal coverage of +/- z SE: 2 Phi(z) - 1."""
    return math.erf(z / math.sqrt(2.0))


def interval(xs: Sequence[float], z: float = Z95) -> dict[str, Any]:
    """Mean, clustered SE and the arena-convention Elo interval of per-cluster values.

    ``*_ci95`` is always the +/- 1.96 SE interval; ``*_ci`` is the interval at
    ``z`` (the one the verdict reads). They coincide at the default z.
    """
    m, se = _mean_se(xs)
    out: dict[str, Any] = {"n_clusters": len(xs), "excess": m, "se": se, "elo": _elo(m), "z": z}
    if se is None or se == 0.0:
        out.update(elo_ci95=[None, None], excess_ci95=[None, None], elo_ci=[None, None], excess_ci=[None, None],
                   interval_status=("unavailable_insufficient_clusters" if se is None
                                    else "unavailable_zero_empirical_variance"))
    else:
        out.update(elo_ci95=[_elo(m - Z95 * se), _elo(m + Z95 * se)],
                   excess_ci95=[m - Z95 * se, m + Z95 * se],
                   elo_ci=[_elo(m - z * se), _elo(m + z * se)],
                   excess_ci=[m - z * se, m + z * se],
                   interval_status="available_normal_approximation")
    return out


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return None if sxx == 0.0 or syy == 0.0 else sxy / math.sqrt(sxx * syy)


def load_games(path: Path) -> tuple[dict[str, Any], Games]:
    """Header and the LAST row per (pair_id, half), as the arena's own resume reads them."""
    header: dict[str, Any] | None = None
    last: Games = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("kind") == "header":
            header = d
        elif d.get("kind") == "game":
            last[(int(d["pair_id"]), int(d["half"]))] = d
    if header is None:
        raise PoolError(f"{path}: no header line")
    return header, last


def load_row(path: Path, fingerprint: str | None, label: str | None) -> dict[str, Any]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("game_log_fingerprint") == fingerprint and (label is None or r.get("label") == label):
                rows.append(r)
    if len(rows) > 1:
        # A resumed run can leave an earlier, incomplete row beside the final one.
        rows = [r for r in rows if not r.get("truncated") and r.get("games_requested") is not None
                and r.get("games") == r.get("games_requested")]
    if len(rows) != 1:
        want = f"game_log_fingerprint {fingerprint!r}" + ("" if label is None else f" and label {label!r}")
        raise PoolError(f"{path}: expected exactly one (complete, non-truncated, when several) result row "
                        f"with {want}, found {len(rows)}" + ("; pass --a-label / --b-label" if len(rows) > 1 else ""))
    return rows[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


Relocate = dict[str, str]


def parse_relocate(specs: Sequence[str] | None) -> Relocate:
    """``OLD=NEW`` pairs: a net recorded at OLD is now at NEW (moved after it was played)."""
    out: Relocate = {}
    for spec in specs or ():
        old, sep, new = spec.partition("=")
        if not sep or not old or not new:
            raise PoolError(f"--relocate {spec!r}: expected OLD=NEW")
        key = str(Path(old).resolve())
        if key in out:
            raise PoolError(f"--relocate names {old} twice")
        out[key] = new
    return out


def pool_time_files(role: str, path: str, relocate: Relocate | None = None) -> list[dict[str, Any]]:
    """The seat's files hashed NOW. A path named by ``--relocate`` is hashed at its new location (never
    skipped), and each such file records the path the run recorded (``recorded_path``)."""
    new = (relocate or {}).get(str(Path(path).resolve()))
    try:
        files = [dict(f) for f in seat_files(new if new is not None else path)]
    except ProvenanceError as exc:
        where = path if new is None else f"{path} (relocated to {new})"
        raise PoolError(f"{role} net {where}: {exc} (cannot record its sha256)") from exc
    if new is not None:
        files[0]["recorded_path"] = path
    return files


def run_start(header: dict[str, Any], name: str) -> float:
    """The log header's ``created`` as a POSIX time (naive = local, as the arena writes it)."""
    created = str(header.get("created") or "")
    for text in (created, created.replace("Z", "+00:00")):
        try:
            return datetime.datetime.fromisoformat(text).timestamp()
        except ValueError:
            continue
    raise PoolError(f"run {name}: log header has no parseable 'created' ({created!r}); "
                    "cannot check that the nets predate the run")


ROW_SHA_FIELDS = ("candidate_sha256", "reference_sha256")
# "absent" = a pre-provenance log resumed: the games played before the resume were never sha-checked.
ROW_SHA_LOG_PROVENANCE = frozenset({"this_run", "recorded"})


def _file_shas(files: Sequence[dict[str, Any]]) -> list[tuple[Any, Any]]:
    return [(f.get("role"), f.get("sha256")) for f in files]


def _value_signature(block: dict[str, Any] | None) -> tuple[Any, Any] | None:
    """A hybrid seat's (value-from sha256, [T, b]); None for a plain seat."""
    if not block:
        return None
    vf = block.get("value_from")
    recal = block.get("recal")
    return (None if vf is None else vf.get("sha256"), None if recal is None else tuple(float(x) for x in recal))


def check_seat_values(a_row: dict[str, Any], b_row: dict[str, Any],
                      relocate: Relocate | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """A hybrid seat (``--<seat>-value-from`` / ``-value-recal``) must be the SAME hybrid in both runs
    (A's candidate is B's reference), and every value-from file must still hash to what was played.
    Returns (problems, pool-time value-from files)."""
    problems: list[str] = []
    files: list[dict[str, Any]] = []
    for who, a_field, b_field in (("arm", "candidate_value", "reference_value"),
                                  ("base", "reference_value", "candidate_value")):
        va, vb = a_row.get(a_field), b_row.get(b_field)
        if _value_signature(va) != _value_signature(vb):
            problems.append(f"the {who} seat's value differs between runs: A {a_field} {_value_signature(va)} vs "
                            f"B {b_field} {_value_signature(vb)} (a hybrid seat never pools with another seat)")
        for name, block in (("A", va), ("B", vb)):
            vf = (block or {}).get("value_from")
            if not vf:
                continue
            try:
                now = pool_time_files(f"{who} value-from", str(vf.get("path")), relocate)
            except PoolError as exc:
                problems.append(str(exc))
                continue
            files.append({"seat": who, "run": name, **now[0]})
            if now[0]["sha256"] != vf.get("sha256"):
                problems.append(f"run {name} {who} value-from {vf.get('path')} sha256 {vf.get('sha256')} != "
                                f"pool-time sha256 {now[0]['sha256']}: the file changed after it was played")
    return problems, files


def check_nets(a_row: dict[str, Any], b_row: dict[str, Any], a_header: dict[str, Any], b_header: dict[str, Any],
               arm: str, base: str, *, allow_legacy: bool,
               relocate: Relocate | None = None) -> tuple[dict[str, Any], str, list[str]]:
    """Prove the pooled nets are the ones played; return (pool-time files, check mode, notes)."""
    nets = {"arm": pool_time_files("arm", arm, relocate), "base": pool_time_files("base", base, relocate)}
    with_shas = [all(r.get(f) is not None for f in ROW_SHA_FIELDS) for r in (a_row, b_row)]
    if any(with_shas) and not all(with_shas):
        raise PoolError("mixed pair: run " + ("A" if with_shas[0] else "B") + " banks seat sha256s and run "
                        + ("B" if with_shas[0] else "A") + " does not, so the two runs are not from one arena "
                        "version and the shas cannot cross-check; --allow-legacy-rows is for two legacy rows only")
    if all(with_shas):
        problems: list[str] = []
        for name, row in (("A", a_row), ("B", b_row)):
            prov = row.get("game_log_provenance")
            if prov not in ROW_SHA_LOG_PROVENANCE:
                problems.append(f"run {name} game_log_provenance is {prov!r}, not one of "
                                f"{sorted(ROW_SHA_LOG_PROVENANCE)}: its earlier games were never sha-checked")
            if row.get("openings_sha256") is None:
                problems.append(f"run {name} openings_sha256 is null: the openings cannot be proven")
        if a_row["candidate_sha256"] != b_row["reference_sha256"] or a_row["reference_sha256"] != b_row["candidate_sha256"]:
            problems.append(f"seat sha256s are not swapped: A candidate {a_row['candidate_sha256']} / reference "
                            f"{a_row['reference_sha256']}, B candidate {b_row['candidate_sha256']} / reference "
                            f"{b_row['reference_sha256']}")
        for name, row, arm_field, base_field in (("A", a_row, "candidate", "reference"),
                                                 ("B", b_row, "reference", "candidate")):
            for role, field in (("arm", arm_field), ("base", base_field)):
                now = nets[role][0]["sha256"]
                if row[f"{field}_sha256"] != now:
                    problems.append(f"run {name} {field}_sha256 {row[f'{field}_sha256']} != pool-time sha256 of "
                                    f"the {role} ({now}): the file changed after it was played")
                banked = row.get(f"{field}_files")
                if banked is not None and _file_shas(banked) != _file_shas(nets[role]):
                    problems.append(f"run {name} {field}_files {_file_shas(banked)} != pool-time "
                                    f"{_file_shas(nets[role])}")
        if a_row.get("openings_sha256") != b_row.get("openings_sha256"):
            problems.append(f"openings_sha256: A {a_row.get('openings_sha256')} != B {b_row.get('openings_sha256')}")
        value_problems, value_files = check_seat_values(a_row, b_row, relocate)
        problems.extend(value_problems)
        if value_files:
            nets["value_from"] = value_files
        if problems:
            raise PoolError("net provenance refused: " + "; ".join(problems))
        return nets, "row_sha256", []
    if not allow_legacy:
        raise PoolError("result rows carry no candidate_sha256 / reference_sha256 (written before the arena banked "
                        "them), so the played nets cannot be proven; --allow-legacy-rows applies the older mtime "
                        "check, for banked rows only")
    started = min(run_start(a_header, "A"), run_start(b_header, "B"))
    late = []
    for role, files in nets.items():
        for f in files:
            mtime = Path(f["path"]).stat().st_mtime
            f["mtime_unix"] = mtime
            if mtime > started:
                late.append(f"{role} {f['path']}")
    if late:
        raise PoolError("net file(s) modified after the earlier run started, so the banked games may not be "
                        "these nets: " + "; ".join(late))
    return nets, "legacy_mtime", [
        "legacy rows (no banked sha256): net check is the mtime heuristic; a net rewritten before the runs "
        "started, or rewritten with its mtime preserved, cannot be detected"]


def opening_key(game: dict[str, Any]) -> str:
    return json.dumps([game["opening_fen"], list(game.get("opening_uci") or [])])


def check_complete(name: str, row: dict[str, Any], header: dict[str, Any], games: Games,
                   *, allow_legacy: bool) -> list[str]:
    """Refuse an incomplete or unverifiable run; return notes for waived legacy fields."""
    problems: list[str] = []
    notes: list[str] = []
    header_games = (header.get("settings") or {}).get("games")
    if row.get("games_requested") is None or row.get("games") != row.get("games_requested"):
        problems.append(f"games {row.get('games')} != games_requested {row.get('games_requested')}")
    if header_games != row.get("games_requested"):
        problems.append(f"log header games {header_games} != games_requested {row.get('games_requested')}")
    if row.get("truncated"):
        problems.append("truncated is true")
    if row.get("game_log_agrees") is not True:
        problems.append("game_log_agrees is not true")
    if row.get("eval_leaf_cap_bound") is not False:
        problems.append(f"eval_leaf_cap_bound is {row.get('eval_leaf_cap_bound')!r}, not false")
    for field in LEGACY_OPTIONAL:
        if field not in row:
            if allow_legacy:
                notes.append(f"{name}: {field} absent (legacy row, waived by --allow-legacy-rows)")
            else:
                problems.append(f"{field} absent (a row from before the arena recorded it; "
                                "--allow-legacy-rows waives this for banked rows only)")
    if "buffer_full_flushes" in row and row["buffer_full_flushes"] != 0:
        problems.append(f"buffer_full_flushes {row['buffer_full_flushes']} != 0")
    if "eval_max_batch_realised" in row and row["eval_max_batch_realised"] is None:
        problems.append("eval_max_batch_realised is null")
    pairs = row.get("pairs")
    if not isinstance(pairs, int) or pairs < 1:
        problems.append(f"pairs {pairs!r}")
    else:
        missing = [(p, h) for p in range(pairs) for h in (0, 1) if (p, h) not in games]
        extra = sorted({p for p, _ in games if not 0 <= p < pairs})
        if missing:
            problems.append(f"{len(missing)} game(s) missing from the log (first: pair/half {missing[0]})")
        if extra:
            problems.append(f"log has pair ids outside 0..{pairs - 1}: {extra[:5]}")
        if not missing and not extra:
            mixed = [p for p in range(pairs) if opening_key(games[(p, 0)]) != opening_key(games[(p, 1)])]
            if mixed:
                problems.append(f"pair {mixed[0]}: the two colourings have different openings")
            counts = Counter(games[(p, 0)]["score_candidate"] + games[(p, 1)]["score_candidate"]
                             for p in range(pairs))
            penta = {k: counts.get(b, 0) for k, b in zip(PENTA_KEYS, PAIR_SCORE_BINS, strict=True)}
            if row.get("pentanomial") is not None and penta != row["pentanomial"]:
                problems.append(f"pentanomial from the log {penta} != row {row['pentanomial']}")
    if problems:
        raise PoolError(f"run {name} ({row.get('label')}) refused: " + "; ".join(problems))
    return notes


def _swap_key(key: str) -> str:
    if key == "candidate":
        return "reference"
    if key == "reference":
        return "candidate"
    if key.endswith("_candidate"):
        return key[: -len("_candidate")] + "_reference"
    if key.endswith("_reference"):
        return key[: -len("_reference")] + "_candidate"
    return key


def check_pairing(a_header: dict[str, Any], b_header: dict[str, Any],
                  a_row: dict[str, Any], b_row: dict[str, Any]) -> list[str]:
    """Refuse unless B is A with the seats swapped; return the header settings keys compared."""
    sa, sb = a_header.get("settings") or {}, b_header.get("settings") or {}
    problems: list[str] = []
    if sa.get("candidate") != sb.get("reference") or sa.get("reference") != sb.get("candidate"):
        problems.append(f"seats are not swapped: A {sa.get('candidate')} vs {sa.get('reference')}, "
                        f"B {sb.get('candidate')} vs {sb.get('reference')}")
    if sa.get("volatility_candidate") is not None or sb.get("volatility_candidate") is not None:
        problems.append("volatility_candidate is set; it has no reference-side twin, so a seat swap cannot carry it")
    keys = sorted(set(sa) | set(sb))
    for k in keys:
        if k in ("candidate", "reference", "volatility_candidate"):
            continue
        twin = _swap_key(k)
        if sa.get(k) != sb.get(twin):
            problems.append(f"settings.{k} in A != settings.{twin} in B" if twin != k
                            else f"settings.{k}: A {sa.get(k)!r} != B {sb.get(k)!r}")
    problems.extend(f"{f}: A {a_row.get(f)!r} != B {b_row.get(f)!r}"
                    for f in ROW_MATCH_FIELDS if a_row.get(f) != b_row.get(f))
    if problems:
        raise PoolError("runs A and B are not a seat-balanced pair: " + "; ".join(problems))
    return keys


def check_opening_maps(a_games: Games, b_games: Games, pairs: int) -> None:
    diff = [p for p in range(pairs) if opening_key(a_games[(p, 0)]) != opening_key(b_games[(p, 0)])]
    if diff:
        raise PoolError(f"runs A and B played different openings: the index -> opening maps differ at "
                        f"{len(diff)} of {pairs} pair id(s) (first: {diff[0]}; key = opening_fen + opening_uci)")


def per_opening(games: Games, pairs: int) -> dict[str, list[float]]:
    """Candidate per-game excess of every pair, grouped by opening key."""
    out: dict[str, list[float]] = defaultdict(list)
    for p in range(pairs):
        s = games[(p, 0)]["score_candidate"] + games[(p, 1)]["score_candidate"]
        out[opening_key(games[(p, 0)])].append(s / 2.0 - 0.5)
    return out


def pool(a_games: Games, b_games: Games, pairs: int, z: float = Z95) -> dict[str, Any]:
    return pool_looks([(a_games, b_games, pairs)], z)


def pool_looks(looks: Sequence[tuple[Games, Games, int]], z: float = Z95) -> dict[str, Any]:
    """Pool one or more looks; an opening key that occurs in several looks is one cluster."""
    ca: dict[str, list[float]] = defaultdict(list)
    cb: dict[str, list[float]] = defaultdict(list)
    for a_games, b_games, pairs in looks:
        for k, v in per_opening(a_games, pairs).items():
            ca[k].extend(v)
        for k, v in per_opening(b_games, pairs).items():
            cb[k].extend(v)
    pairs = sum(n for _, _, n in looks)
    keys = sorted(ca)
    e_a = [sum(ca[k]) / len(ca[k]) for k in keys]                 # arm as candidate
    e_b = [-sum(cb[k]) / len(cb[k]) for k in keys]                # arm as reference: 0.5 - cand
    d = [(x + y) / 2.0 for x, y in zip(e_a, e_b, strict=True)]
    s = [(x - y) / 2.0 for x, y in zip(e_a, e_b, strict=True)]
    return {
        "n_clusters": len(keys),
        "pairs_per_run": pairs,
        "looks": len(looks),
        "openings_repeated_within_run": sum(1 for k in keys if len(ca[k]) > 1),
        "ab_correlation": correlation(e_a, e_b),
        "pooled": interval(d, z),
        "arm_as_candidate": interval(e_a, z),
        "arm_as_reference": interval(e_b, z),
        "seat_offset": interval(s, z),
    }


def verdict(pooled: dict[str, Any], rule: str) -> str:
    if rule != "lb-gt-zero":
        raise PoolError(f"unknown verdict rule {rule!r}")
    lo, hi = pooled.get("excess_ci") or (None, None)      # the interval at the report's z
    if lo is None or hi is None:
        return "unread"
    if lo > 0.0:
        return "success"
    if hi < 0.0:
        return "kill"
    return "unread"


def _fmt(block: dict[str, Any]) -> str:
    elo = block["elo"]
    lo, hi = block["elo_ci"]
    se = block["se"]
    e = "n/a" if elo is None else f"{elo:+.1f}"
    ci = "[n/a]" if lo is None or hi is None else f"[{lo:+.1f}, {hi:+.1f}]"
    return f"{e} {ci}  (excess {block['excess']:+.4f}, SE {'n/a' if se is None else f'{se:.4f}'})"


def summary_text(report: dict[str, Any]) -> str:
    r = report["result"]
    corr = r["ab_correlation"]
    lines = [
        f"seat-balanced pool: A = {report['a']['label']} (arm candidate), B = {report['b']['label']} (arm reference)",
        f"  arm  {report['arm']}  (--arm; run A candidate, run B reference)",
        f"  base {report['base']}",
        f"  nets checked by: {report['net_check']}",
        f"  {r['n_clusters']} opening clusters, {r['pairs_per_run']} pairs per run, "
        f"{r['openings_repeated_within_run']} opening(s) repeated within a run; "
        f"A/B correlation {'n/a' if corr is None else f'{corr:+.3f}'}",
        f"  POOLED (arm vs base, Elo, z {report['z']:g} = {100 * report['coverage']:.2f}% two-sided):",
        f"                                   {_fmt(r['pooled'])}",
        f"  arm as candidate (run A):        {_fmt(r['arm_as_candidate'])}",
        f"  arm as reference (run B):        {_fmt(r['arm_as_reference'])}",
        f"  seat offset (candidate seat):    {_fmt(r['seat_offset'])}",
        f"  interval: normal approximation on the score scale, x +/- {report['z']:g} SE clustered by opening, "
        "ends mapped by Elo = -400 log10(1/(0.5+x) - 1)",
    ]
    if len(report.get("looks", [])) > 1:
        lines.insert(1, f"  {len(report['looks'])} looks pooled (cumulative); openings drawn in several looks "
                        "are one cluster")
    lines.extend(f"  NOTE {note}" for note in report["notes"])
    if "verdict" in report:
        lines.append(f"  VERDICT ({report['verdict_rule']}): {report['verdict']}")
    return "\n".join(lines)


LOOK_FREE_SETTINGS = ("seed", "games")
LOOK_FREE_ROW_FIELDS = ("games_requested",)


def check_look(a_games_path: Path, b_games_path: Path, a_result: Path, b_result: Path, *, arm: str,
               a_label: str | None = None, b_label: str | None = None,
               allow_legacy: bool = False, relocate: Relocate | None = None) -> dict[str, Any]:
    """Every per-run and A/B check for ONE look; returns what pooling and the report need."""
    a_header, a_g = load_games(a_games_path)
    b_header, b_g = load_games(b_games_path)
    a_row = load_row(a_result, a_header.get("fingerprint"), a_label)
    b_row = load_row(b_result, b_header.get("fingerprint"), b_label)
    notes = check_complete("A", a_row, a_header, a_g, allow_legacy=allow_legacy)
    notes += check_complete("B", b_row, b_header, b_g, allow_legacy=allow_legacy)
    compared = check_pairing(a_header, b_header, a_row, b_row)
    pairs = int(a_row["pairs"])
    check_opening_maps(a_g, b_g, pairs)
    sa, sb = a_header["settings"], b_header["settings"]
    want = Path(arm).resolve()
    if Path(str(sa.get("candidate"))).resolve() != want or Path(str(sb.get("reference"))).resolve() != want:
        raise PoolError(f"--arm {arm} is not run A's candidate ({sa.get('candidate')}) and run B's reference "
                        f"({sb.get('reference')}); are A and B passed in the right order?")
    nets, net_check, net_notes = check_nets(a_row, b_row, a_header, b_header, str(sa["candidate"]),
                                            str(sa["reference"]), allow_legacy=allow_legacy, relocate=relocate)
    notes += net_notes
    return {
        "a_header": a_header, "b_header": b_header, "a_row": a_row, "b_row": b_row,
        "a_games": a_g, "b_games": b_g, "pairs": pairs, "compared": compared,
        "nets": nets, "net_check": net_check, "notes": notes,
        "input_sha256": {"a_games": sha256_file(a_games_path), "b_games": sha256_file(b_games_path),
                         "a_result": sha256_file(a_result), "b_result": sha256_file(b_result)},
        "a": {"label": a_row.get("label"), "games": str(a_games_path), "result": str(a_result),
              "game_log_fingerprint": a_header.get("fingerprint"), "row_elo": a_row.get("elo"),
              "row_elo_ci95": a_row.get("elo_ci95")},
        "b": {"label": b_row.get("label"), "games": str(b_games_path), "result": str(b_result),
              "game_log_fingerprint": b_header.get("fingerprint"), "row_candidate_elo": b_row.get("elo"),
              "row_candidate_elo_ci95": b_row.get("elo_ci95")},
    }


def check_looks_are_one_match(looks: Sequence[dict[str, Any]]) -> None:
    """Refuse looks that are not the same match played on further seeds."""
    first = looks[0]
    problems: list[str] = []
    seen_seeds: dict[Any, int] = {}
    seen_logs: dict[str, int] = {}
    for i, look in enumerate(looks):
        for side in ("a", "b"):
            path = str(Path(look[side]["games"]).resolve())
            if path in seen_logs:
                problems.append(f"look {i + 1}: run {side.upper()}'s game log is also look {seen_logs[path] + 1}'s")
            seen_logs.setdefault(path, i)
        seed = (look["a_header"].get("settings") or {}).get("seed")
        if seed in seen_seeds:
            problems.append(f"look {i + 1} repeats look {seen_seeds[seed] + 1}'s seed {seed!r}: the same openings "
                            "and games would be counted twice")
        seen_seeds.setdefault(seed, i)
        if i == 0:
            continue
        for side in ("a", "b"):
            s0 = first[f"{side}_header"].get("settings") or {}
            si = look[f"{side}_header"].get("settings") or {}
            problems.extend(f"look {i + 1} run {side.upper()}: settings.{k} {si.get(k)!r} != look 1's {s0.get(k)!r}"
                            for k in sorted(set(s0) | set(si))
                            if k not in LOOK_FREE_SETTINGS and s0.get(k) != si.get(k))
            r0, ri = first[f"{side}_row"], look[f"{side}_row"]
            problems.extend(f"look {i + 1} run {side.upper()}: {f} {ri.get(f)!r} != look 1's {r0.get(f)!r}"
                            for f in ROW_MATCH_FIELDS
                            if f not in LOOK_FREE_ROW_FIELDS and r0.get(f) != ri.get(f))
        if look["nets"] != first["nets"]:
            problems.append(f"look {i + 1}: the nets (pool-time sha256s / seat values) differ from look 1's")
        if look["net_check"] != first["net_check"]:
            problems.append(f"look {i + 1}: net check {look['net_check']!r} != look 1's {first['net_check']!r}")
    if problems:
        raise PoolError("the looks are not one match: " + "; ".join(problems))


def run_looks(paths: Sequence[tuple[Path, Path, Path, Path]], *, arm: str,
              labels: Sequence[tuple[str | None, str | None]] | None = None, allow_legacy: bool = False,
              rule: str | None = None, z: float = Z95, relocate: Relocate | None = None) -> dict[str, Any]:
    if not (math.isfinite(z) and z > 0.0):
        raise PoolError(f"--z must be a finite positive number, got {z!r}")
    if not paths:
        raise PoolError("no looks given")
    labels = list(labels) if labels is not None else [(None, None)] * len(paths)
    if len(labels) != len(paths):
        raise PoolError(f"{len(labels)} label pair(s) for {len(paths)} look(s)")
    looks = [check_look(*p, arm=arm, a_label=la, b_label=lb, allow_legacy=allow_legacy, relocate=relocate)
             for p, (la, lb) in zip(paths, labels, strict=True)]
    check_looks_are_one_match(looks)
    first = looks[0]
    sa = first["a_header"]["settings"]
    notes = [n for look in looks for n in look["notes"]]
    report: dict[str, Any] = {
        "arm": sa.get("candidate"), "base": sa.get("reference"),
        "input_sha256": first["input_sha256"],
        "nets": first["nets"], "net_check": first["net_check"],
        "relocated": dict(relocate or {}),
        "z": z, "coverage": two_sided_coverage(z),
        "a": first["a"], "b": first["b"],
        "compared_header_settings": first["compared"], "compared_row_fields": list(ROW_MATCH_FIELDS),
        "method": {"unit": "opening key (opening_fen + opening_uci); repeated openings form one cluster",
                   "interval": f"normal approximation on the score scale, x +/- z SE (z = {z:g}; *_ci95 is "
                               "always at 1.96), SE = sd/sqrt(K) (sample variance), ends mapped by "
                               "Elo = -400 log10(1/(0.5+x) - 1); the convention of "
                               "arena_standard.summarize_pentanomial"},
        "notes": notes,
        "result": pool_looks([(lk["a_games"], lk["b_games"], lk["pairs"]) for lk in looks], z),
    }
    if len(looks) > 1:
        report["method"]["looks"] = ("cumulative over every look; an opening drawn in several looks is one "
                                     "cluster; looks differ only in seed and games")
        report["looks"] = [
            {"look": i + 1, "seed": (lk["a_header"].get("settings") or {}).get("seed"), "a": lk["a"], "b": lk["b"],
             "input_sha256": lk["input_sha256"],
             "result": pool(lk["a_games"], lk["b_games"], lk["pairs"], z)}
            for i, lk in enumerate(looks)
        ]
    if rule:
        report["verdict_rule"] = rule
        report["verdict"] = verdict(report["result"]["pooled"], rule)
    return report


def run(a_games_path: Path, b_games_path: Path, a_result: Path, b_result: Path, *, arm: str,
        a_label: str | None = None, b_label: str | None = None, allow_legacy: bool = False,
        rule: str | None = None, z: float = Z95, relocate: Relocate | None = None) -> dict[str, Any]:
    """One look (the classic two-run pool)."""
    return run_looks([(a_games_path, b_games_path, a_result, b_result)], arm=arm,
                     labels=[(a_label, b_label)], allow_legacy=allow_legacy, rule=rule, z=z, relocate=relocate)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="the arm's net path: must be run A's candidate and run B's reference")
    p.add_argument("--a-games", required=True, action="append",
                   help="run A's --games-out log (arm = candidate); repeat once per look, in look order")
    p.add_argument("--b-games", required=True, action="append", help="run B's --games-out log (arm = reference)")
    p.add_argument("--a-result", required=True, action="append",
                   help="results JSONL holding run A's row (matched by fingerprint)")
    p.add_argument("--b-result", required=True, action="append", help="results JSONL holding run B's row")
    p.add_argument("--a-label", default=None, action="append",
                   help="narrow run A's row by label when several share the fingerprint (once per look, or never)")
    p.add_argument("--b-label", default=None, action="append")
    p.add_argument("--allow-legacy-rows", action="store_true",
                   help="for rows written before the arena recorded them: waive buffer_full_flushes / "
                        "eval_max_batch_realised, and check the nets by mtime instead of banked sha256 "
                        "(banked reproductions only; never for a registered read). Both rows must be legacy: "
                        "a mixed pair is refused regardless")
    p.add_argument("--verdict-rule", choices=("lb-gt-zero",), default=None)
    p.add_argument("--z", type=float, default=Z95,
                   help="interval half-width in SE, the one the verdict reads (default 1.96 = 95%% two-sided; "
                        "2.2414 = 97.5%% two-sided; O'Brien-Fleming looks 2.797 / 1.977). Recorded in the output")
    p.add_argument("--relocate", action="append", default=None, metavar="OLD=NEW",
                   help="a net recorded at OLD was moved to NEW after it was played: hash it at NEW against the "
                        "recorded sha256 (never skipped) and record the mapping (repeatable)")
    p.add_argument("--out", default=None, help="write the JSON report here")
    a = p.parse_args(argv)
    n = len(a.a_games)
    if not (len(a.b_games) == len(a.a_result) == len(a.b_result) == n):
        p.error("--a-games, --b-games, --a-result and --b-result must be given the same number of times (once per look)")
    for name, given in (("--a-label", a.a_label), ("--b-label", a.b_label)):
        if given is not None and len(given) != n:
            p.error(f"{name} must be given once per look ({n}) or not at all")
    labels = list(zip(a.a_label or [None] * n, a.b_label or [None] * n, strict=True))
    paths = [(Path(ag), Path(bg), Path(ar), Path(br))
             for ag, bg, ar, br in zip(a.a_games, a.b_games, a.a_result, a.b_result, strict=True)]
    report = run_looks(paths, arm=a.arm, labels=labels, allow_legacy=a.allow_legacy_rows, rule=a.verdict_rule, z=a.z,
                       relocate=parse_relocate(a.relocate))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(summary_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
