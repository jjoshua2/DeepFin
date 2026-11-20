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
normal approximation on the score scale, x +/- 1.96 SE (sample variance,
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
  Rows without the seat shas are legacy: refused unless ``--allow-legacy-rows``,
  under which the older mtime heuristic applies (no net file may be modified
  after the earlier run's header ``created``) and the output marks the check as
  ``legacy_mtime``. A mixed pair (one row with shas, one without) is refused
  outright, waiver or not.

Provenance: the output records the sha256 of the four input files and of every
net file at pool time, and which net check ran (``row_sha256`` or
``legacy_mtime``).

When several result rows share a fingerprint (and label), as a resumed run can
leave them, the unique complete, non-truncated one is used; anything else is
refused.

Output: JSON (``--out``) and a one-screen text summary on stdout.
``--verdict-rule lb-gt-zero`` adds the pre-registered reading, decided on the
excess (score) scale so that an Elo end that is undefined at an extreme score
cannot hide a clear result: success when the pooled lower 95% bound of the excess
is above 0 (equivalently, Elo lower bound > 0), kill when the upper bound is below
0, otherwise unread.
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


def interval(xs: Sequence[float]) -> dict[str, Any]:
    """Mean, clustered SE and the arena-convention Elo interval of per-cluster values."""
    m, se = _mean_se(xs)
    out: dict[str, Any] = {"n_clusters": len(xs), "excess": m, "se": se, "elo": _elo(m)}
    if se is None:
        out.update(elo_ci95=[None, None], excess_ci95=[None, None],
                   interval_status="unavailable_insufficient_clusters")
    elif se == 0.0:
        out.update(elo_ci95=[None, None], excess_ci95=[None, None],
                   interval_status="unavailable_zero_empirical_variance")
    else:
        out.update(elo_ci95=[_elo(m - Z95 * se), _elo(m + Z95 * se)],
                   excess_ci95=[m - Z95 * se, m + Z95 * se],
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


def pool_time_files(role: str, path: str) -> list[dict[str, Any]]:
    try:
        return [dict(f) for f in seat_files(path)]
    except ProvenanceError as exc:
        raise PoolError(f"{role} net {path}: {exc} (cannot record its sha256)") from exc


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


def check_nets(a_row: dict[str, Any], b_row: dict[str, Any], a_header: dict[str, Any], b_header: dict[str, Any],
               arm: str, base: str, *, allow_legacy: bool) -> tuple[dict[str, Any], str, list[str]]:
    """Prove the pooled nets are the ones played; return (pool-time files, check mode, notes)."""
    nets = {"arm": pool_time_files("arm", arm), "base": pool_time_files("base", base)}
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


def pool(a_games: Games, b_games: Games, pairs: int) -> dict[str, Any]:
    ca, cb = per_opening(a_games, pairs), per_opening(b_games, pairs)
    keys = sorted(ca)
    e_a = [sum(ca[k]) / len(ca[k]) for k in keys]                 # arm as candidate
    e_b = [-sum(cb[k]) / len(cb[k]) for k in keys]                # arm as reference: 0.5 - cand
    d = [(x + y) / 2.0 for x, y in zip(e_a, e_b, strict=True)]
    s = [(x - y) / 2.0 for x, y in zip(e_a, e_b, strict=True)]
    return {
        "n_clusters": len(keys),
        "pairs_per_run": pairs,
        "openings_repeated_within_run": sum(1 for k in keys if len(ca[k]) > 1),
        "ab_correlation": correlation(e_a, e_b),
        "pooled": interval(d),
        "arm_as_candidate": interval(e_a),
        "arm_as_reference": interval(e_b),
        "seat_offset": interval(s),
    }


def verdict(pooled: dict[str, Any], rule: str) -> str:
    if rule != "lb-gt-zero":
        raise PoolError(f"unknown verdict rule {rule!r}")
    lo, hi = pooled.get("excess_ci95") or (None, None)
    if lo is None or hi is None:
        return "unread"
    if lo > 0.0:
        return "success"
    if hi < 0.0:
        return "kill"
    return "unread"


def _fmt(block: dict[str, Any]) -> str:
    elo = block["elo"]
    lo, hi = block["elo_ci95"]
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
        f"  POOLED (arm vs base, Elo, 95%):  {_fmt(r['pooled'])}",
        f"  arm as candidate (run A):        {_fmt(r['arm_as_candidate'])}",
        f"  arm as reference (run B):        {_fmt(r['arm_as_reference'])}",
        f"  seat offset (candidate seat):    {_fmt(r['seat_offset'])}",
        "  interval: normal approximation on the score scale, x +/- 1.96 SE clustered by opening, "
        "ends mapped by Elo = -400 log10(1/(0.5+x) - 1)",
    ]
    lines.extend(f"  NOTE {note}" for note in report["notes"])
    if "verdict" in report:
        lines.append(f"  VERDICT ({report['verdict_rule']}): {report['verdict']}")
    return "\n".join(lines)


def run(a_games_path: Path, b_games_path: Path, a_result: Path, b_result: Path, *, arm: str,
        a_label: str | None = None, b_label: str | None = None, allow_legacy: bool = False,
        rule: str | None = None) -> dict[str, Any]:
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
                                            str(sa["reference"]), allow_legacy=allow_legacy)
    notes += net_notes
    report: dict[str, Any] = {
        "arm": sa.get("candidate"), "base": sa.get("reference"),
        "input_sha256": {"a_games": sha256_file(a_games_path), "b_games": sha256_file(b_games_path),
                         "a_result": sha256_file(a_result), "b_result": sha256_file(b_result)},
        "nets": nets, "net_check": net_check,
        "a": {"label": a_row.get("label"), "games": str(a_games_path), "result": str(a_result),
              "game_log_fingerprint": a_header.get("fingerprint"), "row_elo": a_row.get("elo"),
              "row_elo_ci95": a_row.get("elo_ci95")},
        "b": {"label": b_row.get("label"), "games": str(b_games_path), "result": str(b_result),
              "game_log_fingerprint": b_header.get("fingerprint"), "row_candidate_elo": b_row.get("elo"),
              "row_candidate_elo_ci95": b_row.get("elo_ci95")},
        "compared_header_settings": compared, "compared_row_fields": list(ROW_MATCH_FIELDS),
        "method": {"unit": "opening key (opening_fen + opening_uci); repeated openings form one cluster",
                   "interval": "normal approximation on the score scale, x +/- 1.96 SE, SE = sd/sqrt(K) "
                               "(sample variance), ends mapped by Elo = -400 log10(1/(0.5+x) - 1); "
                               "the convention of arena_standard.summarize_pentanomial"},
        "notes": notes,
        "result": pool(a_g, b_g, pairs),
    }
    if rule:
        report["verdict_rule"] = rule
        report["verdict"] = verdict(report["result"]["pooled"], rule)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="the arm's net path: must be run A's candidate and run B's reference")
    p.add_argument("--a-games", required=True, help="run A's --games-out log (arm = candidate)")
    p.add_argument("--b-games", required=True, help="run B's --games-out log (arm = reference)")
    p.add_argument("--a-result", required=True, help="results JSONL holding run A's row (matched by fingerprint)")
    p.add_argument("--b-result", required=True, help="results JSONL holding run B's row")
    p.add_argument("--a-label", default=None, help="narrow run A's row by label when several share the fingerprint")
    p.add_argument("--b-label", default=None)
    p.add_argument("--allow-legacy-rows", action="store_true",
                   help="for rows written before the arena recorded them: waive buffer_full_flushes / "
                        "eval_max_batch_realised, and check the nets by mtime instead of banked sha256 "
                        "(banked reproductions only; never for a registered read). Both rows must be legacy: "
                        "a mixed pair is refused regardless")
    p.add_argument("--verdict-rule", choices=("lb-gt-zero",), default=None)
    p.add_argument("--out", default=None, help="write the JSON report here")
    a = p.parse_args(argv)
    report = run(Path(a.a_games), Path(a.b_games), Path(a.a_result), Path(a.b_result), arm=a.arm,
                 a_label=a.a_label, b_label=a.b_label, allow_legacy=a.allow_legacy_rows, rule=a.verdict_rule)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(summary_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
