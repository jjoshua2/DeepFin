"""Live auto-retirement step for the cadenced FEN-seed monitor.

Re-scores the CURRENT live seed list (from the yaml's opening_fen_list_path) with
a checkpoint, tracks per-seed consecutive-AWARE reads in a state file, and RETIRES
seeds that read AWARE (net_q <= --resolved-below) TWICE IN A ROW — so a single-read
fluke never drops a seed, and a seed that regresses resets its streak.

Retiring is intended to write a NEW versioned seed file (minus the retired seeds — required, since
selfplay/opening.py::_load_fen_list is lru_cached by PATH, so an in-place edit is
ignored), validates a staged YAML replacement, then atomically publishes it.
Participating feed/retire writers coordinate; arbitrary editors still need operator
coordination. A postcommit error is unconfirmed publication, never a rollback.
Prints one summary line for the monitor log. Fail-soft: never raises.

Probation re-check (default ON; ``--no-probation`` opt-out): retired seeds are kept
in the state file and re-scored every read with the active pool. Any retired seed
whose net_q later reads > -0.2 is RE-FED (restored verbatim, including ``# weight=N``
markers) so retirement is self-correcting rather than irreversible.

Optional ``--retire-require-deep-read``: retirement also requires that at least one
of the consecutive AWARE reads was deep (net_q <= -0.5), filtering hair-width /
shallow-shell flukes at the -0.4 bar.

Intended to be called once per monitor cadence with the same checkpoint the panels
use. Off unless --yaml + --checkpoint are given; --dry-run scores + reports only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid
from typing import Any

# Probation un-retire band (design: -0.2). Still-AWARE-ish (-0.4, -0.2] stays
# retired to avoid thrash; only genuine regression re-feeds.
REFEED_ABOVE = -0.2
# Deep-read bar for --retire-require-deep-read (stable core is ~<= -0.5).
DEEP_BELOW = -0.5

# State-file meta keys (non-int values). Streak counters remain top-level ints so
# pre-probation state files load unchanged.
_META_RETIRED = "__retired__"  # placement key -> original seed line (verbatim)
_META_DEEP = "__deep__"        # placement key -> saw deep AWARE in current streak
_META_KEYS = frozenset({_META_RETIRED, _META_DEEP})


def _placement(line: str) -> str:
    """Position key = piece placement only (ignore move counters / ep).

    Keyed on the LOST terminal (blame-dropped plies re-appended) so a backed
    seed keeps one stable streak identity even if a later gate run backs the
    same lost position up a different number of plies. Key keeps side-to-move/
    castling/ep (clocks ignored) — the gate deliberately emits same-placement
    seeds differing in those bits as DISTINCT, so retirement must not collapse
    their streaks (one resolving would retire both). NOTE: this key format
    change resets existing streak state once (old keys were placement-only) —
    worst case each seed needs one extra AWARE read before retiring.
    """
    # Lazy imports: keep decision-logic importable without the C extension so
    # unit tests need no GPU / Stockfish / build_ext.
    from chess_anti_engine.selfplay.opening import seed_board_from_line
    from scripts.blindspot_resolution import reconstruct_lost_line

    return " ".join(seed_board_from_line(reconstruct_lost_line(line)).fen().split()[:4])


def update_streaks(
    keys: list[str], net_q: list[float], state: dict[str, int], *,
    resolved_below: float, min_consecutive: int,
    deep_seen: dict[str, bool] | None = None,
    require_deep_read: bool = False,
    deep_below: float = DEEP_BELOW,
) -> tuple[set[str], int]:
    """Advance each seed's consecutive-AWARE streak (mutates ``state``) and return
    (retire_keys, resolved_count). A seed AWARE this read (net_q <= resolved_below)
    increments its streak; otherwise the streak RESETS to 0. A seed is retired once
    its streak reaches ``min_consecutive`` — i.e. it read AWARE that many reads in a
    row, so a single-read fluke never drops it and a regression un-arms it.

    When ``require_deep_read`` is set, retirement also needs at least one of the
    consecutive AWARE reads to be deep (net_q <= deep_below). ``deep_seen`` tracks
    that flag across the current streak (mutated; reset with the streak).
    """
    resolved = 0
    for k, qi in zip(keys, net_q):
        if qi <= resolved_below:
            state[k] = int(state.get(k, 0)) + 1
            resolved += 1
            if deep_seen is not None and qi <= deep_below:
                deep_seen[k] = True
        else:
            state[k] = 0
            if deep_seen is not None:
                deep_seen[k] = False
    retire_keys: set[str] = set()
    for k in keys:
        if state.get(k, 0) < min_consecutive:
            continue
        if require_deep_read and not (deep_seen or {}).get(k, False):
            continue
        retire_keys.add(k)
    return retire_keys, resolved


def refeed_retired(
    keys: list[str],
    net_q: list[float],
    *,
    refeed_above: float = REFEED_ABOVE,
) -> set[str]:
    """Return retired seed keys that should re-enter the active pool.

    Re-feed when net_q is strictly above ``refeed_above`` (default -0.2). Reads in
    (-0.4, -0.2] stay retired — the stability study saw 4/9 retirees re-cross
    -0.4 within 3 iters (thrash) but 0/9 cross -0.2 (quiet unless true regression).
    """
    return {k for k, qi in zip(keys, net_q) if qi > refeed_above}


def build_active_list(
    active_lines: list[str],
    active_keys: list[str],
    retire_keys: set[str],
    retired_lines: dict[str, str],
    refeed_keys: set[str],
    *,
    min_pool: int,
) -> tuple[list[str], set[str], set[str]]:
    """Compose the next active seed list from retire + re-feed decisions.

    Returns ``(new_lines, applied_retire, applied_refeed)``. Retirement is
    suppressed when it would drop the pool below ``min_pool`` (or remove nothing).
    Re-feed always applies when requested — it only grows the pool. Re-fed lines
    are the original strings stored at retirement (verbatim, incl. weight markers).
    """
    keep = [ln for ln, k in zip(active_lines, active_keys) if k not in retire_keys]
    applied_retire = set(retire_keys)
    if not retire_keys or len(keep) < min_pool or len(keep) >= len(active_lines):
        keep = list(active_lines)
        applied_retire = set()
    # Preserve retired_lines insertion order for deterministic list output.
    applied_refeed = {k for k in retired_lines if k in refeed_keys}
    refeed_out = [retired_lines[k] for k in retired_lines if k in applied_refeed]
    return keep + refeed_out, applied_retire, applied_refeed


def load_retire_state(
    raw: dict[str, Any],
) -> tuple[dict[str, int], dict[str, str], dict[str, bool]]:
    """Split a state-file dict into (streaks, retired_lines, deep_seen).

    Old state files are a flat ``{placement: streak_int}`` map — loaded as
    streaks with empty retired/deep. Extended files add ``__retired__`` /
    ``__deep__`` meta entries (non-int); those are filtered out of streaks.
    """
    retired_raw = raw.get(_META_RETIRED) or {}
    deep_raw = raw.get(_META_DEEP) or {}
    retired = {str(k): str(v) for k, v in dict(retired_raw).items()}
    deep = {str(k): bool(v) for k, v in dict(deep_raw).items()}
    streaks: dict[str, int] = {}
    for k, v in raw.items():
        if k in _META_KEYS:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        streaks[str(k)] = int(v)
    return streaks, retired, deep


def dump_retire_state(
    streaks: dict[str, int],
    retired: dict[str, str],
    deep_seen: dict[str, bool],
) -> dict[str, Any]:
    """Serialize streaks + optional meta into a state-file dict."""
    out: dict[str, Any] = dict(streaks)
    if retired:
        out[_META_RETIRED] = dict(retired)
    # Persist only True flags (False is the default after a streak reset).
    deep_true = {k: True for k, v in deep_seen.items() if v}
    if deep_true:
        out[_META_DEEP] = deep_true
    return out


def _current_seed_path(yaml_path: str) -> str:
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file
    return str(flatten_run_config_defaults(load_yaml_file(yaml_path))["opening_fen_list_path"])


def _write_seed_file(path: str, lines: list[str], *, note: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {note}\n")
        for ln in lines:
            fh.write(ln + "\n")



def _publish_seed_file(
    out_dir: str, lines: list[str], *, note: str, tag: str = "",
) -> str:
    """Publish complete bytes at a fresh path; never replace a prior generation.

    A same-directory hard link makes publication atomic and no-clobber. Staging
    uses ordinary file permissions (0666 filtered by umask), like the old writer.
    Published generations survive later validation/repoint failures for diagnosis.
    This does not protect against arbitrary external edits or make YAML/state a
    transaction. The output directory must already exist on a hard-link filesystem.
    """
    tag = tag or "auto"
    if any(character in tag for character in ("/", "\\", "\0")):
        raise ValueError("seed tag must be a single filename component")
    directory = os.path.abspath(out_dir)
    for _ in range(10):
        token = uuid.uuid4().hex
        final = os.path.join(directory, f"blindspot_fens_retire_{tag}_{token}.txt")
        staging = os.path.join(directory, f".blindspot_fens_retire_{token}.tmp")
        try:
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"# {note}\n")
                for ln in lines:
                    fh.write(ln + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            try:
                os.link(staging, final)
            except FileExistsError:
                continue
            return final
        finally:
            os.unlink(staging)
    raise FileExistsError("could not allocate a fresh seed generation")


def _yaml_signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _yaml_snapshot(path: Path) -> tuple[bytes, tuple[int, ...]]:
    """Capture a regular non-link file; reject an observed in-place read race."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("YAML target must be a regular non-symlink file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("YAML target must be a regular file")
        data = handle.read()
        after = os.fstat(handle.fileno())
    if _yaml_signature(before) != _yaml_signature(after):
        raise ValueError("YAML changed while being captured")
    return data, _yaml_signature(after)


@contextmanager
def _yaml_repoint_lock(path: Path):
    """Stable private lock for this helper's writers, not arbitrary editors.

    Never unlink this file: replacing it would split cooperating lock domains.
    Existing linked/nonregular/nonempty or nonprivate control files refuse.
    """
    lock = path.with_name(f".{path.name}.seed-repoint.lock")
    try:
        existing = lock.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(existing.st_mode):
            raise ValueError("YAML repoint lock must be regular")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size or info.st_nlink != 1
                or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("YAML repoint lock must be empty, private and owned")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _repoint_yaml(
    yaml_path: str, new_seed_path: str, *, expected_seed_path: str | None = None,
) -> bool:
    """Validate exact staged bytes before atomic publication; never roll back.

    expected_seed_path binds a prepared list to the captured config's pointer,
    not to seed-file contents. Current unrelated settings are preserved. A
    final identity/bytes check detects observed edits, but is NOT atomic CAS
    against a nonparticipating writer's check-to-replace race. The private lock
    coordinates only calls through this helper (including feed and retire).

    True confirms publication; False can also mean a post-replace durability
    error, NOT proof the old file remains. Atomic replacement preserves mode,
    not inode/hard-link identity, owner, ACL or extended metadata. Terminal
    symlinks/nonregular files refuse; parent namespace must remain stable.
    """
    from chess_anti_engine.utils.atomic import atomic_write
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

    try:
        path = Path(yaml_path)
        path = path.parent.resolve(strict=True) / path.name
        with _yaml_repoint_lock(path):
            original, identity = _yaml_snapshot(path)
            original_text = original.decode("utf-8")
            pat = re.compile(r"^([ \t]*opening_fen_list_path:[ \t]*)\S+.*$", re.MULTILINE)
            if len(pat.findall(original_text)) != 1:
                raise ValueError("opening_fen_list_path not uniquely found in yaml")
            updated = pat.sub(lambda m: m.group(1) + json.dumps(new_seed_path),
                              original_text, count=1).encode("utf-8")

            def write_validated(tmp: Path) -> None:
                # Validate the captured original, not a second live-path read.
                tmp.write_bytes(original)
                current = flatten_run_config_defaults(load_yaml_file(tmp))
                if (expected_seed_path is not None
                        and str(current["opening_fen_list_path"]) != expected_seed_path):
                    raise ValueError("seed pointer changed since list preparation")
                tmp.write_bytes(updated)
                candidate = flatten_run_config_defaults(load_yaml_file(tmp))
                if str(candidate["opening_fen_list_path"]) != new_seed_path:
                    raise ValueError("staged seed pointer did not round-trip")
                latest, latest_identity = _yaml_snapshot(path)
                if latest != original or latest_identity != identity:
                    raise ValueError("YAML changed before publication")

            # Existing helper creates private staging at the original mode.
            # Its replace is the commit point; no exception causes rollback.
            atomic_write(path, write_validated, mode=stat.S_IMODE(identity[2]))
        return True
    except Exception as exc:
        print(f"[retire] ABORT: yaml publication unconfirmed ({type(exc).__name__}); "
              "no rollback attempted", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--yaml", default="configs/pbt2_small.yaml")
    ap.add_argument("--state", default="scratchpad/live_read/retire_state.json")
    ap.add_argument("--resolved-below", type=float, default=-0.4,
                    help="net_q <= this = AWARE/solved (the panel's AWARE bar)")
    ap.add_argument("--min-consecutive", type=int, default=2,
                    help="retire only after this many consecutive AWARE reads")
    ap.add_argument("--min-pool", type=int, default=20, help="never retire below this many seeds")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tag", default="", help="filename tag for the new list (e.g. checkpoint num)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-probation", action="store_true",
                    help="disable probation re-check (default: re-score retired seeds and "
                         f"re-feed any with net_q > {REFEED_ABOVE})")
    ap.add_argument("--retire-require-deep-read", action="store_true",
                    help="retire only if at least one of the consecutive AWARE reads was "
                         f"deep (net_q <= {DEEP_BELOW})")
    args = ap.parse_args()
    probation = not args.no_probation

    try:
        from scripts.blindspot_resolution import load_seed_lines, score_seeds

        seed_path = _current_seed_path(args.yaml)
        lines = load_seed_lines(seed_path)
        keys = [_placement(ln) for ln in lines]
        active_key_set = set(keys)

        streaks: dict[str, int] = {}
        retired: dict[str, str] = {}
        deep_seen: dict[str, bool] = {}
        if os.path.exists(args.state):
            with open(args.state, encoding="utf-8") as fh:
                streaks, retired, deep_seen = load_retire_state(json.load(fh))

        # Drop retired entries that are somehow already active (list was hand-edited
        # or reinstated offline) so we never double-score / double-list them.
        retired = {k: ln for k, ln in retired.items() if k not in active_key_set}

        # Full cumulative score list: active first, then retired (probation only).
        retired_items = list(retired.items()) if probation else []
        score_lines = list(lines) + [ln for _, ln in retired_items]
        score_keys = list(keys) + [k for k, _ in retired_items]

        q = score_seeds(args.checkpoint, score_lines, device=args.device,
                        gpu_mem_fraction=args.gpu_mem_fraction)
        q_list = q.tolist()
        active_q = q_list[:len(lines)]
        retired_q = q_list[len(lines):]

        # Per-seed net_q dump (next to the state file, tagged by --tag): the
        # 2026-07-09 stability study had to copy 4 live checkpoints because
        # only streak counters survive a read — this makes every future
        # stability/probation analysis free. Fail-soft like everything here.
        # With probation ON the dump covers the full cumulative list (retired
        # included) so post-retirement regression is visible without a re-score.
        try:
            dump_path = os.path.join(
                os.path.dirname(args.state) or ".",
                f"retire_netq_{args.tag or 'untagged'}.jsonl")
            with open(dump_path, "w", encoding="utf-8") as fh:
                for k, qi, ln in zip(score_keys, q_list, score_lines):
                    fh.write(json.dumps({"key": k, "net_q": round(float(qi), 4),
                                         "line": ln.partition("#")[0].strip(),
                                         "retired": k not in active_key_set}) + chr(10))
        except Exception:
            pass

        retire_keys, resolved_now = update_streaks(
            keys, active_q, streaks,
            resolved_below=args.resolved_below, min_consecutive=args.min_consecutive,
            deep_seen=deep_seen,
            require_deep_read=args.retire_require_deep_read,
            deep_below=DEEP_BELOW)

        refeed_keys: set[str] = set()
        if probation and retired_items:
            refeed_keys = refeed_retired(
                [k for k, _ in retired_items], retired_q, refeed_above=REFEED_ABOVE)

        keep, applied_retire, applied_refeed = build_active_list(
            lines, keys, retire_keys, retired, refeed_keys, min_pool=args.min_pool)

        did_change = False
        new_base = os.path.basename(seed_path)
        if applied_retire or applied_refeed:
            if args.dry_run:
                did_change = True  # report intent
            else:
                parts = []
                if applied_retire:
                    parts.append(f"auto-retired {len(applied_retire)} seeds "
                                 f"(2x-consecutive AWARE @net_q<={args.resolved_below})")
                if applied_refeed:
                    parts.append(f"re-fed {len(applied_refeed)} seeds "
                                 f"(probation net_q>{REFEED_ABOVE})")
                note = f"{'; '.join(parts)} from {new_base}"
                new_path = _publish_seed_file(args.out_dir, keep, note=note, tag=args.tag)
                # only repoint if the new file loads and the yaml stays valid
                from chess_anti_engine.selfplay.opening import _load_fen_list
                n_loaded = len(_load_fen_list(new_path))
                # min_pool guards retirement; re-feed-only may write a list
                # already below min_pool (hand-trimmed) — require a full load.
                load_ok = n_loaded >= args.min_pool or (
                    not applied_retire and n_loaded >= len(keep) and n_loaded > 0)
                if load_ok and _repoint_yaml(
                    args.yaml, new_path, expected_seed_path=seed_path):
                    # Only confirmed publication finalizes retirement bookkeeping;
                    # unconfirmed postcommit errors are not an old-YAML guarantee.
                    line_by_key = dict(zip(keys, lines))
                    for k in applied_retire:
                        # Move active line into the retired store (verbatim).
                        if k in line_by_key:
                            retired[k] = line_by_key[k]
                        streaks.pop(k, None)
                        deep_seen.pop(k, None)
                    for k in applied_refeed:
                        retired.pop(k, None)
                        streaks[k] = 0  # fresh streak after re-feed
                        deep_seen[k] = False
                    new_base = os.path.basename(new_path)
                    did_change = True

        if not args.dry_run:
            # Without probation, never accumulate a retired store (pre-probation
            # behaviour: streak keys are dropped only after a successful write,
            # which the branch above already did).
            if not probation:
                retired = {}
            os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
            with open(args.state, "w", encoding="utf-8") as fh:
                json.dump(dump_retire_state(streaks, retired, deep_seen), fh)

        n_retired = len(applied_retire) if did_change else 0
        n_refed = len(applied_refeed) if did_change else 0
        pool_now = len(keep) if did_change else len(lines)
        dry = "DRY:" if args.dry_run else ""
        print(f"retire: pool={len(lines)} resolved={resolved_now}/{len(lines)} "
              f"(net<={args.resolved_below}) retired={dry}{n_retired} "
              f"refed={dry}{n_refed} (2x) -> pool={pool_now} list={new_base}")
    except Exception as e:  # side tool — must never break the monitor
        print(f"retire: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
