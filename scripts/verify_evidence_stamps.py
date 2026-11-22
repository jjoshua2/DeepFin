#!/usr/bin/env python3
"""Check every bank-file citation in an evidence record against the bank on disk.

Read-only and CPU-only. NOT wired into CI: CI has no banks. Run it before any PR
that touches `docs/experiments/2026-11-21-rl-loop-evidence-log.md` (or any record
that cites bank files) and paste the SUMMARY line into the PR body.

What counts as a citation
-------------------------
* A backticked token that starts with a bank prefix (`RC/...`, `R/...`, `F/...`,
  `WK/...`, `~/...`) or with `.../` (= the directory of the previous citation).
  These are always checked, stamped or not. A token that IS a prefix
  (`~/chess-artifacts`, `RC/`, `.../`) is a mention, not a citation.
* A stamp that a hard wrap moved to the start of the next line attaches to the
  token that ends the previous line.
* A backticked bare relative name (`results.jsonl`, `logs/p1.log`) only when a
  stamp follows it. It resolves against the directory of the most recent
  prefixed citation: earlier on the same line, else on an earlier line.
  Unstamped bare names are prose (repo paths, flags), not citations. A commit is
  cited as code `a0bc34416` (a backticked token), never as a bracketed stamp: a
  stamped repo path is a bank citation and reads MISSING if the bank lacks it.

Stamps
------
A stamp directly follows its token: `[`<hex>`]`, a lower-case sha256 prefix of at
least 8 hex digits (the house style), or `[mtime MM-DD HH:MM]`. A sha stamp on a
brace path may list one prefix per option: `[`aaaa1111` / `bbbb2222`]` or
`[`aaaa1111`] / [`bbbb2222`]`.

* One sha stamp must hold for every file the path matches (a byte-equality claim
  when the path is a glob or a brace). A longer list is compared per brace option
  when it lists one prefix per option (every file the option matches must carry
  it), else pairwise when it lists one prefix per matched file; any other count is
  DRIFT.
* mtimes are formatted in the record's declared time zone (the
  "Stamp time zone: `<IANA name>`" line in the prefix source, or `--tz`), never the
  machine's. A directory or glob matches if the stamp equals either the directory's
  own mtime or the newest file under it. Minute granularity cannot detect a rewrite
  within the same minute; prefer sha stamps for anything that is not still growing.

Statuses
--------
One line per citation: OK, DRIFT (the stamp does not match the file), MISSING (the
path does not resolve) and MALFORMED: a stamp-like bracket the strict form rejects
(too short, upper-case, no backticks, letters-only hex without backticks, a bad
mtime, or separated from its token by a colon), or a near-miss of a known prefix
(`rc/`, `RC /`). A MALFORMED citation would otherwise be silently skipped. The
check fires only where a stamp can be: directly after a backticked token, or wrapped
onto the next line after one. Bare lower-case hex in brackets (no backticks) fires
only after a path-like token (one containing `/`, a glob or brace, or ending in an
extension), because prose like `v2` [cafe1234 rows] has the same shape. Brackets in prose (`fixed in [0c795fb7a]`, a markdown link
`[a0bc34416](...)`, `the [mtime column]`) are not stamps and stay silent.

Prefix map
----------
Read from the `| Prefix | Expands to |` table in the evidence log's header
(`--prefix-source`), never hard-coded here. A value starting with
`~/chess-artifacts` is rebased onto the artifacts root (`--artifacts-root`, else
env `CHESS_ARTIFACTS_ROOT`, else `~/chess-artifacts`); any other `~/` value is the
home directory. The table's own rows are not scanned.

Exit status
-----------
0 only if there is no DRIFT, no MALFORMED, and every MISSING token is in the
allow-list (`<record stem>.stamp-allowlist.txt` beside the record, or
`--allow-list`). Each allow-list line is `<token> | <reason>`. 1 otherwise.
2 on usage errors (including an allow-list line without a reason).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "docs/experiments/2026-11-21-rl-loop-evidence-log.md"
ARTIFACTS_TILDE = "~/chess-artifacts"
MIN_SHA = 8

_HEX = rf"[0-9a-f]{{{MIN_SHA},64}}"
_TOKEN_RE = re.compile(
    r"`(?P<token>[^`\n]+)`"
    r"(?P<stamp>\s*(?:(?P<shas>\[`" + _HEX + r"`\]?(?:\s*/\s*\[?`" + _HEX + r"`\]?)*)"
    r"|\[mtime (?P<mtime>\d\d-\d\d \d\d:\d\d)\]))?"
)
# Stamp-like brackets directly after a backticked token that the strict form did not
# consume. STRONG shapes fire after ANY backticked token (prose does not produce
# them): backticked hex of any length or case, bare hex of 6+ carrying an upper-case
# A-F, `[mtime <digit>`, or any stamp shape after a colon. The WEAK shape, bare
# lower-case hex of 6+ (letters-only included), also matches prose such as
# `v2` [cafe1234 rows], so it fires only after a path-like token. Never a markdown
# link `[...](`.
_STAMP_SHAPE = r"\[\s*(?:`[0-9A-Fa-f]+`|[0-9A-Fa-f]{6,64}(?=[\s\]/])|mtime\s*\d)"
_LOOSE_STRONG_RE = re.compile(
    r"\s*:\s*" + _STAMP_SHAPE
    + r"|\s*\[\s*(?:`[0-9A-Fa-f]+`|(?=[0-9a-f]*[A-F])[0-9A-Fa-f]{6,64}(?=[\s\]/])|mtime\s*\d)"
)
_LOOSE_WEAK_RE = re.compile(r"\s*\[\s*[0-9a-f]{6,64}(?=[\s\]/])")
_PATH_LIKE_RE = re.compile(r"[/*{}]|\.[A-Za-z0-9]+$")
_PREFIX_TABLE_RE = re.compile(r"^\|\s*Prefix\s*\|\s*Expands to\s*\|")
_TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")
_TZ_RE = re.compile(r"Stamp time zone:\s*`([^`]+)`")
_NEAR_PREFIX_RE = re.compile(r"^([A-Za-z~.]+)\s*/")


class UsageError(Exception):
    pass


@dataclass
class Citation:
    file: str
    line: int
    token: str
    status: str
    detail: str


def _sha256(path: str, cache: dict[str, str]) -> str:
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        cache[path] = h.hexdigest()
    return cache[path]


def _newest_file_mtime(path: str) -> float | None:
    newest: float | None = None
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                ts = os.path.getmtime(os.path.join(root, name))
            except OSError:
                continue
            if newest is None or ts > newest:
                newest = ts
    return newest


def read_prefix_map(source: Path, artifacts_root: Path) -> dict[str, str]:
    """Parse the `| Prefix | Expands to |` table; returns prefix -> absolute dir."""
    out: dict[str, str] = {}
    in_table = False
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not in_table:
            in_table = bool(_PREFIX_TABLE_RE.match(line))
            continue
        if line.startswith(("|---", "| ---")):
            continue
        m = _TABLE_ROW_RE.match(line)
        if not m:
            break
        prefix, value = m.group(1), m.group(2)
        if not prefix.endswith("/"):
            raise UsageError(f"prefix {prefix!r} in {source} must end with '/'")
        out[prefix] = _expand_root(value, artifacts_root)
    if not out:
        raise UsageError(f"no '| Prefix | Expands to |' table found in {source}")
    return out


def read_declared_tz(source: Path) -> str | None:
    m = _TZ_RE.search(source.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _expand_root(value: str, artifacts_root: Path) -> str:
    if value == ARTIFACTS_TILDE or value.startswith(ARTIFACTS_TILDE + "/"):
        return str(artifacts_root) + value[len(ARTIFACTS_TILDE):]
    if value.startswith("~/") or value == "~":
        return os.path.expanduser(value)
    raise UsageError(f"prefix value {value!r} must start with '~/'")


def read_allow_list(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    allowed: dict[str, str] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token, sep, reason = line.partition("|")
        token, reason = token.strip().strip("`"), reason.strip()
        if not sep or not token or not reason:
            raise UsageError(f"{path}:{n}: allow-list line needs '<token> | <reason>'")
        allowed[token] = reason
    return allowed


def _expand_braces(path: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", path)
    if not m:
        return [path]
    out: list[str] = []
    for opt in m.group(1).split(","):
        out.extend(_expand_braces(path[: m.start()] + opt + path[m.end():]))
    return out


def _resolve(path: str) -> list[list[str]]:
    """One match list per brace option (globs expanded, sorted)."""
    groups = []
    for p in _expand_braces(path):
        if glob.has_magic(p):
            groups.append(sorted(glob.glob(p)))
        else:
            groups.append([p] if os.path.exists(p) else [])
    return groups


_WRAPPED_STAMP_RE = re.compile(r"^\s*\[(?:`|mtime\s*\d|[0-9A-Fa-f]{6,64}(?=[\s\]/]))")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Lines, with a stamp that a hard wrap put at the start of a line joined back
    onto the previous line (reported at the previous line's number)."""
    out: list[tuple[int, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if out and _WRAPPED_STAMP_RE.match(line) and out[-1][1].rstrip().endswith("`"):
            out[-1] = (out[-1][0], out[-1][1].rstrip() + " " + line.lstrip())
        else:
            out.append((n, line))
    return out


class Verifier:
    def __init__(self, prefixes: dict[str, str], tz: str):
        # Longest prefix first so `~/chess-artifacts/` beats `~/`.
        self.prefixes = sorted(prefixes.items(), key=lambda kv: -len(kv[0]))
        self.mentions = set(prefixes) | {p.rstrip("/") for p in prefixes} | {".../", "~"}
        self.prefix_heads = {p.rstrip("/").lower() for p in prefixes if not p.startswith("~")}
        try:
            self.zone = ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise UsageError(f"unknown time zone {tz!r}") from exc
        self.sha_cache: dict[str, str] = {}

    def _fmt_mtime(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, self.zone).strftime("%m-%d %H:%M")

    def _prefixed(self, token: str) -> str | None:
        for prefix, root in self.prefixes:
            if token.startswith(prefix):
                return root.rstrip("/") + "/" + token[len(prefix):]
        return None

    def _near_miss(self, token: str) -> bool:
        m = _NEAR_PREFIX_RE.match(token)
        return bool(m) and m.group(1).lower() in self.prefix_heads

    def check_file(self, record: Path) -> list[Citation]:
        cites: list[Citation] = []
        last_dir: str | None = None
        in_prefix_table = False
        for lineno, line in _logical_lines(record.read_text(encoding="utf-8")):
            # The prefix table defines the map; its rows are not citations.
            if _PREFIX_TABLE_RE.match(line.strip()):
                in_prefix_table = True
            if in_prefix_table:
                if line.strip().startswith("|"):
                    continue
                in_prefix_table = False

            def cite(token: str, status: str, detail: str, _n: int = lineno) -> None:
                cites.append(Citation(str(record), _n, token, status, detail))

            consumed: list[tuple[int, int]] = []
            token_spans: list[tuple[int, int, str]] = []
            for m in _TOKEN_RE.finditer(line):
                token = m.group("token").strip()
                token_spans.append((m.start(), m.end("token") + 1, token))
                if m.group("stamp"):
                    consumed.append(m.span("stamp"))
                shas = re.findall(_HEX, m.group("shas") or "")
                mtime = m.group("mtime")
                stamped = bool(shas or mtime)
                if token in self.mentions:
                    continue
                path = self._prefixed(token)
                prefixed = path is not None
                if path is None and self._near_miss(token):
                    cite(token, "MALFORMED", "near-miss of a bank prefix (case or a space before '/')")
                    continue
                if path is None and token.startswith(".../"):
                    if last_dir is None:
                        cite(token, "MISSING", "no earlier citation for '.../'")
                        continue
                    path, prefixed = last_dir + "/" + token[4:], True
                if path is None:
                    if not stamped or token.startswith("/") or " " in token:
                        continue
                    if last_dir is None:
                        cite(token, "MISSING", "bare name with no earlier citation")
                        continue
                    path = last_dir + "/" + token
                c = self._check(record, lineno, token, path, shas, mtime)
                cites.append(c)
                if c.status != "MISSING" and prefixed:
                    groups = _resolve(path)
                    first = groups[0][0] if groups and groups[0] else None
                    if first is not None and len(groups) == 1 and len(groups[0]) == 1:
                        last_dir = first if os.path.isdir(first) else os.path.dirname(first)
                    elif first is not None:
                        last_dir = os.path.dirname(first)
            for _start, end, token in token_spans:
                if any(a <= end < b for a, b in consumed):
                    continue
                lm = _LOOSE_STRONG_RE.match(line, end)
                if lm is None and _PATH_LIKE_RE.search(token):
                    lm = _LOOSE_WEAK_RE.match(line, end)
                if lm is None:
                    continue
                close = line.find("]", lm.end())
                if close >= 0 and line[close + 1: close + 2] == "(":
                    continue  # a markdown link, not a stamp
                text = line[line.find("[", end): close + 1 if close >= 0 else lm.end() + 40]
                cite(token, "MALFORMED", f"stamp-like {text!r} not in the strict form "
                     f"([`<{MIN_SHA}+ lower-case hex>`] or [mtime MM-DD HH:MM] directly after the token)")
        return cites

    def _check(self, record: Path, lineno: int, token: str, path: str,
               shas: list[str], mtime: str | None) -> Citation:
        def cite(status: str, detail: str) -> Citation:
            return Citation(str(record), lineno, token, status, detail)

        groups = _resolve(path)
        if any(not g for g in groups):
            return cite("MISSING", path)
        matches = [f for g in groups for f in g]
        if shas:
            if len(shas) == 1:
                pairs = [(shas[0], f) for f in matches]  # one stamp, every match byte-equal
            elif len(shas) == len(groups):
                pairs = [(stamp, f) for stamp, g in zip(shas, groups) for f in g]
            elif len(shas) == len(matches):
                pairs = list(zip(shas, matches))
            else:
                return cite("DRIFT", f"{len(shas)} sha stamp(s) for {len(groups)} option(s) / {len(matches)} match(es)")
            for stamp, f in pairs:
                if not os.path.isfile(f):
                    return cite("DRIFT", f"sha stamp on a directory: {f}")
                actual = _sha256(f, self.sha_cache)
                if not actual.startswith(stamp):
                    return cite("DRIFT", f"stamp {stamp} != {actual[:len(stamp)]} ({f})")
            return cite("OK", "sha " + " / ".join(shas))
        if mtime:
            if len(matches) == 1 and os.path.isfile(matches[0]):
                actual = self._fmt_mtime(os.path.getmtime(matches[0]))
                ok = actual == mtime
            else:
                candidates = set()
                for f in matches:
                    candidates.add(self._fmt_mtime(os.path.getmtime(f)))
                    if os.path.isdir(f):
                        newest = _newest_file_mtime(f)
                        if newest is not None:
                            candidates.add(self._fmt_mtime(newest))
                if len(matches) > 1:
                    candidates.add(self._fmt_mtime(max(os.path.getmtime(f) for f in matches)))
                ok = mtime in candidates
                actual = "/".join(sorted(candidates))
            return cite("OK" if ok else "DRIFT", f"mtime {mtime}" + ("" if ok else f" != {actual}"))
        return cite("OK", "exists, unstamped")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("records", nargs="*", type=Path, help=f"records to check (default {DEFAULT_LOG.relative_to(REPO_ROOT)})")
    ap.add_argument("--prefix-source", type=Path, default=DEFAULT_LOG,
                    help="file holding the '| Prefix | Expands to |' table and the stamp time zone (default: the evidence log)")
    ap.add_argument("--artifacts-root", type=Path, default=None,
                    help="bank root for '~/chess-artifacts' (default env CHESS_ARTIFACTS_ROOT, else ~/chess-artifacts)")
    ap.add_argument("--tz", default=None,
                    help="IANA zone for mtime stamps (default: the prefix source's 'Stamp time zone' line)")
    ap.add_argument("--allow-list", type=Path, default=None,
                    help="MISSING allow-list (default <record stem>.stamp-allowlist.txt beside each record)")
    ap.add_argument("--quiet", action="store_true", help="print only non-OK lines and the summary")
    args = ap.parse_args(argv)

    root = args.artifacts_root or Path(os.environ.get("CHESS_ARTIFACTS_ROOT") or os.path.expanduser(ARTIFACTS_TILDE))
    records = args.records or [DEFAULT_LOG]
    try:
        tz = args.tz or read_declared_tz(args.prefix_source)
        if not tz:
            raise UsageError(f"no 'Stamp time zone: `<zone>`' line in {args.prefix_source}; pass --tz")
        verifier = Verifier(read_prefix_map(args.prefix_source, root), tz)
        failed = False
        for record in records:
            if not record.exists():
                raise UsageError(f"record not found: {record}")
            allow_path = args.allow_list or record.with_name(record.stem + ".stamp-allowlist.txt")
            allowed = read_allow_list(allow_path)
            cites = verifier.check_file(record)
            counts = {"OK": 0, "DRIFT": 0, "MISSING": 0, "MALFORMED": 0, "ALLOWED": 0}
            missing_tokens = set()
            for c in cites:
                status = c.status
                if status == "MISSING":
                    missing_tokens.add(c.token)
                    if c.token in allowed:
                        status = "ALLOWED"
                counts[status] += 1
                if not (args.quiet and status in ("OK", "ALLOWED")):
                    print(f"{status} {c.file}:{c.line} {c.token} -- {c.detail}")
            for token in sorted(set(allowed) - missing_tokens):
                print(f"STALE-ALLOW {allow_path}: {token} no longer MISSING (remove it)")
            print(f"SUMMARY {record}: {len(cites)} citations, OK {counts['OK']}, DRIFT {counts['DRIFT']}, "
                  f"MISSING {counts['MISSING']} (+{counts['ALLOWED']} allow-listed), MALFORMED {counts['MALFORMED']}")
            failed = failed or counts["DRIFT"] > 0 or counts["MISSING"] > 0 or counts["MALFORMED"] > 0
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
