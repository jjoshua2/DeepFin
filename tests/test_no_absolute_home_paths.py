"""No TRACKED file may leak the maintainer's identity or local layout.

This repository is PUBLIC. An absolute home path leaks the maintainer's
username and local directory layout into every clone and every diff, and it is
also a plain defect: a default that names one machine's filesystem either fails
outright on any other checkout or -- worse -- silently resolves to something
stale. `docs/REVIEW_BUG_HUNT.md` findings F035/F036/F037 are three separate
rounds of exactly that, each fixed the same way (derive the repo root from the
file's own location, allow an env override), and each time the paths grew back
somewhere else because nothing failed when they did. This is the something.

⚑ THE FILE SET COMES FROM `git ls-files`, not a glob, for the same reason
`test_worker_secret_sources.py` gives: what makes a path disclosed is that it
is TRACKED. A `scripts/**` + `docs/**` glob would have scanned neither the
top-level `TRAINING_LOG.md` nor `CLAUDE.md`, both of which carried hits.

⚑⚑ FOUR SHAPES, NOT ONE -- AND THE COUNT WENT UP TWICE
------------------------------------------------------
The first revision of this guard matched `/home/<user>` only, and announced that
"the maintainer's absolute home path" was gone -- while other shapes of the SAME
disclosure stayed tracked and green, because the pattern structurally could not
match them. Each round of review found the shape the previous round's census had
declared clean:

* `session_scratch` -- the agent scratchpad root `/tmp/claude-<uid>/-home-<user>-...`.
  The home path is FLATTENED into a single directory name, so there is no `/home/`
  substring anywhere in it. 7 tracked instances (6 in `docs/experiment_ledger.md`,
  1 in `docs/lc0_adapter_probe_run.md`).
* `worker_username` -- `distributed_worker_username: <name>` in 17 configs, and
  in SEVEN places in `tests/test_worker_secret_sources.py`. Not a path at all, so
  no path pattern will ever see it.
* `hostname` -- a machine NAME. `tests/test_server_telemetry_redaction.py` carried
  the maintainer's real hostname in five tracked places while this guard reported
  "outside configs: 0". A hostname is neither a `/home/` path nor a
  `distributed_worker_username` scalar, so all three patterns above swept past it.
  Found by the second independent review of PR #441, in a file class -- a test
  fixture with a real identifier baked in -- that the `worker_username` widening
  had already caught once, one directory over.

⚑⚑ AND A SHAPE IS NOT ALWAYS A PATTERN. `docs/experiment_ledger.md` carried
`TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_<user>` -- the login, in a `/tmp`
path, under no key at all. `abs_home` needs `/home/`; `session_scratch` needs
`-home-`; `worker_username` and `hostname` need their key. Nothing keyed or
path-shaped can see it, and no pattern that could be written down here would,
because the only thing that identifies it IS the login -- and writing the login
into this file to detect it would BE the disclosure.

So there is a fifth check that is not a pattern at all:
`test_no_tracked_file_carries_this_machines_identity` derives the tokens from
the RUNNING MACHINE (`getpass.getuser()`, `socket.gethostname()`, `Path.home()`)
and sweeps for them as whole words. It catches ANY shape -- bare hostname,
`/tmp/..._<user>`, `<user>@host`, a yaml anchor under an unrelated key -- and it
is the only check that can.

⚑ ITS LIMIT, STATED, because a check that reports clean where it structurally
cannot fire is this repo's signature defect: it only fires on a machine whose
identity is the leaked one. On a CI runner it sweeps a tree that contains no
token of that runner and passes vacuously. That is not a reason to omit it --
tracked files are written on the machine whose identity leaks, which is exactly
where it does fire -- but it means the PORTABLE guarantee is the four patterns
below and nothing more. The sweep's MECHANISM is proved on every host by
`test_the_local_identity_sweep_can_actually_fire`, which feeds it an explicit
token; only the tree half is machine-dependent.

A guard that reports "clean" while naming only the shape it happens to match is
worse than no guard, because it retires the question.

Scanning is WHOLE-BLOB, not line-by-line
----------------------------------------
The first two revisions ran each pattern against each line separately, which
made every multi-line spelling of a leak invisible: a yaml value on the line
after its key, a `>`/`|` block scalar, and an absolute path wrapped across a
markdown line all swept green (measured by the reviewer of PR #441). The
patterns now run over the whole file and the line number is derived from the
match offset.

⚑⚑ AND COUNTED PER OCCURRENCE, NOT PER LINE. The revision that introduced
whole-blob scanning still collapsed each line to one hit, and called it
deliberate. The next review demonstrated the hole in one command: append a
third `/home/<user>/...` to `configs/pbt2_small.yaml`'s existing `syzygy_path`
line and the guard stays green, because that line was already spending its
budget. The budgets below are occurrence counts now, which is why
`pbt2_small.yaml` reads 12 rather than 11 -- the `syzygy_path` PAIR was always
two leaks on one line, and the old pin recorded it as one.

What the patterns deliberately do NOT match
-------------------------------------------
`/home/<user>/...` -- the angle-bracket placeholder. Several prose passages are
*about* hardcoded home paths ("`scripts/monitor_pbt.sh` hardcoded ...") and
their meaning does not survive being rewritten into a repo-relative form, so
they keep the shape and lose only the username. `<` is not in any character
class here, so every placeholder form reads as documentation rather than as a
leak -- including `/tmp/claude-1000/<session-id>/`,
`distributed_worker_username: <name>` and `hostname: <host>`. `~/...` is
likewise fine and is what the ledger's recorded commands now use: it is
shell-expandable, copy-pasteable, and names no user.

Only `/home/` is covered -- not `/Users/` (macOS) or `C:\\Users\\` (Windows).
Neither has ever appeared here, and a pattern with no possible trigger is a
gate that cannot fail; add one when there is something for it to catch.

⚑ KNOWN, UNCLOSED, AND WRITTEN DOWN RATHER THAN IMPLIED
-------------------------------------------------------
MEASURED by running 26 spellings against these patterns, not reasoned about: 19
are caught, 7 are not, and the 7 are all ONE limit wearing different clothes --
**an identifier that is not under a key this file names, and not inside a path
shape.** Specifically:

  * a login under some OTHER key -- `worker_ssh: <login>@host`;
  * a yaml ANCHOR DEFINITION -- `_u: &u <login>` -- which
    `distributed_worker_username: *<alias>` then refers to. (The alias
    REFERENCE is caught: an unresolvable `*alias` is a finding, because the
    guard cannot see what it stands for. The definition is not.)
  * a machine name under `machine:` / `node_name:` / inside an f-string;
  * a login in free text -- a git URL, `~<login>/projects`.

Two of those could be "closed" and are deliberately not:

  * `~<login>/` LOOKS cheap and is not. In this tree `~` almost always means
    *approximately* -- `~118s/`, `~1/`, `~len/` -- so the shape has a false
    positive on ordinary prose in a dozen tracked files, and a guard that cries
    wolf is one people re-run rather than read.
  * WIDENING THE HOSTNAME KEY SET to `machine:` is worse than useless: in this
    repo `machine:` overwhelmingly means an ARCHITECTURE (`x86_64`), so the
    shape would fire on non-leaks and teach the reader to ignore it.

On the machine that authors those files the identity sweep catches every one of
the seven; on CI nothing does. That is the honest boundary of this file, and it
is why the sweep exists rather than being replaced by "one more pattern".
"""
from __future__ import annotations

import getpass
import os
import re
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The scalar half of a `<key>: <value>` shape, shared by `worker_username` and
# `hostname` so the two cannot drift apart. `%s` is the value's character class.
#
# TWO BRANCHES, and the split is load-bearing:
#
#  1. NEXT LINE. `key:` with the value on the following, more-indented line is
#     legal YAML — and so is a `>` / `|` block scalar — and a line-scoped scan
#     could see neither (reviewer smuggles #1 and #2). ⚑ The continuation only
#     counts when the value is THE WHOLE LINE (optionally trailed by a comment),
#     because `if hostname:` in Python is also "a colon at end of line followed
#     by an indented block": without that anchor this branch read `entry` out of
#     `chess_anti_engine/server/app.py` and reported the server as leaking.
#  2. SAME LINE. The trailing lookahead is what keeps a Python EXPRESSION from
#     reading as a scalar — `str(username)` names a variable, not a person, and
#     `entry[...]` is a subscript, not a machine. ⚑ It does NOT exclude `:`,
#     which an earlier revision did: that was covering for the next-line branch
#     before the whole-line anchor existed, and its cost was that `hostname:
#     <host>:8000` — a machine name with a PORT — swept green.
#
# Either branch may be an ALIAS REFERENCE (`*u`), which the guard cannot resolve
# and therefore refuses to clear: unverifiable is not the same as neutral
# (reviewer smuggle #3, key half). The `(?!<)` sits AFTER the optional quote, so
# the documentation placeholder stays exempt written `<name>` or `"<name>"`.
#: Optional yaml decorations between a key's colon and its value: a `>`/`|`
#: block-scalar indicator (with chomping/indent digits) and/or an explicit tag
#: (`!!str`). Both are legal, both are what a yaml author reaches for when a
#: value needs quoting, and neither was matched.
_YAML_DECORATION = rb"(?:[ \t]*(?:[>|][-+0-9]*|!![a-z]+(?=[ \t\r\n])))*"

_VALUE = (
    rb"(?:"
    # NEXT-LINE branch. A blank line between key and value is legal, and a `- `
    # sequence item is how a one-element list gets written.
    + _YAML_DECORATION + rb"[ \t]*(?:\r?\n[ \t]*)*\r?\n[ \t]+(?:-[ \t]+)?"
    rb"[\"']?(?!<)(\*[A-Za-z0-9._-]+|%s)"
    rb"[\"']?[ \t]*(?:#[^\n]*)?(?=\r?\n|\Z)"
    rb"|"
    # SAME-LINE branch.
    + _YAML_DECORATION + rb"[ \t]*[\"']?(?!<)(\*[A-Za-z0-9._-]+|%s)"
    rb"(?![A-Za-z0-9._(\[-])"
    rb")"
)

# ⚑ BYTES, not str. The first revision read files as UTF-8 text and returned []
# on UnicodeDecodeError, with the comment "binary or unreadable: nothing to leak
# in text form" -- which is untrue of a pickle, an .npz or any container that
# stores paths as ASCII inside a binary envelope, and it exempted every tracked
# binary from the sweep BY CONSTRUCTION. Scanning bytes has no such hole and
# costs nothing.
_PATTERNS: dict[str, re.Pattern[bytes]] = {
    # A real home directory: `/home/` followed by at least one name character.
    #
    # ⚑ `/home/+`, not `/home/`: `/home//<user>/...` is a perfectly valid path
    # and a single trailing slash let it through (reviewer smuggle #4).
    # ⚑ And an optional LINE WRAP between the slash and the name: docs here wrap
    # at ~88 columns, so `/home/` at end-of-line with the login starting the next
    # one is a realistic spelling that a line-scoped scan could never see
    # (reviewer smuggle #5).
    # ⚑ `[\\/]home[\\/]+`, not `/home/`: three spellings, one leak.
    #   `/home//<user>` — a double slash is a valid path and a single trailing
    #     slash let it through (reviewer smuggle #4);
    #   `\home\<user>` — the WSL/UNC form, which this project actually uses;
    #   `\/home\/<user>` — the JSON/JS-escaped form.
    # Zero tracked files contain a backslash `home` today, so widening here is
    # free; it is the shape that shows up the moment somebody pastes a Windows
    # path into a doc.
    "abs_home": re.compile(rb"[\\/]home[\\/]+[ \t]*(?:\r?\n[ \t]*)?[A-Za-z0-9._-]"),
    # The agent-session scratchpad root, where the home path is flattened into
    # one directory name (`-home-<user>-projects-chess`) and so contains no
    # `/home/` substring.
    #
    # ⚑ NOT anchored on a leading `/`. The first revision was (`/-home-`), and
    # the reviewer got the bare token past it: `-home-<user>-projects-chess`
    # with no slash in front is exactly how the flattened directory NAME appears
    # in `ls`, `tar` and `rsync` output and in prose that names a scratchpad
    # bank. The boundary is "not preceded by a name character", which still
    # refuses to fire inside an ordinary hyphenated word (`multi-home-server`).
    # ⚑ Case-insensitive: the boundary, not the case, is what keeps
    # `multi-home-server` out, so ignoring case costs nothing and catches the
    # token as it appears in an upper-cased log or a shouted heading.
    "session_scratch": re.compile(rb"(?<![A-Za-z0-9])-home-[A-Za-z0-9]", re.IGNORECASE),
    # Not a path: the fleet's SSH username, carried by every distributed config.
    #
    # ⚑⚑ THE QUOTES ARE THE WHOLE POINT. The first revision opened its character
    # class immediately after the whitespace, so a QUOTED yaml scalar never
    # matched — `"` is not `<`, so the negative lookahead passed, and then
    # `[A-Za-z0-9._-]` failed on the quote. MEASURED by the independent reviewer:
    # a new tracked config carrying `distributed_worker_username: "<the real
    # login>"` swept GREEN, while the same file unquoted went red. YAML authors
    # habitually quote strings, so the shape most likely to land was the one
    # shape the guard could not see.
    #
    # The key may also be quoted (a JSON dump of the same config) and may carry
    # spaces before its colon (`key : value` is legal YAML). The value half is
    # `_VALUE` above, shared with `hostname`; it is CAPTURED so it can be
    # checked against `_NEUTRAL_USERNAMES` below.
    "worker_username": re.compile(
        rb"distributed_worker_username[\"']?[ \t]*:" + _VALUE % (
            rb"[A-Za-z0-9._-]+", rb"[A-Za-z0-9._-]+",
        ),
    ),
    # A machine NAME. Same construction as `worker_username`, and it exists for
    # the same reason: the second review of PR #441 found the maintainer's real
    # hostname in five tracked places while every pattern above reported clean.
    #
    # Covers `hostname`, `host_name`, and any `<prefix>_hostname` the telemetry
    # accumulator uses (`last_hostname`, `worker_hostname`). Values are
    # deny-by-default against `_NEUTRAL_HOSTNAMES`, exactly like the login.
    #
    # ⚑ The value class starts `[A-Za-z0-9]`, so a Python identifier (`_HOST`)
    # or an expression (`str(...)` / `entry[...]`, caught by `_VALUE`'s trailing
    # lookahead) is not a finding: those name a variable, not a machine.
    "hostname": re.compile(
        rb"(?<![A-Za-z0-9])(?:[a-z][a-z0-9]*_)?host_?name[\"']?[ \t]*:" + _VALUE % (
            rb"[A-Za-z0-9][A-Za-z0-9.-]*", rb"[A-Za-z0-9][A-Za-z0-9.-]*",
        ),
        re.IGNORECASE,
    ),
}

#: Login names that are NOT anybody's account — test fixtures and the guard's
#: own controls.
#:
#: ⚑ DENY BY DEFAULT. The shape is a finding unless its value is named here, so
#: a config carrying a real login is red without anyone having to predict it.
#: The inverse — permit everything, list the bad names — cannot be written in a
#: public repo at all, because the list would BE the disclosure.
#:
#: ⚑ This list is what caught the leak the widened pattern was written for:
#: `tests/test_worker_secret_sources.py` carried the maintainer's actual login
#: in seven tracked places, and the narrower pre-review pattern matched none of
#: them. Scrubbed to `fleetuser` in the same commit.
#:
#: Limit, stated: a real account that happens to be called `worker` would pass.
#: That is the price of a list that can be written down publicly.
#: ⚑ `exampleuser` is deliberately NOT here. It is the controls' value, and a
#: control whose value is neutral proves only that the regex matched — not that
#: the guard would report it. The positive controls have to survive the polarity
#: check too, or `test_every_pattern_can_actually_fire` is measuring half the
#: detector.
_NEUTRAL_USERNAMES: frozenset[bytes] = frozenset({
    b"worker", b"tune_worker_old", b"fleetuser",
})

#: The same polarity for machine names. `box1`..`box4` are
#: `tests/test_server_trial_lease.py`'s fixtures; `fleetbox-01` is
#: `tests/test_server_telemetry_redaction.py`'s, which replaced the real one.
_NEUTRAL_HOSTNAMES: frozenset[bytes] = frozenset({
    b"box0", b"box1", b"box2", b"box3", b"box4",
    b"fleetbox-01", b"examplehost", b"localhost",
})

_NEUTRAL_VALUES: dict[str, frozenset[bytes]] = {
    "worker_username": _NEUTRAL_USERNAMES,
    "hostname": _NEUTRAL_HOSTNAMES,
}

# Built by concatenation ON PURPOSE. This file is scanned by its own sweep
# (excluding it would leave a hole exactly where the patterns are defined), so a
# literal leak string here would make the guard report itself.
_CONTROLS: dict[str, str] = {
    "abs_home": "/home/" + "exampleuser" + "/projects/chess/data/syzygy_6",
    "session_scratch": "/tmp/claude-1000" + "/-home-" + "exampleuser" + "-projects-chess",
    "worker_username": "distributed_worker_username:" + " exampleuser",
    "hostname": "last_" + "hostname:" + " gpu-rig-7",
}

#: ⚑ THE VARIANTS THAT HAVE ACTUALLY BEEN SMUGGLED PAST THIS GUARD — not a claim
#: to cover "every shape the leak can take", which is what the previous revision
#: of this comment said and which the review then falsified twice. A control
#: proves a pattern CAN fire; these prove it fires on the spellings a human (or
#: a reviewer trying to get one past it) actually produces. Every entry below was
#: GREEN under some earlier revision of the pattern above:
#:   - the quoted username swept green past the whole guard (review P1);
#:   - the bare flattened scratchpad token swept green past it too (review P2);
#:   - the double slash, the wrapped path, the next-line value and the block
#:     scalar swept green past the line-scoped scan (review F3, smuggles 1/2/4/5);
#:   - the bare hostname had no shape at all (review F1).
_VARIANT_CONTROLS: dict[str, tuple[str, ...]] = {
    "abs_home": (
        "/home/" + "exampleuser",
        "cd /home/" + "exampleuser" + "/projects/chess",
        '"/home/' + 'exampleuser' + '/projects/chess/data"',
        # ← smuggle #4: a double slash is still a valid path.
        "/home//" + "exampleuser" + "/projects/chess",
        # ← smuggle #5: wrapped across a markdown line at ~88 columns.
        "the pair is /home/" + "\n" + "exampleuser" + "/projects/chess/data",
        # Found by the fixer's own smuggle pass, after the six above were closed:
        "\\\\wsl$\\Ubuntu\\home\\" + "exampleuser" + "\\projects",   # WSL/UNC
        '{"p": "\\/home\\/' + 'exampleuser' + '\\/x"}',                # JSON-escaped
        "/home/" + "\r\n" + "exampleuser" + "/projects",             # CRLF wrap
    ),
    "session_scratch": (
        "/tmp/claude-1000" + "/-home-" + "exampleuser" + "-projects-chess",
        # No leading slash: how the flattened directory NAME appears bare.
        "-home-" + "exampleuser" + "-projects-chess/scratchpad/x.json",
        "see " + "-home-" + "exampleuser" + "-projects-chess",
        # Upper-cased, e.g. in a shouted heading or an upper-cased log.
        "-HOME-" + "EXAMPLEUSER" + "-PROJECTS-CHESS",
    ),
    "worker_username": (
        "distributed_worker_username:" + " exampleuser",
        'distributed_worker_username:' + ' "exampleuser"',      # ← the P1 hole
        "distributed_worker_username:" + " 'exampleuser'",
        "distributed_worker_username:" + "exampleuser",
        "distributed_worker_username" + " : exampleuser",
        '"distributed_worker_username":' + ' "exampleuser"',     # JSON dump
        # ← smuggle #1: the value on the following line. Legal YAML.
        "distributed_worker_username:\n  " + "exampleuser",
        # ← smuggle #2: folded and literal block scalars.
        "distributed_worker_username: >\n  " + "exampleuser",
        "distributed_worker_username: |\n  " + "exampleuser",
        # ← smuggle #3 (key half): an alias whose anchor this guard cannot see.
        #   Unresolvable is not the same as neutral, so it is a finding.
        "distributed_worker_username:" + " *u",
        # Found by the fixer's own smuggle pass, after the six above were closed:
        "distributed_worker_username: !!str " + "exampleuser",     # explicit tag
        "distributed_worker_username:\n\n  " + "exampleuser",       # blank line
        "distributed_worker_username:\n  - " + "exampleuser",       # list item
        "{distributed_worker_username: " + "exampleuser" + "}",     # flow mapping
    ),
    "hostname": (
        "last_" + "hostname:" + " gpu-rig-7",
        '"last_' + 'hostname":' + ' "gpu-rig-7"',
        "host" + "name:" + " gpu-rig-7",
        "host_" + "name:" + " gpu-rig-7",
        "worker_" + "hostname:" + " gpu-rig-7",
        "last_" + "hostname:" + "\n  gpu-rig-7",
        # Found by the fixer's own smuggle pass: a machine name with a PORT.
        # The same-line lookahead used to exclude `:`, which the next-line
        # branch's end-of-line anchor already makes unnecessary.
        "host" + "name:" + " gpu-rig-7:8000",
        '{"last_' + 'hostname":' + '"gpu-rig-7"}',                       # JSON one-liner
    ),
}

_PLACEHOLDER_CONTROLS: dict[str, tuple[str, ...]] = {
    "abs_home": ("/home/" + "<user>" + "/projects/chess",),
    "session_scratch": (
        "/tmp/claude-1000/" + "<session-id>" + "/scratchpad",
        # An ordinary hyphenated word must not become a finding.
        "multi-home-server", "the -homeostasis- note",
    ),
    "worker_username": (
        "distributed_worker_username:" + " <name>",
        # ⚑ The QUOTED placeholder too: prose about this defect gets quoted the
        # same way real yaml does, and documentation must not read as a leak.
        'distributed_worker_username:' + ' "<name>"',
        "distributed_worker_username:" + " '<name>'",
        # An EMPTY value must not capture the NEXT key as the login.
        "distributed_worker_username:\n  distributed_worker_password: x",
        # A Python expression names a variable, not a person.
        '"distributed_worker_username":' + " str(username),",
    ),
    "hostname": (
        "host" + "name:" + " <host>",
        '"last_' + 'hostname":' + ' "<host>"',
        # The neutral fixtures must stay non-findings, or every server test
        # that needs a machine name reads as a leak.
        '{"host' + 'name": "box1"}',
        '"last_' + 'hostname":' + ' "fleetbox-01"',
        # A Python identifier or expression as the value.
        '"last_host' + 'name": _FIXTURE_HOST,',
        '"host' + 'name": str(socket.gethostname()),',
    ),
}

# ⚑⚑ AN EXEMPTION MUST ARGUE INFEASIBILITY, NOT UNIMPORTANCE. Until the #441
# review, 16 of these 17 read "research config, default off" — which is a reason
# nobody would MIND the leak, not a reason it cannot be removed, and it would
# have justified exempting anything anyone did not care about. The real blocker
# is below, and it applies to all 17 equally.
_YAML_HAS_NO_SELF_LOCATION = (
    "yaml cannot locate itself: every functional scrub in this repo works by "
    "deriving the root from __file__ / $0, and a config has neither. The "
    "alternatives are a relative path (meaning then depends on the loader's cwd "
    "— unproven against TrialConfig.from_dict and the tune harness) or ~ (which "
    "the loader does not expand). Several paths also point OUTSIDE the checkout "
    "(syzygy_path's 151G 6-man tablebases) and have no repo-relative form at all"
)
_LIVE_CONFIG_REASON = (
    f"{_YAML_HAS_NO_SELF_LOCATION}; AND this one is production, re-read every "
    "iteration on the live branch, and key changes are restart-gated"
)
_RESEARCH_REASON = _YAML_HAS_NO_SELF_LOCATION

# (path, number of `abs_home` hits present when the exemption was written)
_CONFIGS = (
    ("configs/pbt2_small.yaml", 12),
    ("configs/scratch_pc.yaml", 11),
    ("configs/bt4_aurora_asha.yaml", 9),
    ("configs/exp_cat_search.yaml", 9),
    ("configs/exp_categorical_continuous.yaml", 9),
    ("configs/exp_dynamic_relations.yaml", 9),
    ("configs/exp_label_escalation.yaml", 11),
    ("configs/exp_repetition_fix.yaml", 9),
    ("configs/exp_soft_policy_ablation.yaml", 9),
    ("configs/exp_soft_policy_divergent_only.yaml", 9),
    ("configs/exp_threat_planes.yaml", 9),
    ("configs/exp_throughput_views.yaml", 9),
    ("configs/exp_v3_checks.yaml", 9),
    ("configs/exp_v3_passers.yaml", 9),
    ("configs/exp_v3_see.yaml", 9),
    ("configs/exp_v3_xray.yaml", 9),
    ("configs/exp_volatility_search.yaml", 9),
)

# Paths that still leak, each for a stated reason, EXEMPT PER SHAPE AND PER COUNT.
#
# ⚑ Per shape, not per file. The first revision skipped an allowlisted file
# WHOLLY, so `configs/pbt2_small.yaml` could grow an unrelated new leak and stay
# green -- an exemption for one known reason silently became an exemption for
# everything. A file listed here is still swept for every shape it does not name.
#
# ⚑⚑ AND PER COUNT, since the #441 review. Exempting the SHAPE was still too
# coarse: it subtracted the whole detector before the file was scanned, so a
# config could gain a SECOND, unrelated absolute path -- or a fleet login where
# it previously had none -- and stay green for as long as one old exempt
# occurrence survived. These 17 configs are the files most likely to acquire a
# fresh machine-specific path, which made them the worst possible place to turn
# the detector off. The number below is the number of hits that were there when
# the exemption was written; hit N+1 is a finding, and hit N-1 fails the
# staleness pin. (Codex inline review.)
#
# ⚑ EXACT names, never a `configs/` prefix rule: a prefix rule would let a NEW
# config land with a fresh leak and stay green.
_ALLOWED: dict[str, dict[str, tuple[int, str]]] = {
    # `syzygy_path` and friends point OUTSIDE the checkout (the 151G 6-man
    # tablebases are not in the repo), so there is no repo-relative form, and
    # these files are edited on the LIVE branch while a run re-reads them every
    # iteration -- a merge conflict here reverts live experiments. Scrubbing
    # them is a separate, operator-scheduled change.
    #
    # `distributed_worker_username` is the SSH login the fleet actually uses.
    # It is a real value, not a path, so there is no derived form either; it
    # goes when these configs are scrubbed, in the same operator-scheduled
    # change, and `pbt2_small.yaml` in particular is RESTART-GATED.
    **{
        rel: {
            "abs_home": (
                n_abs_home,
                _LIVE_CONFIG_REASON if rel == "configs/pbt2_small.yaml" else _RESEARCH_REASON,
            ),
            "worker_username": (
                1, "fleet SSH login; scrubbed with the config, restart-gated",
            ),
        }
        for rel, n_abs_home in _CONFIGS
    },
    # PINNED BY A TEST, so it cannot be scrubbed on its own:
    # `test_param_count.py::test_claude_md_syzygy_pair_matches_the_production_config`
    # requires CLAUDE.md to quote `configs/pbt2_small.yaml`'s `syzygy_path`
    # VERBATIM, and to quote no other pair. The doc is downstream of the config;
    # it goes when the config goes.
    "CLAUDE.md": {
        "abs_home": (2, "must quote the production syzygy_path verbatim (test_param_count)"),
    },
}

#: Tokens that are somebody's identity on one machine and an ordinary English
#: word (or a CI runner's account) everywhere else. Sweeping for these would
#: turn a green CI red on a false positive, which is the fastest way to teach
#: people to ignore a guard.
_GENERIC_IDENTITY_TOKENS: frozenset[str] = frozenset({
    "user", "users", "root", "admin", "administrator", "build", "builder",
    "runner", "ubuntu", "debian", "docker", "container", "github", "actions",
    "jenkins", "vagrant", "codespace", "localhost", "worker", "test", "tests",
    "home", "guest", "default", "chess", "main", "master",
})


def _local_identity_tokens() -> list[str]:
    """Whole-word tokens that identify the machine this sweep is running on.

    ⚑ DERIVED, NEVER WRITTEN DOWN. The login and hostname cannot be literals in
    a public repo -- the denylist would be the disclosure. Reading them from the
    environment gives a deny-by-default check with nothing to disclose.
    """
    host = socket.gethostname() or ""
    raw = {getpass.getuser(), host, host.split(".")[0], Path.home().name}
    tokens = set()
    for token in raw:
        clean = (token or "").strip().lower()
        # Under 4 characters a token collides with ordinary text far more often
        # than it identifies anybody; `_GENERIC_IDENTITY_TOKENS` covers the rest.
        if len(clean) < 4 or clean in _GENERIC_IDENTITY_TOKENS:
            continue
        if clean.encode() in _NEUTRAL_USERNAMES or clean.encode() in _NEUTRAL_HOSTNAMES:
            continue
        tokens.add(clean)
    return sorted(tokens)


def _identity_hits(blob: bytes, tokens: list[str]) -> list[tuple[str, int, str]]:
    """(token, line number, line) for every WHOLE-WORD occurrence of a token.

    The boundary is "not adjacent to an alphanumeric", so `_` and `-` and `/`
    are all boundaries — `/tmp/torchinductor_<user>` and `<user>-desktop` are
    both findings — while a token embedded in a longer word (a public pseudonym
    that happens to contain the login) is not.
    """
    found: list[tuple[str, int, str]] = []
    for token in tokens:
        pattern = re.compile(
            rb"(?<![A-Za-z0-9])" + re.escape(token.encode()) + rb"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(blob):
            lineno = blob.count(b"\n", 0, match.start()) + 1
            line = blob.split(b"\n")[lineno - 1]
            found.append((token, lineno, line.decode("utf-8", "replace").strip()[:200]))
    return found


def _tracked_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return sorted(REPO_ROOT / name for name in listed.split("\0") if name)


def _read(path: Path) -> bytes:
    """The bytes git published for ``path``.

    ⚑ A TRACKED SYMLINK IS ITS TARGET STRING, not the file it points at. Git
    stores the target verbatim in the tree, so `ln -s /home/<user>/private x`
    publishes that path to every clone — while `read_bytes()` FOLLOWS the link
    and scans something else entirely (or raises, and the `except OSError`
    below returned no hits, so the disclosure swept green). `os.readlink` reads
    what git actually stored. Codex inline review, #441.
    """
    if path.is_symlink():
        try:
            return os.readlink(path).encode("utf-8", "surrogateescape")
        except OSError:
            return b""
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _scan(shape: str, blob: bytes) -> list[tuple[str, int, str]]:
    """(shape, line number, line) for one shape over one blob.

    ⚑ The polarity check lives HERE, not in the sweep, so the controls exercise
    the same code the tree does. `_VALUE` has two alternative branches (next
    line / same line) and therefore two capture groups; whichever one matched is
    the value, and a shape with no groups at all (`abs_home`) has no value to
    deny.

    ⚑⚑ ONE HIT PER OCCURRENCE, NOT PER LINE. Two earlier revisions collapsed
    every match on a line to a single hit, and the docstring called it
    deliberate ("so the budgets keep the meaning they were measured with").
    It was a HOLE, and the #441 review demonstrated it: a new absolute path
    APPENDED TO AN ALREADY-LEAKING LINE cost nothing. `configs/pbt2_small.yaml`
    line 522 is `syzygy_path: <path>:<path>` — two occurrences, one line — so a
    third could be pasted onto that same line and the sweep stayed green. The
    line numbers repeat in the output now, which is correct: two leaks on one
    line are two leaks, and the budgets below are re-pinned to the occurrence
    counts (`abs_home` on that file went 11 → 12 for exactly this reason).
    """
    neutral = _NEUTRAL_VALUES.get(shape)
    lines = blob.split(b"\n")
    found: list[tuple[str, int, str]] = []
    for match in _PATTERNS[shape].finditer(blob):
        if neutral is not None:
            value = next((g for g in match.groups() if g is not None), b"")
            if value in neutral:
                continue
        lineno = blob.count(b"\n", 0, match.start()) + 1
        found.append(
            (shape, lineno, lines[lineno - 1].decode("utf-8", "replace").strip()[:200])
        )
    return found


def _hits(path: Path, shapes: frozenset[str] | None = None) -> list[tuple[str, int, str]]:
    """(shape, line number, line) for every leak in ``path``.

    Reads BYTES so a binary file is scanned rather than exempted; the offending
    line is decoded only to render the failure message.

    ⚑ WHOLE-BLOB, one hit per OCCURRENCE. Scanning line by line made every
    multi-line spelling invisible (a yaml value on the next line, a `>` block
    scalar, a path wrapped by a markdown line); scanning the blob sees them. A
    hit is attributed to the line its match STARTS on, and a line that carries
    two occurrences of a shape yields two hits — see `_scan`.
    """
    blob = _read(path)
    wanted = sorted(_PATTERNS if shapes is None else (set(_PATTERNS) & set(shapes)))
    found: list[tuple[str, int, str]] = []
    for shape in wanted:
        found.extend(_scan(shape, blob))
    return sorted(found, key=lambda h: (h[1], h[0]))


def _offenders(paths: list[Path], *, root: Path = REPO_ROOT) -> list[str]:
    """Leaks in ``paths`` that the allowlist does not budget for.

    ⚑ EVERY shape is scanned on EVERY file, including allowlisted ones. The
    exemption is applied to the HITS afterwards, as a BUDGET, not to the
    detector beforehand as an off switch — see `_ALLOWED`. Factored out of the
    sweep so the budget arithmetic itself can be tested on a fixture rather than
    only on a tree that is (correctly) clean.
    """
    offenders: list[str] = []
    for path in paths:
        rel = str(path.relative_to(root))
        allowed = _ALLOWED.get(rel, {})
        hits = _hits(path)
        for shape in sorted(_PATTERNS):
            of_shape = [(lineno, line) for sh, lineno, line in hits if sh == shape]
            budget = allowed.get(shape, (0, ""))[0]
            if len(of_shape) <= budget:
                continue
            # Which specific lines are the NEW ones cannot be known from the
            # count alone, so the whole set is named with both numbers and the
            # operator reads the diff.
            offenders.append(
                f"{rel}: [{shape}] {len(of_shape)} hits, {budget} allowed — "
                f"e.g. line {of_shape[-1][0]}: {of_shape[-1][1]}"
            )
    return offenders


def test_every_pattern_can_actually_fire() -> None:
    """Positive AND placeholder controls, per shape.

    A shape with no control is a shape nobody has proved can match -- which is
    how `session_scratch`, `worker_username` and `hostname` each went unnoticed
    for a round.
    """
    assert (set(_CONTROLS) == set(_PATTERNS) == set(_PLACEHOLDER_CONTROLS)
            == set(_VARIANT_CONTROLS)), (
        "every shape needs a positive control, its realistic variants, and a "
        "placeholder control"
    )
    for shape in _PATTERNS:
        assert _scan(shape, _CONTROLS[shape].encode()), (
            f"the {shape!r} shape reports nothing"
        )
        for variant in _VARIANT_CONTROLS[shape]:
            assert _scan(shape, variant.encode()), (
                f"the {shape!r} shape misses a REALISTIC form of the same "
                f"leak: {variant!r}. A leak that only the tidiest spelling "
                "trips is a guard that the untidy spelling walks past."
            )
        for placeholder in _PLACEHOLDER_CONTROLS[shape]:
            assert not _scan(shape, placeholder.encode()), (
                f"the {shape!r} placeholder {placeholder!r} must NOT be a "
                "finding — the prose that documents this defect class is "
                "written in it"
            )
    # And the split-literal trick above must really be invisible to the sweep,
    # or this file reports itself and every reader learns to ignore the guard.
    assert not _hits(Path(__file__)), (
        "this guard file matches its own patterns; the controls must stay split"
    )


def test_a_binary_file_is_scanned_not_exempted(tmp_path: Path) -> None:
    """The UTF-8 hole, closed and pinned.

    A pickle or .npz stores ASCII paths inside a binary envelope. Reading as
    text and swallowing UnicodeDecodeError exempted every such file BY
    CONSTRUCTION while the docstring claimed there was "nothing to leak in text
    form".
    """
    blob = tmp_path / "fixture.npz"
    blob.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe" + _CONTROLS["abs_home"].encode() + b"\x00\xff")
    hits = _hits(blob)
    assert [shape for shape, _, _ in hits] == ["abs_home"], hits


def test_a_multi_line_leak_is_found_and_reported_on_its_STARTING_line(
    tmp_path: Path,
) -> None:
    """⚑ Review F3, smuggles #1/#2/#5: every multi-line spelling swept green.

    The scan was line-scoped, so a yaml value on the line after its key, a
    `>`/`|` block scalar and an absolute path wrapped by a markdown line were
    all invisible — not because the patterns were wrong but because the input
    was chopped before they ever ran. The blob is scanned whole now, and the hit
    is attributed to the line the match STARTS on so the exact-count budgets in
    `_ALLOWED` keep the meaning they were measured with.
    """
    fixture = tmp_path / "smuggled.yaml"
    fixture.write_bytes(
        b"# line 1\n"
        b"distributed_worker_username:\n  "
        + b"exampleuser".replace(b"example", b"real")
        + b"\n"
    )
    hits = _hits(fixture)
    assert [(shape, lineno) for shape, lineno, _ in hits] == [("worker_username", 2)], hits


def test_no_tracked_file_leaks() -> None:
    tracked = _tracked_files()
    assert len(tracked) > 100, (
        f"git ls-files returned only {len(tracked)} files — the sweep is vacuous"
    )

    offenders = _offenders(tracked)
    scanned = len(tracked)
    assert scanned > 100, f"only {scanned} files were actually scanned"
    assert not offenders, (
        "these TRACKED files leak the maintainer's identity or local layout, in "
        f"a PUBLIC repository: {offenders}. Derive paths from the file's own "
        "location instead — `Path(__file__).resolve().parents[1]` in Python, "
        '`cd \"$(dirname \"$0\")/..\"` in shell — or, if it is prose rather than '
        "a path the code uses, write it `~/...` (a recorded command), "
        "`<session-scratch>/...` (an agent scratchpad bank) or "
        "`/home/<user>/...` (prose about this defect itself). A real hostname "
        "or login in a test fixture gets a NEUTRAL one instead."
    )


def test_the_local_identity_sweep_can_actually_fire() -> None:
    """⚑ The mechanism, proved on EVERY host including a CI runner.

    The tree half of the identity sweep only fires on the machine whose identity
    leaked, so on CI it passes vacuously — a check that structurally cannot fail
    is this repo's signature defect and must not be the only thing standing
    behind a claim. This control feeds the scanner an EXPLICIT token, so the
    matching itself is falsifiable anywhere, and it uses the two shapes that
    actually walked past every keyed pattern in the tree.
    """
    token = "exampleuser"

    # The `/tmp` shape: the login under no key, in a path with no `/home/`.
    cache_dir = b"TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_" + token.encode()
    assert _identity_hits(cache_dir, [token]), (
        "a login inside a /tmp path is not detected; that is exactly the "
        "docs/experiment_ledger.md leak that four keyed patterns missed"
    )

    # The bare-hostname shape: a machine name as a plain string literal, with no
    # `hostname:` key anywhere near it.
    bare = b'    assert "' + token.encode() + b'-desktop" not in r.text'
    assert _identity_hits(bare, [token]), (
        "a bare hostname built from the login is not detected"
    )

    # ...and the login under an unrelated key (reviewer smuggle #6) and as a
    # yaml anchor definition (smuggle #3's other half), neither of which any
    # key-anchored pattern can reach.
    assert _identity_hits(b"worker_ssh: " + token.encode() + b"@host", [token])
    assert _identity_hits(b"_u: &u " + token.encode(), [token])

    # A token embedded in a longer word is NOT a finding: the repo's public
    # pseudonym contains the login as a substring and must stay readable.
    assert not _identity_hits(b"j" + token.encode() + b"3 wrote this", [token])
    assert not _identity_hits(b"the " + token.encode() + b"ing pattern", [token])


def test_no_tracked_file_carries_this_machines_identity() -> None:
    """⚑⚑ THE SHAPE THAT IS NOT A PATTERN.

    `docs/experiment_ledger.md` carried `/tmp/torchinductor_<user>` and
    `tests/test_server_telemetry_redaction.py` carried a real hostname in five
    places, both while the census above reported "outside configs: 0". Neither
    is a `/home/` path, and only one of the six was under a key any pattern
    names. What identifies them is the LOGIN — which cannot be written into a
    public repo to detect it.

    So the tokens come from the machine instead. LIMIT, STATED: this fires only
    where the identity is the local one, i.e. on the machine that authors the
    files, and passes vacuously on a CI runner. The mechanism is proved
    separately, above.

    The `_ALLOWED` files are skipped WHOLE here: their 162 remaining
    `/home/<user>` occurrences are deferred with a stated reason, and re-reporting
    them under a second name would just teach people to skip this test.
    """
    tokens = _local_identity_tokens()
    deferred = set(_ALLOWED)
    offenders: list[str] = []
    for path in _tracked_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in deferred or rel == str(Path(__file__).relative_to(REPO_ROOT)):
            continue
        for token, lineno, line in _identity_hits(_read(path), tokens):
            offenders.append(f"{rel}:{lineno} [{token}] {line}")
    assert not offenders, (
        f"tracked files carry this machine's identity ({tokens}) in a PUBLIC "
        f"repository: {offenders[:20]}. A hostname or login in a fixture gets a "
        "neutral stand-in; in a recorded command it gets `$USER` / `~`."
    )


def test_every_allowlist_entry_is_still_needed() -> None:
    """An allowlist that outlives its reason is how the next leak stays green.

    Each entry is an EXACT-COUNT pin PER SHAPE: when a config is partly or
    wholly scrubbed, this fails and names the number to correct, so the
    exemption cannot quietly become permission for a file that no longer needs
    it -- nor for a shape it never needed, nor for MORE hits than it had.

    ⚑ The over-budget direction is enforced by `test_no_tracked_file_leaks`;
    this is the under-budget half. Both are needed: an entry that is too LARGE
    is a hole, an entry that outlives its reason is clutter that the next
    reader trusts.
    """
    stale: list[str] = []
    for rel, shapes in sorted(_ALLOWED.items()):
        hits = _hits(REPO_ROOT / rel)
        for shape, (budget, _reason) in sorted(shapes.items()):
            n = sum(1 for sh, _, _ in hits if sh == shape)
            if n < budget:
                stale.append(f"{rel}[{shape}]: {n} hits remain, {budget} exempted")
    assert not stale, (
        f"{stale} — the exemption is now larger than the leak. Lower the count "
        "(or delete the entry) so the file is covered by the sweep again"
    )

    missing = [rel for rel in sorted(_ALLOWED) if not (REPO_ROOT / rel).is_file()]
    assert not missing, f"_ALLOWED names files that do not exist: {missing}"


def test_an_allowlisted_file_is_still_swept_for_other_shapes(tmp_path: Path) -> None:
    """The second structural gap, pinned.

    `configs/pbt2_small.yaml` is exempt for two shapes and must remain covered
    for the others, or the exemption is a licence rather than an exception.
    """
    exempt = frozenset(_ALLOWED["configs/pbt2_small.yaml"])
    assert frozenset(_PATTERNS) - exempt, (
        "pbt2_small.yaml is exempt from EVERY shape — it is no longer swept at all"
    )
    poisoned = tmp_path / "pbt2_small.yaml"
    poisoned.write_bytes(_CONTROLS["session_scratch"].encode())
    assert _hits(poisoned, frozenset(_PATTERNS) - exempt), (
        "a new leak of a non-exempt shape in an allowlisted file must still be found"
    )
    # ...including the shape added by the second review: an allowlisted config
    # that grows a machine name is not covered by its path/login exemptions.
    poisoned.write_bytes(_CONTROLS["hostname"].encode())
    assert _hits(poisoned, frozenset(_PATTERNS) - exempt), (
        "an allowlisted config gaining a real hostname must still be found"
    )


def test_one_more_hit_in_an_allowlisted_file_is_still_a_finding(tmp_path: Path) -> None:
    """⚑ The Codex finding, pinned: the exemption is a BUDGET, not an off switch.

    While `_ALLOWED` subtracted the whole detector, the 17 exempt configs — the
    files most likely to acquire a fresh machine-specific path — could gain any
    number of NEW absolute paths and stay green for as long as one old exempt
    occurrence survived. `test_every_allowlist_entry_is_still_needed` could not
    see it either: it only asked whether at least one hit of the shape remained.
    """
    rel = "configs/pbt2_small.yaml"
    budget, _reason = _ALLOWED[rel]["abs_home"]
    real = (REPO_ROOT / rel).read_bytes()
    fixture_root = tmp_path / "root"
    (fixture_root / "configs").mkdir(parents=True)
    poisoned = fixture_root / rel

    # Exactly the budget: green, or the sweep would fail on the real tree.
    poisoned.write_bytes(real)
    assert not _offenders([poisoned], root=fixture_root)

    # One more, and only one more: red.
    poisoned.write_bytes(real + b"\nsome_new_key: " + _CONTROLS["abs_home"].encode() + b"\n")
    found = _offenders([poisoned], root=fixture_root)
    assert found, (
        f"a {budget + 1}th abs_home hit in an allowlisted config was not a "
        "finding — the exemption is an off switch again"
    )
    assert f"{budget + 1} hits, {budget} allowed" in found[0], found


def test_a_tracked_symlink_is_scanned_as_its_TARGET_STRING(tmp_path: Path) -> None:
    """⚑ Codex finding: git stores the target verbatim, so the target IS the leak.

    `Path.read_bytes()` follows the link. If the target does not exist on the
    machine running the sweep — which is the whole point of a leaked absolute
    home path — the `OSError` branch returned no hits and the disclosure swept
    green. If it DOES exist, some unrelated file's contents were scanned
    instead, which is no better.
    """
    target = _CONTROLS["abs_home"]            # a path that does not exist here
    link = tmp_path / "some_link"
    link.symlink_to(target)
    assert not link.exists(), "fixture broken: the target must be absent"
    hits = _hits(link)
    assert [shape for shape, _, _ in hits] == ["abs_home"], hits
    assert target in hits[0][2]
