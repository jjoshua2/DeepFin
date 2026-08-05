"""Where a worker credential may come from, and where it may not.

⚑ THE YAML IS A TRACKED FILE IN A PUBLIC REPOSITORY. `distributed_worker_password`
sat there in plaintext next to `distributed_server_host: 0.0.0.0`, so the secret
is disclosed to anyone who has ever cloned the repo and stays disclosed in the
git history after any edit. Rotating it is necessary and is tracked separately
(task #161, user-gated); this module is the plumbing that makes a rotation
*land somewhere safe*, so the next secret is not written straight back into the
same public file.

The rule this module exists to enforce: **a secret comes from the process
environment or from an untracked file, and a config value is never a secret.**

Resolution order, first hit wins:

1. ``CAE_WORKER_PASSWORD`` — the one-line operator step. Set it in the shell
   that launches training and nothing else has to change.
2. the environment variable *named* by ``distributed_worker_password_env`` —
   for operators who already keep it under another name.
3. ``CAE_WORKER_PASSWORD_FILE``, or the default untracked path
   ``<repo>/.secrets/worker_password`` — a file mode-checked and gitignored.

``distributed_worker_password`` in the yaml is **deliberately not on that
list**. It is still ACCEPTED by the schema — see `refuse_config_password` for
why removing it would be the more dangerous change — but its value can never
become a credential again.

⚑ AND THERE IS NO FOURTH FALLBACK. When every source is empty this module
RAISES. The failure this codebase is worst at is a value accepted and then
silently ignored, and the version of it here is the expensive one: an empty
password that authenticates nobody would be bad, but an empty password that a
`verify_password("")` path accepts, or a provisioning step that creates an
account with a blank secret on a `0.0.0.0` listener, is a server that is open
rather than merely broken. Refusing to serve is the safe direction, so absence
is a hard error with an actionable message, never a default.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

# The env var an operator sets. Named here rather than inlined so the error
# message, the docs and the resolver cannot drift apart.
WORKER_PASSWORD_ENV = "CAE_WORKER_PASSWORD"
WORKER_PASSWORD_FILE_ENV = "CAE_WORKER_PASSWORD_FILE"

# Untracked by `.gitignore`. Under the repo so an operator can find it without
# a second piece of knowledge, and outside every directory the packaging and
# the shard pipeline walk.
DEFAULT_SECRETS_DIRNAME = ".secrets"
DEFAULT_WORKER_PASSWORD_FILENAME = "worker_password"


class MissingWorkerSecret(RuntimeError):
    """No credential source held a secret. Never caught into a default."""


class InsecureWorkerSecret(RuntimeError):
    """A secret was found somewhere it must not be, or readable by others."""


def default_secrets_dir(repo_root: Path | None = None) -> Path:
    """``<repo>/.secrets``, derived from this file's location by default."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    return Path(root) / DEFAULT_SECRETS_DIRNAME


def default_worker_password_file(repo_root: Path | None = None) -> Path:
    return default_secrets_dir(repo_root) / DEFAULT_WORKER_PASSWORD_FILENAME


def _read_secret_file(path: Path) -> str:
    """Read a secret file, refusing one that other users can read.

    The mode check is not ceremony. A secrets file created by a careless
    redirect inherits the umask and is frequently world-readable, and a
    credential that every account on the box can read is not meaningfully
    different from the yaml this module exists to get it out of. Group and
    other bits are the check; owner bits are not our business.
    """
    try:
        st = path.stat()
    except OSError as exc:
        raise MissingWorkerSecret(
            f"worker password file {path} cannot be read: {exc}"
        ) from exc
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise InsecureWorkerSecret(
            f"worker password file {path} is accessible to group/other "
            f"(mode {stat.filemode(st.st_mode)}). Run: chmod 600 {path}"
        )
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise MissingWorkerSecret(
            f"worker password file {path} cannot be read: {exc}"
        ) from exc


def resolve_worker_password(
    config: dict | None = None, *, repo_root: Path | None = None,
) -> tuple[str, str]:
    """The worker password and the human-readable source it came from.

    Raises `MissingWorkerSecret` when nothing holds one. There is deliberately
    no ``default=`` parameter: a caller that wants to continue without a
    credential is asking for the defect this module prevents, and should say so
    at its own call site where a reviewer can see it.
    """
    cfg = config or {}

    direct = str(os.environ.get(WORKER_PASSWORD_ENV, "") or "").strip()
    if direct:
        return direct, f"${WORKER_PASSWORD_ENV}"

    named = str(cfg.get("distributed_worker_password_env", "") or "").strip()
    if named:
        value = str(os.environ.get(named, "") or "").strip()
        if value:
            return value, f"${named} (via distributed_worker_password_env)"
  # A named-but-empty env var is its own error rather than a fall-through to
  # the file: the operator said WHERE the secret lives, and silently reading a
  # different source would be the accepted-then-ignored shape again.
        raise MissingWorkerSecret(
            f"distributed_worker_password_env names {named!r} but that "
            f"environment variable is empty or unset in this process. Export "
            f"it in the shell that launches training, or unset the config key "
            f"to fall back to ${WORKER_PASSWORD_ENV}."
        )

    file_raw = str(os.environ.get(WORKER_PASSWORD_FILE_ENV, "") or "").strip()
    path = Path(file_raw).expanduser() if file_raw else default_worker_password_file(repo_root)
    if path.exists():
        value = _read_secret_file(path)
        if value:
            return value, str(path)
        raise MissingWorkerSecret(f"worker password file {path} is empty")

    raise MissingWorkerSecret(
        "no worker credential is available, and this server will not start "
        "without one.\n"
        f"  Set it for this run:      export {WORKER_PASSWORD_ENV}='<secret>'\n"
        f"  or keep it in a file:     mkdir -p {default_secrets_dir(repo_root)} && "
        f"printf '%s\\n' '<secret>' > {default_worker_password_file(repo_root)} && "
        f"chmod 600 {default_worker_password_file(repo_root)}\n"
        f"  or point at your own:     export {WORKER_PASSWORD_FILE_ENV}=/path/to/file\n"
        "The yaml key distributed_worker_password is NOT a source: it is a "
        "tracked file in a public repository."
    )


def refuse_config_password(config: dict | None) -> str | None:
    """Warning text if the yaml still carries a plaintext password, else None.

    ⚑ WHY THIS WARNS RATHER THAN RAISES, and why the key is still in the
    schema allowlist. The live-yaml validator is ALL-OR-NOTHING: one unknown
    key rejects the entire reload, so deleting `distributed_worker_password`
    from the allowlist would make the running trial stop applying *every* live
    yaml change the moment it reloaded a file that still had the line -- a much
    larger outage than the disclosure, and one that arrives silently.

    So the key stays parseable and its value stays inert. `resolve_worker_password`
    never reads it, so no rotation can put a live secret back into the public
    file; this function is what makes the dead value AUDIBLE instead of merely
    unused, because an ignored key with no observation is how the next operator
    concludes it still works.

    The caller decides the severity. `_prepare_distributed_worker_auth` prints
    it; a startup that has NO other source raises separately, from
    `resolve_worker_password`, so "yaml has a password" can never be mistaken
    for "a credential is available".
    """
    value = str((config or {}).get("distributed_worker_password", "") or "").strip()
    if not value:
        return None
    return (
        "SECURITY: distributed_worker_password is set in the yaml, which is a "
        "TRACKED FILE IN A PUBLIC REPOSITORY. Its value is IGNORED -- the "
        f"credential comes from ${WORKER_PASSWORD_ENV} or the secrets file. "
        "Treat the yaml value as DISCLOSED (it is in the git history "
        "regardless of edits), rotate it at the next restart window, and "
        "delete the line."
    )
