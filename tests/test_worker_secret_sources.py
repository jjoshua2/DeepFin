"""A credential comes from the environment or an untracked file, never a config.

The `distributed_worker_password` line in `configs/pbt2_small.yaml` is a
plaintext secret in a tracked file in a PUBLIC repository, next to
`distributed_server_host: 0.0.0.0`. Rotating it is a separate, user-gated step;
these tests pin the plumbing that makes a rotation land somewhere safe rather
than straight back into the same public file.

⚑ THE REFUSAL TESTS ARE THE POINT. A missing secret must stop the server, not
produce an empty password — an account provisioned with a blank secret on a
`0.0.0.0` listener is a server that is open, not one that is broken, and
"accepted then silently ignored" is this codebase's signature defect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_anti_engine.server.secrets import (
    WORKER_PASSWORD_ENV,
    WORKER_PASSWORD_FILE_ENV,
    InsecureWorkerSecret,
    MissingWorkerSecret,
    default_worker_password_file,
    refuse_config_password,
    resolve_worker_password,
)


@pytest.fixture(autouse=True)
def _no_ambient_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Every test starts with no credential anywhere.

    Without this the suite would pass on a developer box that happens to export
    the variable — a gate that cannot fail on the machine that runs it.
    """
    monkeypatch.delenv(WORKER_PASSWORD_ENV, raising=False)
    monkeypatch.delenv(WORKER_PASSWORD_FILE_ENV, raising=False)
    monkeypatch.setenv(WORKER_PASSWORD_FILE_ENV, str(tmp_path / "absent" / "worker_password"))


def _secret_file(tmp_path: Path, value: str, *, mode: int = 0o600) -> Path:
    path = tmp_path / "worker_password"
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def test_an_absent_secret_refuses_rather_than_returning_empty() -> None:
    """THE REFUSAL PATH. No source, no credential, no startup."""
    with pytest.raises(MissingWorkerSecret) as excinfo:
        resolve_worker_password({})
    message = str(excinfo.value)
    assert WORKER_PASSWORD_ENV in message, "the error must name the fix"
    assert "will not start" in message


def test_a_yaml_password_is_not_a_source() -> None:
    """⚑ THE CENTREPIECE. A plaintext yaml password must NOT authenticate.

    This is the test that would have gone red on the old code path, where
    `_prepare_distributed_worker_auth` read `distributed_worker_password`
    straight out of the config. A rotation done in the yaml has to fail loudly,
    or the next secret lands in the public repo exactly like the last one.
    """
    config = {"distributed_worker_password": "a-plaintext-secret-in-a-public-repo"}
    with pytest.raises(MissingWorkerSecret):
        resolve_worker_password(config)


def test_the_yaml_password_is_ignored_but_never_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inert AND audible. An ignored key with no observation is how the next
    operator concludes it still works."""
    monkeypatch.setenv(WORKER_PASSWORD_ENV, "from-the-environment")
    config = {"distributed_worker_password": "burned"}

    secret, source = resolve_worker_password(config)

    assert secret == "from-the-environment"
    assert source == f"${WORKER_PASSWORD_ENV}"
    nag = refuse_config_password(config)
    assert nag is not None
    assert "PUBLIC REPOSITORY" in nag
    assert "IGNORED" in nag
    assert refuse_config_password({}) is None
    assert refuse_config_password({"distributed_worker_password": ""}) is None


def test_an_explicitly_configured_name_beats_the_ambient_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ EXPLICIT CONFIG > AMBIENT ENVIRONMENT, and this test is the pin.

    The first version of this module read `$CAE_WORKER_PASSWORD` first, so an
    operator who had configured `distributed_worker_password_env: MY_SECRET`
    got silently authenticated with whatever the launching shell happened to
    export — a wrong-credential generator that reports success. The order is
    now: what the config names, then the ambient convenience variable.
    """
    monkeypatch.setenv(WORKER_PASSWORD_ENV, "ambient")
    monkeypatch.setenv("SOME_OTHER_NAME", "explicitly-configured")
    secret, source = resolve_worker_password(
        {"distributed_worker_password_env": "SOME_OTHER_NAME"}
    )
    assert secret == "explicitly-configured"
    assert "SOME_OTHER_NAME" in source


def test_the_ambient_var_is_used_when_the_config_names_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-line operator step still works for everyone not configuring a name."""
    monkeypatch.setenv(WORKER_PASSWORD_ENV, "ambient")
    secret, source = resolve_worker_password({})
    assert secret == "ambient"
    assert source == f"${WORKER_PASSWORD_ENV}"


def test_a_named_env_var_is_used_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_WORKER_SECRET", "named-secret")
    secret, source = resolve_worker_password(
        {"distributed_worker_password_env": "MY_WORKER_SECRET"}
    )
    assert secret == "named-secret"
    assert "MY_WORKER_SECRET" in source


def test_a_named_but_empty_env_var_refuses_instead_of_falling_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The operator said WHERE the secret lives. Reading a different source
    behind their back is how a run "succeeds" with the wrong credential."""
    path = _secret_file(tmp_path, "file-secret")
    monkeypatch.setenv(WORKER_PASSWORD_FILE_ENV, str(path))
    monkeypatch.delenv("MY_WORKER_SECRET", raising=False)

    with pytest.raises(MissingWorkerSecret) as excinfo:
        resolve_worker_password({"distributed_worker_password_env": "MY_WORKER_SECRET"})
    assert "MY_WORKER_SECRET" in str(excinfo.value)


def test_a_secret_file_is_read_and_stripped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = _secret_file(tmp_path, "  file-secret  ")
    monkeypatch.setenv(WORKER_PASSWORD_FILE_ENV, str(path))
    secret, source = resolve_worker_password({})
    assert secret == "file-secret"
    assert source == str(path)


def test_a_world_readable_secret_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A credential every account on the box can read is not meaningfully
    different from the yaml this change exists to get it out of."""
    path = _secret_file(tmp_path, "file-secret", mode=0o644)
    monkeypatch.setenv(WORKER_PASSWORD_FILE_ENV, str(path))
    with pytest.raises(InsecureWorkerSecret) as excinfo:
        resolve_worker_password({})
    assert "chmod 600" in str(excinfo.value)


def test_an_empty_secret_file_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = _secret_file(tmp_path, "")
    monkeypatch.setenv(WORKER_PASSWORD_FILE_ENV, str(path))
    with pytest.raises(MissingWorkerSecret):
        resolve_worker_password({})


def test_the_default_secrets_path_is_gitignored() -> None:
    """The documented default must actually be untracked.

    A documented path that git would happily commit is worse than no default:
    it reads as safe. Asserted against the shipped `.gitignore` rules rather
    than by running git, so this holds in a source checkout with no git dir.
    """
    repo_root = Path(__file__).resolve().parents[1]
    rules = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".secrets/" in rules
    assert "*.password" in rules
    default = default_worker_password_file()
    assert default.parent.name == ".secrets"
    assert default.is_relative_to(repo_root)


def test_no_tracked_file_carries_the_resolver_as_a_config_read() -> None:
    """The plumbing must not grow a fourth, quiet fallback.

    Pins the ABSENCE of a `config.get("distributed_worker_password")` read
    anywhere that could turn it back into a credential: `secrets.py` reads it
    only inside `refuse_config_password`, which returns a warning string and
    never a secret.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "chess_anti_engine" / "tune" / "harness.py"
    ).read_text(encoding="utf-8")
    assert 'cfg.get("distributed_worker_password"' not in src
    assert 'base_config.get("distributed_worker_password"' not in src
    assert "resolve_worker_password(cfg)" in src


def test_the_real_provisioning_path_refuses_without_a_secret(tmp_path: Path) -> None:
    """END TO END on `_prepare_distributed_worker_auth`, the function the
    driver actually calls, because a resolver nothing calls is the defect.

    Old behaviour on this exact input: the yaml password provisioned the
    account and wrote the `.password` file. New behaviour: refuse.
    """
    from chess_anti_engine.tune.harness import _prepare_distributed_worker_auth

    server_root = tmp_path / "server"
    server_root.mkdir()
    config = {
        "distributed_worker_username": "josh",
        "distributed_worker_password": "the-public-yaml-secret",
    }
    with pytest.raises(MissingWorkerSecret):
        _prepare_distributed_worker_auth(server_root=server_root, config=config)

    assert not (server_root / "users.json").exists(), "no account may be created"
    assert not list(server_root.glob("*.password")), "no credential file may be written"


def test_the_real_provisioning_path_uses_the_environment_and_writes_0600(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The one-line operator step, end to end, plus the file mode.

    The mode assertion is not decoration: the old code wrote the file and THEN
    chmod'ed it, leaving a window at the prevailing umask where any account on
    the box could read the credential.
    """
    from chess_anti_engine.server.auth import load_users, verify_password
    from chess_anti_engine.tune.harness import _prepare_distributed_worker_auth

    monkeypatch.setenv(WORKER_PASSWORD_ENV, "rotated-secret")
    server_root = tmp_path / "server"
    server_root.mkdir()

    username, password_file = _prepare_distributed_worker_auth(
        server_root=server_root,
        config={"distributed_worker_username": "josh",
                "distributed_worker_password": "the-public-yaml-secret"},
    )

    assert username == "josh"
    assert password_file.read_text(encoding="utf-8").strip() == "rotated-secret"
    assert oct(password_file.stat().st_mode & 0o777) == "0o600"

    # users.json carries the KDF material and no plaintext, of either secret.
    raw = (server_root / "users.json").read_text(encoding="utf-8")
    assert "rotated-secret" not in raw
    assert "the-public-yaml-secret" not in raw
    record = load_users(server_root / "users.json")["josh"]
    assert verify_password("rotated-secret", record)
    assert not verify_password("the-public-yaml-secret", record), (
        "the yaml value must never have become the credential"
    )
    assert not verify_password("", record), "an empty password must not authenticate"


def test_no_tracked_config_carries_a_plaintext_password() -> None:
    """⚑ THE REGRESSION GATE FOR THE WHOLE CHANGE.

    Everything else here is plumbing; this is the assertion that the plaintext
    is actually gone and cannot come back in a later config edit. It sweeps
    every tracked yaml, not just the production one — the burned secret was in
    16 of them, and stripping only `pbt2_small.yaml` would have left 15 copies
    in a public repository while every other test in this file passed.

    ⚑ THE FILE SET COMES FROM `git ls-files`, not a `configs/**` glob. What
    makes the value disclosed is that it is TRACKED, so the scope of the sweep
    has to be exactly that: a glob under `configs/` scanned 18 of the 20 tracked
    yamls and would have gone green on a secret pasted into any yaml outside
    that directory — a docker-compose file, a CI workflow, an example.

    `null` is allowed: it is a schema placeholder, not a secret.
    """
    import re
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    listed = subprocess.run(
        ["git", "ls-files", "-z", "*.yaml", "*.yml"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout
    tracked = sorted(repo_root / name for name in listed.split("\0") if name)
    assert tracked, "git ls-files returned no yaml at all — the sweep is vacuous"
    offenders: list[str] = []
    for path in tracked:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.match(r"\s*distributed_worker_password:\s*(\S.*)$", line)
            if match and match.group(1).strip() not in ("null", "~"):
                offenders.append(f"{path.relative_to(repo_root)}:{lineno}")
    assert not offenders, (
        "these TRACKED configs carry a plaintext worker password, in a PUBLIC "
        f"repository: {offenders}. Move the secret to $"
        f"{WORKER_PASSWORD_ENV} or .secrets/worker_password and delete the line "
        "— and treat the value as disclosed, because it is in the git history."
    )


def test_the_config_key_stays_in_the_schema_allowlist() -> None:
    """Deleting the KEY would be the more dangerous change, so pin that it stays.

    The live-yaml validator is ALL-OR-NOTHING: one unknown key rejects the
    entire reload. Removing `distributed_worker_password` from the allowlist
    would mean a running trial that reloads any yaml still carrying the line
    silently stops applying EVERY live change — a bigger and quieter outage
    than the disclosure. The key stays parseable; its value stays inert.
    """
    from chess_anti_engine.utils.config_yaml import _TUNE_KEYS

    assert "distributed_worker_password" in _TUNE_KEYS
    assert "distributed_worker_password_env" in _TUNE_KEYS


def test_the_credential_source_is_announced_and_names_the_explicit_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """⚑ THE PRECEDENCE IS ONLY SAFE IF THE CHOICE IS VISIBLE.

    With two candidate sources present, "which one did it use" must not require
    reading the source. The log line is the operator's only instrument for
    that, so it is asserted rather than assumed — an unasserted print is a
    feature that can be deleted by accident.
    """
    from chess_anti_engine.tune.harness import _prepare_distributed_worker_auth

    monkeypatch.setenv(WORKER_PASSWORD_ENV, "ambient-secret")
    monkeypatch.setenv("MY_WORKER_SECRET", "explicit-secret")
    server_root = tmp_path / "server"
    server_root.mkdir()
    _prepare_distributed_worker_auth(
        server_root=server_root,
        config={
            "distributed_worker_username": "josh",
            "distributed_worker_password_env": "MY_WORKER_SECRET",
        },
    )
    out = capsys.readouterr().out
    assert "worker credential source: $MY_WORKER_SECRET" in out
    assert f"${WORKER_PASSWORD_ENV}" not in out


def test_a_weak_boot_credential_warns_but_still_boots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The deliberate asymmetry: the CLI refuses a short password, the BOOT
    path only warns.

    Refusing here would mean a minimum-length policy that can stop production
    from starting on a credential that already works. Pinned as behaviour so
    the asymmetry is a decision, not a hole someone later "fixes" into an
    outage.
    """
    from chess_anti_engine.tune.harness import _prepare_distributed_worker_auth

    monkeypatch.setenv(WORKER_PASSWORD_ENV, "short7x")
    server_root = tmp_path / "server"
    server_root.mkdir()
    _prepare_distributed_worker_auth(
        server_root=server_root,
        config={"distributed_worker_username": "josh"},
    )
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "weak" in out
    assert (server_root / "users.json").exists(), "the run must still boot"


def test_users_json_is_0600_regardless_of_the_umask(tmp_path: Path) -> None:
    """⚑ A WORLD-WRITABLE users.json IS AUTH BYPASS, not disclosure.

    The file holds PBKDF2 material for every account. Readable is bad; WRITABLE
    is worse — anyone who can write it replaces a hash with one of their own
    and authenticates as that user. `atomic_write_text` inherited the process
    umask, so at `umask 000` this file was created 0666.

    The umask is set to 0 deliberately: at the default 022 the file comes out
    0644 either way and the test would pass without the fix, which is a gate
    that cannot fail.
    """
    import os
    import stat

    from chess_anti_engine.server.auth import ensure_user

    saved = os.umask(0)
    try:
        db = tmp_path / "users.json"
        ensure_user(db, username="josh", password="a-good-password")
        mode = stat.S_IMODE(db.stat().st_mode)
    finally:
        os.umask(saved)
    assert mode == 0o600, f"users.json is {mode:04o}, not 0600"
