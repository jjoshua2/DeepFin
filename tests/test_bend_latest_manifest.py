"""latest.json must still resolve to a sha256-checked Bend tarball."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from native.bend_engine.resolve_latest import (
    BendLatestError,
    BendRelease,
    default_platform,
    fetch_github_bytes,
    resolve_latest,
)


HELPER = Path(__file__).resolve().parents[1] / "native" / "bend_engine" / "resolve_latest.py"
LINUX_X64_SHA = "2eb85cdeceff378bea67e20450347edbe4e0b327ecb18cebf0dc4ca27435cd1c"
LINUX_X64_URL = (
    "https://github.com/bendlang/bend/releases/download/v2.0.13/"
    "bend-2.0.13-linux-x64.tar.gz"
)
OLD_DL_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OLD_DL_URL = "https://bend-lang.com/dl/bend-2.0.4-linux-x64.tar.gz"

NEW_LATEST = {"ver": "2.0.13", "notice": ""}
OLD_LATEST = {
    "ver": "2.0.4",
    "sha256": OLD_DL_SHA,
    "url": OLD_DL_URL,
}
LINUX_ASSET = {
    "name": "bend-2.0.13-linux-x64.tar.gz",
    "digest": f"sha256:{LINUX_X64_SHA}",
    "browser_download_url": LINUX_X64_URL,
}
GITHUB_RELEASE = {
    "tag_name": "v2.0.13",
    "assets": [
        {
            "name": "bend-2.0.13-darwin-arm64.tar.gz",
            "digest": "sha256:310a5771134c6983d2a58b2a165915f0bf94cf916986e39bd7707df81a3738f6",
            "browser_download_url": (
                "https://github.com/bendlang/bend/releases/download/v2.0.13/"
                "bend-2.0.13-darwin-arm64.tar.gz"
            ),
        },
        LINUX_ASSET,
    ],
}


def _fetch_should_not_run(url: str) -> bytes:
    raise AssertionError(f"must not fetch GitHub for old latest.json: {url}")


def test_old_latest_json_uses_feed_sha_and_url() -> None:
    release = resolve_latest(
        OLD_LATEST,
        platform="linux-x64",
        fetch=_fetch_should_not_run,
    )
    assert release == BendRelease("2.0.4", OLD_DL_SHA, OLD_DL_URL)


def test_new_latest_json_uses_github_asset_digest() -> None:
    seen: list[str] = []

    def fetch(url: str) -> bytes:
        seen.append(url)
        return json.dumps(GITHUB_RELEASE).encode()

    release = resolve_latest(NEW_LATEST, platform="linux-x64", fetch=fetch)
    assert seen == [
        "https://api.github.com/repos/bendlang/bend/releases/tags/v2.0.13"
    ]
    assert release == BendRelease("2.0.13", LINUX_X64_SHA, LINUX_X64_URL)


def test_new_latest_json_without_github_fails() -> None:
    with pytest.raises(BendLatestError, match="GitHub release metadata required"):
        resolve_latest(NEW_LATEST, platform="linux-x64")


@pytest.mark.parametrize(("payload", "message"), [
    ({"notice": ""}, "invalid Bend version"),
    ({"ver": "2.0.13", "sha256": LINUX_X64_SHA}, "partial sha256/url"),
    ({"ver": "2.0.13", "url": LINUX_X64_URL}, "partial sha256/url"),
    ({"ver": "2.0.13", "sha256": "", "url": OLD_DL_URL}, "invalid Bend sha256"),
    ({"ver": "2.0.4", "sha256": OLD_DL_SHA, "url": "https://evil.example/bend.tgz"},
     "unexpected Bend tarball URL"),
    ({"ver": "2.0.4", "sha256": OLD_DL_SHA,
      "url": "https://bend-lang.com/dl/../secret"}, "unexpected Bend tarball URL"),
    ({"ver": "2.0.4", "sha256": "deadbeef", "url": OLD_DL_URL}, "invalid Bend sha256"),
    (["2.0.13"], "latest.json is not an object"),
])
def test_latest_json_rejects_partial_or_untrusted(
    payload: object, message: str
) -> None:
    with pytest.raises(BendLatestError, match=message):
        resolve_latest(payload, platform="linux-x64", github_release=GITHUB_RELEASE)


@pytest.mark.parametrize(("github", "message"), [
    ({"tag_name": "v2.0.12", "assets": GITHUB_RELEASE["assets"]},
     "does not match latest.json"),
    ({"tag_name": "v2.0.13", "assets": []}, "expected one GitHub asset"),
    ({"tag_name": "v2.0.13", "assets": [
        {**LINUX_ASSET, "digest": "md5:00"}
    ]}, "missing a sha256 digest"),
    ({"tag_name": "v2.0.13", "assets": [
        {**LINUX_ASSET, "digest": "sha256:abcd"}
    ]}, "invalid Bend sha256"),
    ({"tag_name": "v2.0.13", "assets": [
        {**LINUX_ASSET, "browser_download_url": "https://evil.example/bend.tgz"}
    ]}, "unexpected Bend tarball URL"),
    ({"tag_name": "v2.0.13"}, "GitHub release assets are missing"),
])
def test_github_release_is_fail_closed(github: object, message: str) -> None:
    with pytest.raises(BendLatestError, match=message):
        resolve_latest(NEW_LATEST, platform="linux-x64", github_release=github)


def test_fetch_github_bytes_refuses_non_allowlisted_url() -> None:
    with pytest.raises(BendLatestError, match="refusing to fetch unexpected URL"):
        fetch_github_bytes("https://example.com/latest.json")


def test_default_platform_linux_x64() -> None:
    assert default_platform("Linux", "x86_64") == "linux-x64"
    assert default_platform("Darwin", "arm64") == "darwin-arm64"
    with pytest.raises(BendLatestError, match="unsupported OS"):
        default_platform("Windows", "x86_64")


def test_cli_prints_old_and_new_shapes(tmp_path: Path) -> None:
    github = tmp_path / "release.json"
    github.write_text(json.dumps(GITHUB_RELEASE))
    new = subprocess.run(
        [sys.executable, str(HELPER), "--platform", "linux-x64",
         "--github-json", str(github)],
        input=json.dumps(NEW_LATEST),
        text=True,
        capture_output=True,
        check=False,
    )
    assert new.returncode == 0, new.stderr
    assert new.stdout.strip() == f"2.0.13 {LINUX_X64_SHA} {LINUX_X64_URL}"

    old = subprocess.run(
        [sys.executable, str(HELPER), "--platform", "linux-x64", "--no-fetch"],
        input=json.dumps(OLD_LATEST),
        text=True,
        capture_output=True,
        check=False,
    )
    assert old.returncode == 0, old.stderr
    assert old.stdout.strip() == f"2.0.4 {OLD_DL_SHA} {OLD_DL_URL}"

    missing = subprocess.run(
        [sys.executable, str(HELPER), "--platform", "linux-x64", "--no-fetch"],
        input=json.dumps(NEW_LATEST),
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "GitHub release metadata required" in missing.stderr
