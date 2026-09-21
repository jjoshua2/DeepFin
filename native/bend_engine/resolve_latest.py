#!/usr/bin/env python3
"""Resolve a sha256-checked Bend tarball from latest.json plus GitHub metadata.

The moving feed at https://bend-lang.com/dl/latest.json used to carry
``{ver, sha256, url}``. It now publishes ``{ver, notice}`` only. Either shape
must still yield a 64-hex digest and an allowlisted URL; missing checksums
are an error, never a skip.
"""
from __future__ import annotations

import argparse
import json
import os
import platform as py_platform
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GITHUB_RELEASE_API = (
    "https://api.github.com/repos/bendlang/bend/releases/tags/v{ver}"
)
GITHUB_TARBALL = (
    "https://github.com/bendlang/bend/releases/download/v{ver}/"
    "bend-{ver}-{platform}.tar.gz"
)
BEND_DL_PREFIX = "https://bend-lang.com/dl/"
PLATFORMS = frozenset(
    {"linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64"}
)
VERSION_RE = re.compile(r"^[0-9A-Za-z._-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
Fetch = Callable[[str], bytes]


class BendLatestError(ValueError):
    """Fail-closed latest.json / GitHub release resolution."""


class BendReleaseFetchError(BendLatestError):
    """Release API unavailable; never used for malformed integrity metadata."""


@dataclass(frozen=True)
class BendRelease:
    version: str
    sha256: str
    url: str


def default_platform(
    system: str | None = None, machine: str | None = None
) -> str:
    sysname = (system if system is not None else py_platform.system()).lower()
    mach = (machine if machine is not None else py_platform.machine()).lower()
    if sysname == "linux":
        os_name = "linux"
    elif sysname in {"darwin", "macos"}:
        os_name = "darwin"
    else:
        raise BendLatestError(f"unsupported OS for Bend tarball: {sysname}")
    if mach in {"x86_64", "amd64"}:
        arch = "x64"
    elif mach in {"aarch64", "arm64"}:
        arch = "arm64"
    else:
        raise BendLatestError(f"unsupported CPU for Bend tarball: {mach}")
    return f"{os_name}-{arch}"


def _require_version(value: object) -> str:
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
        raise BendLatestError(f"invalid Bend version: {value!r}")
    if value in {".", ".."}:
        raise BendLatestError(f"invalid Bend version: {value!r}")
    return value


def _require_sha(value: object, *, source: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise BendLatestError(f"invalid Bend sha256 from {source}")
    return value


def _is_allowed_bend_dl(url: str) -> bool:
    if not url.startswith(BEND_DL_PREFIX) or "?" in url or "#" in url:
        return False
    rest = url[len(BEND_DL_PREFIX):]
    if not rest or rest.startswith("/") or ".." in rest.split("/"):
        return False
    return all(ch.isalnum() or ch in "._-/" for ch in rest)


def _require_url(value: object, *, version: str, platform: str) -> str:
    if not isinstance(value, str) or not value:
        raise BendLatestError("missing Bend tarball URL")
    expected = GITHUB_TARBALL.format(ver=version, platform=platform)
    if value == expected or _is_allowed_bend_dl(value):
        return value
    raise BendLatestError(f"unexpected Bend tarball URL: {value}")


def _require_object(value: object, *, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BendLatestError(f"{what} is not an object")
    return value


def _github_release_url(version: str) -> str:
    return GITHUB_RELEASE_API.format(ver=version)


def fetch_github_bytes(url: str) -> bytes:
    """HTTPS GET of the allowlisted GitHub release API URL only."""
    if not url.startswith(
        "https://api.github.com/repos/bendlang/bend/releases/tags/v"
    ):
        raise BendLatestError(f"refusing to fetch unexpected URL: {url}")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "deepfin-bend-probe",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BendReleaseFetchError(f"GitHub release fetch failed at {url}: {exc}") from exc


def fetch_installer_bytes() -> bytes:
    """Read official metadata as data only; never execute a downloaded script."""
    request = urllib.request.Request(
        "https://bend-lang.com/install.sh",
        headers={"User-Agent": "deepfin-bend-probe"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BendLatestError(f"official installer metadata fetch failed: {exc}") from exc
    if len(data) > 1024 * 1024:
        raise BendLatestError("official installer metadata is oversized")
    return data


def release_from_installer(version: str, platform: str, raw: bytes) -> BendRelease:
    """Extract unique literal version/platform checksums from the official script.

    The published installer pins all platform archive digests independently of the
    GitHub API. Match the feed version exactly; a mismatched publication is an error.
    No shell interpolation, URL from script code, or checksum-free fallback.
    """
    if platform not in PLATFORMS:
        raise BendLatestError(f"unsupported Bend platform: {platform}")
    version = _require_version(version)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BendLatestError("official installer metadata is not UTF-8") from exc

    def literal(key: str) -> str:
        assignments = re.findall(rf"(?m)^\s*{key}=([^\r\n]*)$", text)
        if len(assignments) != 1:
            raise BendLatestError(f"expected one official installer {key} assignment")
        match = re.fullmatch(r'"([^"\r\n]*)"', assignments[0])
        if match is None:
            raise BendLatestError(f"official installer {key} must be a quoted literal")
        return match[1]

    if literal("REPO") != "bendlang/bend":
        raise BendLatestError("unexpected official installer repository")
    if literal("VER") != version:
        raise BendLatestError("official installer version does not match latest.json")
    key = "SHA_" + platform.replace("-", "_").upper()
    sha = _require_sha(literal(key), source="official installer")
    return BendRelease(version, sha, GITHUB_TARBALL.format(ver=version, platform=platform))


def load_github_release(version: str, fetch: Fetch) -> Mapping[str, Any]:
    raw = fetch(_github_release_url(version))
    try:
        payload: object = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BendLatestError("GitHub release metadata is not JSON") from exc
    return _require_object(payload, what="GitHub release metadata")


def release_from_github(
    version: str, platform: str, github_release: Mapping[str, Any]
) -> BendRelease:
    if platform not in PLATFORMS:
        raise BendLatestError(f"unsupported Bend platform: {platform}")
    tag = github_release.get("tag_name")
    if tag != f"v{version}":
        raise BendLatestError(
            f"GitHub tag {tag!r} does not match latest.json ver {version!r}"
        )
    assets = github_release.get("assets")
    if not isinstance(assets, list):
        raise BendLatestError("GitHub release assets are missing")
    name = f"bend-{version}-{platform}.tar.gz"
    matches = [
        asset for asset in assets
        if isinstance(asset, Mapping) and asset.get("name") == name
    ]
    if len(matches) != 1:
        raise BendLatestError(
            f"expected one GitHub asset named {name}, got {len(matches)}"
        )
    digest = matches[0].get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise BendLatestError(f"GitHub asset {name} is missing a sha256 digest")
    sha256 = _require_sha(digest[len("sha256:"):], source="GitHub asset digest")
    url = _require_url(
        matches[0].get("browser_download_url"),
        version=version,
        platform=platform,
    )
    return BendRelease(version=version, sha256=sha256, url=url)


def resolve_latest(
    latest: object,
    *,
    platform: str,
    github_release: object | None = None,
    fetch: Fetch | None = None,
    installer_fetch: Callable[[], bytes] | None = None,
) -> BendRelease:
    """Return version/sha256/url. Never succeeds without a 64-hex digest."""
    if platform not in PLATFORMS:
        raise BendLatestError(f"unsupported Bend platform: {platform}")
    payload = _require_object(latest, what="latest.json")
    version = _require_version(payload.get("ver"))
    has_sha = "sha256" in payload
    has_url = "url" in payload
    if has_sha or has_url:
        if not (has_sha and has_url):
            raise BendLatestError(
                "latest.json has partial sha256/url; both are required"
            )
        sha256 = _require_sha(payload.get("sha256"), source="latest.json")
        url = _require_url(
            payload.get("url"), version=version, platform=platform
        )
        return BendRelease(version=version, sha256=sha256, url=url)
    github_payload: Mapping[str, Any]
    if github_release is not None:
        github_payload = _require_object(
            github_release, what="GitHub release metadata"
        )
    elif fetch is not None:
        try:
            github_payload = load_github_release(version, fetch)
        except BendReleaseFetchError as exc:
            if installer_fetch is None:
                raise
            print(f"bend probe: {exc}; checking official installer metadata", file=sys.stderr)
            return release_from_installer(version, platform, installer_fetch())
    else:
        raise BendLatestError(
            "latest.json has no sha256/url; GitHub release metadata required"
        )
    return release_from_github(version, platform, github_payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        default="",
        help="linux-x64 / linux-arm64 / darwin-x64 / darwin-arm64 "
        "(default: this host)",
    )
    parser.add_argument(
        "--github-json",
        type=Path,
        help="pre-fetched GitHub release JSON; skips the network",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="do not call the GitHub API; fail if sha256/url are absent",
    )
    args = parser.parse_args(argv)
    try:
        latest = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise BendLatestError(f"latest.json is not JSON: {exc}") from exc
    platform = args.platform or default_platform()
    github_release: object | None = None
    if args.github_json is not None:
        github_release = json.loads(args.github_json.read_text())
    fetch = None
    if github_release is None and not args.no_fetch:
        fetch = fetch_github_bytes
    release = resolve_latest(
        latest,
        platform=platform,
        github_release=github_release,
        fetch=fetch,
        installer_fetch=fetch_installer_bytes if fetch is not None else None,
    )
    print(release.version, release.sha256, release.url)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BendLatestError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
