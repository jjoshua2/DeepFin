"""The public metadata fallback retains exact version/platform/checksum checks."""
from __future__ import annotations

import json

import pytest

from native.bend_engine.resolve_latest import (
    BendLatestError,
    BendReleaseFetchError,
    release_from_installer,
    resolve_latest,
)

SCRIPT = '\n'.join([
    'REPO="bendlang/bend"', 'VER="2.0.25"',
    *[f'SHA_{platform}="{digit * 64}"' for platform, digit in
      [('LINUX_X64', 'a'), ('LINUX_ARM64', 'b'), ('DARWIN_X64', 'c'), ('DARWIN_ARM64', 'd')]],
])


def unavailable(url: str) -> bytes:
    raise BendReleaseFetchError(f'404 at {url}')


@pytest.mark.parametrize(('platform', 'digit'), [
    ('linux-x64', 'a'), ('linux-arm64', 'b'), ('darwin-x64', 'c'), ('darwin-arm64', 'd'),
])
def test_unavailable_api_uses_same_version_official_checksum(platform: str, digit: str) -> None:
    release = resolve_latest({'ver': '2.0.25'}, platform=platform,
                             fetch=unavailable, installer_fetch=SCRIPT.encode)
    assert release.version == '2.0.25'
    assert release.sha256 == digit * 64
    assert release.url == f'https://github.com/bendlang/bend/releases/download/v2.0.25/bend-2.0.25-{platform}.tar.gz'


@pytest.mark.parametrize('script', [
    SCRIPT.replace('2.0.25', '2.0.24'),
    SCRIPT.replace('bendlang/bend', 'untrusted/bend'),
    SCRIPT.replace('a' * 64, 'xyz'),
    SCRIPT.replace('SHA_LINUX_X64', 'SHA_UNKNOWN'),
    SCRIPT + '\nVER="2.0.25"',
    SCRIPT + '\nSHA_LINUX_X64="' + 'a' * 64 + '"',
    SCRIPT.replace('VER="2.0.25"', 'VER=$(echo 2.0.25)'),
    SCRIPT.replace('VER="2.0.25"', 'VER="2.0.25"; echo unrelated'),
])
def test_invalid_official_metadata_still_fails(script: str) -> None:
    with pytest.raises(BendLatestError):
        resolve_latest({'ver': '2.0.25'}, platform='linux-x64',
                       fetch=unavailable, installer_fetch=script.encode)


def test_failed_api_without_fallback_stays_failed() -> None:
    with pytest.raises(BendReleaseFetchError):
        resolve_latest({'ver': '2.0.25'}, platform='linux-x64', fetch=unavailable)


@pytest.mark.parametrize('raw', [b'not json', json.dumps({'tag_name': 'v2.0.25', 'assets': []}).encode()])
def test_invalid_successful_api_response_cannot_be_hidden(raw: bytes) -> None:
    def unexpected() -> bytes:
        raise AssertionError('must not bypass malformed API integrity metadata')
    with pytest.raises(BendLatestError):
        resolve_latest({'ver': '2.0.25'}, platform='linux-x64',
                       fetch=lambda _: raw, installer_fetch=unexpected)


def test_source_is_data_not_executed() -> None:
    # Shell code, including commands with side effects, is never evaluated.
    script = SCRIPT + '\nexit 99\n$(definitely_not_a_program)\n'
    assert release_from_installer('2.0.25', 'linux-x64', script.encode()).sha256 == 'a' * 64


def test_non_utf8_metadata_fails() -> None:
    with pytest.raises(BendLatestError, match='UTF-8'):
        release_from_installer('2.0.25', 'linux-x64', b'\xff')
