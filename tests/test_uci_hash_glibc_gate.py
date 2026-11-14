"""The Hash-accounting RSS tests skip off glibc; on glibc they must RUN.

An inverted skip condition would skip them everywhere and keep CI green with
the RSS check never executed. (A separate file, so it stays out of the way of
edits to the RSS tests themselves.)
"""
from __future__ import annotations

import os
import platform
import sys

import tests.test_uci_hash_accounting as hash_tests


def test_the_glibc_skip_is_not_taken_on_glibc() -> None:
    marker = hash_tests._needs_glibc.mark
    assert marker.name == "skipif"
    (condition,) = marker.args
    on_glibc = platform.libc_ver()[0] == "glibc"
    assert condition is (not on_glibc)


def test_ci_on_linux_is_glibc_so_the_rss_tests_run() -> None:
    if not (os.environ.get("CI") and sys.platform.startswith("linux")):
        return
    assert platform.libc_ver()[0] == "glibc"
