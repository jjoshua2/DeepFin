"""Out-of-band observer: prove a script's ``DiskReplayBuffer`` flags took effect.

Injected on ``PYTHONPATH`` so the observed script is not modified::

    PYTHONPATH=.:scripts/probe_site python3 scripts/retarget_retrain.py ...

Prints one ``[probe_site]`` line per buffer construction (the flag it was built
with and whether it got a prefetch thread) and one at exit. Set
``PROBE_FORCE_RACE=1`` to force ``deterministic_refresh`` back OFF after
construction: that is the counterfactual arm, and it is what makes the
observation evidence rather than a check that cannot fail — a probe that can
only ever print ``True`` proves nothing about the script.

**Why ``sitecustomize`` and not ``usercustomize``.** This started life as
``usercustomize.py``, which ``site.py`` imports only when
``site.ENABLE_USER_SITE`` is true. It is true for ``/usr/bin/python3`` and
FALSE inside this project's ``.venv`` (venvs are created with
``include-system-site-packages = false``, which clears the user-site flag), so
under the interpreter the docs tell you to use, the probe never loaded and
printed nothing at all — and "no ``[probe_site]`` line" was indistinguishable
from "no buffer was constructed", an absence that reads as evidence.
``sitecustomize`` is imported unconditionally by ``site.py``, before the
``ENABLE_USER_SITE`` branch, so it loads under both. Verified by
``tests/test_probe_site_observer.py``, which runs it under
``sys.executable`` — i.e. under whatever interpreter the suite is using.

Belt and braces, because a probe's silence must never be readable as a
negative: the banner below prints at IMPORT time, before anything can fail, so
a missing banner means the probe did not load and nothing after it is a
measurement. And the exit line says ``buffers=[]`` explicitly rather than
staying quiet when no buffer was ever built.
"""
from __future__ import annotations

import atexit
import os
import threading

_BANNER = "[probe_site] observer INSTALLED (no line = probe not loaded, NOT a measurement)"


def _install() -> None:
    # Print BEFORE the import: a heavy or failing import must not be able to
    # turn this probe into a silent no-op.
    print(_BANNER, flush=True)

    try:
        from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        print(f"[probe_site] FAILED to import DiskReplayBuffer: {exc!r} — "
              "the observed process is NOT instrumented", flush=True)
        return

    seen: list[bool] = []
    orig = DiskReplayBuffer.__init__

    def patched(self, *a, **k) -> None:
        orig(self, *a, **k)
        if os.environ.get("PROBE_FORCE_RACE") == "1":
            # Counterfactual arm: re-arm the race the flag suppresses, so the
            # line below is proven able to report the other value.
            self._deterministic_refresh = False
            self._ensure_prefetch_thread()
            self._schedule_refresh_prefetch()
        seen.append(bool(self._deterministic_refresh))
        print(f"[probe_site] DiskReplayBuffer built deterministic_refresh="
              f"{self._deterministic_refresh} prefetch_thread="
              f"{self._prefetch_thread is not None}", flush=True)

    DiskReplayBuffer.__init__ = patched

    @atexit.register
    def _report() -> None:
        names = [
            t.name for t in threading.enumerate()
            if t.name.startswith("replay-prefetch-")
        ]
        note = "" if seen else "  <-- NO buffer was constructed in this process"
        print(f"[probe_site] EXIT buffers={seen} live_prefetch_threads={names}{note}",
              flush=True)


_install()
