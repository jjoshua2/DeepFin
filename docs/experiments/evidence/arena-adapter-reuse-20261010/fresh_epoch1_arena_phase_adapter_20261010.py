"""Fresh endpoint bindings for the existing pinned short-E phase adapter."""
import hashlib
from pathlib import Path

SHARED_SOURCE = Path(__file__).with_name("short-e-evaluation-owner-v4") / "arena_phase_adapter_pair_bound_v4.py"
SHARED_SHA256 = "5e7133d525db7854ded59af42a309f58316082e8410e6d1d92755f1e967fb497"


def main():
    # Execute the bytes that were checked, not a second read or a sys.path import.
    raw = SHARED_SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHARED_SHA256:
        raise SystemExit("arena phase adapter refused: shared adapter source changed")
    shared = {"__name__": "_pinned_short_e_phase_adapter", "__file__": str(SHARED_SOURCE)}
    exec(compile(raw, str(SHARED_SOURCE), "exec"), shared)
    try:
        return shared["main"](
            adapter_source=Path(__file__),
            candidate_name="fresh104363327_epoch1",
            reference_name="THREE_frozen206922",
            pair_bound_wrapper=Path("/mnt/c/Users/jjosh/Documents/Codex/2026-10-08/task/fresh_epoch1_pair_bound_20261010.py"),
        )
    except shared["AdapterError"] as exc:
        raise SystemExit(f"arena phase adapter refused: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
