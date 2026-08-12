from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Run chess-anti-engine HTTP server")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=45453)
    ap.add_argument("--server-root", type=str, default="server")
    ap.add_argument("--opening-book-path", type=str, default=None)
    ap.add_argument("--opening-book-path-2", type=str, default=None)
    ap.add_argument("--max-upload-mb", type=int, default=256)
    ap.add_argument("--min-workers-per-trial", type=int, default=1)
    ap.add_argument("--max-worker-delta-per-rebalance", type=int, default=1)
    ap.add_argument("--upload-compact-shard-size", type=int, default=2000)
    ap.add_argument("--upload-compact-max-age-seconds", type=float, default=90.0)
  # Default OFF everywhere it can be defaulted: an unset flag must never open
  # registration, so the closed deployment stays closed if a caller forgets it.
    ap.add_argument(
        "--worker-self-register", action="store_true",
        help="Allow unknown usernames to create an account on first use (TOFU). "
             "Volunteer deployments only; default off.",
    )
    ap.add_argument(
        "--require-worker-lease", action="store_true",
        help="⚑ DO NOT SET THIS ON THE IN-TREE FLEET -- it takes ingest to ZERO. "
             "Refuses shard uploads that do not carry an active lease owned by "
             "the authenticated account and matching the route's trial. The "
             "driver launches every worker with --trial-id, which sets "
             "fixed_trial_id, which SKIPS lease negotiation entirely -- so a "
             "driver-launched worker structurally never obtains or sends a lease "
             "id and is refused 403 on every upload, forever. (Measured on the "
             "live server: 821,818 uploads, zero leases ever issued.) This is "
             "for a volunteer deployment whose workers negotiate leases, and it "
             "is restart-gated, so it detonates only after a full run.py "
             "restart. Default off.",
    )
    args = ap.parse_args()

    try:
        import uvicorn
    except Exception as e:  # pragma: no cover
        raise RuntimeError("server requires uvicorn; install with pip install -e '.[server]' ") from e

    from chess_anti_engine.server.app import create_app
    from chess_anti_engine.server.lease import (
        active_run_prefix,
        prune_non_active_run_leases,
    )

    server_root = str(args.server_root)
    leases_root = str(args.server_root) + "/leases"
    prune_non_active_run_leases(
        leases_root=__import__("pathlib").Path(leases_root),
        active_prefix=active_run_prefix(server_root=__import__("pathlib").Path(server_root)),
    )

    app = create_app(
        server_root=server_root,
        opening_book_path=args.opening_book_path,
        opening_book_path_2=getattr(args, "opening_book_path_2", None),
        worker_self_register=bool(args.worker_self_register),
        require_worker_lease=bool(args.worker_self_register),
        max_upload_mb=int(args.max_upload_mb),
        min_workers_per_trial=int(args.min_workers_per_trial),
        max_worker_delta_per_rebalance=int(args.max_worker_delta_per_rebalance),
        upload_compact_shard_size=int(args.upload_compact_shard_size),
        upload_compact_max_age_seconds=float(args.upload_compact_max_age_seconds),
    )

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
