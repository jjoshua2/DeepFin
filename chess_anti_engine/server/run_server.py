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
    args = ap.parse_args()

    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("server requires uvicorn; install with pip install -e '.[server]' ") from e

    from chess_anti_engine.server.lease import active_run_prefix, prune_non_active_run_leases
    from chess_anti_engine.server.app import create_app

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
        max_upload_mb=int(args.max_upload_mb),
        min_workers_per_trial=int(args.min_workers_per_trial),
        max_worker_delta_per_rebalance=int(args.max_worker_delta_per_rebalance),
        upload_compact_shard_size=int(args.upload_compact_shard_size),
        upload_compact_max_age_seconds=float(args.upload_compact_max_age_seconds),
    )

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
