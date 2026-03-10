from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Run chess-anti-engine HTTP server")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=45453)
    ap.add_argument("--server-root", type=str, default="server")
    ap.add_argument("--opening-book-path", type=str, default=None)
    ap.add_argument("--max-upload-mb", type=int, default=256)
    args = ap.parse_args()

    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("server requires uvicorn; install with pip install -e '.[server]' ") from e

    from chess_anti_engine.server.app import create_app

    app = create_app(
        server_root=str(args.server_root),
        opening_book_path=args.opening_book_path,
        max_upload_mb=int(args.max_upload_mb),
    )

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
