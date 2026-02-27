"""HTTP server for distributing artifacts and collecting selfplay shards."""

from .app import create_app

__all__ = ["create_app"]
