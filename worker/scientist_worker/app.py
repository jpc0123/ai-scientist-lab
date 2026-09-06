from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from scientist_worker.api import create_app
from scientist_worker.settings import WorkerSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scientist-worker")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start Worker HTTP API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--data-root", default=None)
    serve.add_argument("--token", default=None, help="Optional bearer token")
    serve.add_argument(
        "--require-auth",
        action="store_true",
        help="Reject unauthenticated requests even if token unset",
    )
    serve.add_argument(
        "--executor-mode",
        choices=["mock", "docker", "auto"],
        default=None,
        help="Job executor mode (default: auto)",
    )
    serve.add_argument(
        "--project-root",
        default=None,
        help="Scientist-lab project root for dataset/code registries",
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        settings = WorkerSettings()
        if args.host:
            settings.host = args.host
        if args.port is not None:
            settings.port = int(args.port)
        if args.data_root:
            settings.data_root = Path(args.data_root)
        if args.token:
            settings.auth_token = args.token
        if args.require_auth:
            settings.require_auth = True
        if args.executor_mode:
            settings.executor_mode = args.executor_mode
        if args.project_root:
            settings.project_root = Path(args.project_root)
        settings = settings.resolve()
        app = create_app(settings)
        uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
