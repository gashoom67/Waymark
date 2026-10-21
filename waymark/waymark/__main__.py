from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(prog="waymark", description="Multi-hop GraphRAG reader for PDFs and images")
    parser.add_argument("--host", default=os.environ.get("WAYMARK_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WAYMARK_PORT", "8000")))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()
    if args.data_dir:
        os.environ["WAYMARK_DATA_DIR"] = args.data_dir
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    import uvicorn

    from waymark.server.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
