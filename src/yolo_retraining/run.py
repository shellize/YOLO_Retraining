from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yolo_retraining.config import load_config
from yolo_retraining.engine import SequenceRunner, TaskRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo-retraining")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("task", "sequence"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        config = load_config(args.config, args.overrides)
        output = TaskRunner(config).run() if args.command == "task" else SequenceRunner(config).run()
        print(output)
        return 0
    except Exception as error:
        print(f"yolo-retraining error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

