"""Reusable parsing helpers for batch-oriented dataset analysis."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable


_BATCH_TOKEN = re.compile(r"^(?P<start>\d+)(?:-(?P<end>\d+))?$")


def parse_batch_spec(spec: str | Iterable[int]) -> list[int]:
    """Parse a batch specification such as ``"1,2,5-8"``.

    Batch numbers are one-based. Tokens may be individual positive integers or
    inclusive ranges. The returned list is sorted and de-duplicated.
    """

    if not isinstance(spec, str):
        values = [int(value) for value in spec]
        if any(value < 1 for value in values):
            raise ValueError("batch numbers must be positive integers")
        return sorted(set(values))

    if not spec.strip():
        raise ValueError("batch specification cannot be empty")

    batches: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        match = _BATCH_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(
                f"invalid batch token {token!r}; expected values like '2' or '5-8'"
            )
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if start < 1 or end < 1:
            raise ValueError("batch numbers must be positive integers")
        if start > end:
            raise ValueError(f"batch range must be ascending: {token!r}")
        batches.update(range(start, end + 1))

    return sorted(batches)


def batch_names(batch_numbers: Iterable[int]) -> list[str]:
    """Return canonical directory names for batch numbers."""

    return [f"batch_{number:02d}" for number in sorted(set(batch_numbers))]


class BatchSpecAction(argparse.Action):
    """Argparse action that stores a parsed, sorted batch-number list."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            parsed = parse_batch_spec(values)
        except ValueError as exc:
            raise argparse.ArgumentError(self, str(exc)) from exc
        setattr(namespace, self.dest, parsed)


def add_batch_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--batches`` argument to an analysis CLI parser."""

    parser.add_argument(
        "--batches",
        required=True,
        action=BatchSpecAction,
        metavar="SPEC",
        help="batch numbers, e.g. '1,2,5-8' (inclusive ranges)",
    )

