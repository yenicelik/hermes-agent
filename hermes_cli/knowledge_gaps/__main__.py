"""JSON proposal CLI shared by `hermes knowledge-gaps` and module execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from .core import InputError, analyze, compare, evaluate, validate

MAX_FILE_BYTES = 16 * 1024 * 1024


def load(path: str) -> dict:
    """Read one bounded regular file without following a final symlink."""
    # Nonblocking open lets fstat reject FIFOs without waiting for a writer.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise InputError("input must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
        after = os.fstat(fd)
        if len(raw) > MAX_FILE_BYTES or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise InputError("input changed or exceeded byte limit")
    finally:
        os.close(fd)

    def unique(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise InputError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_: str) -> None:
        raise InputError("nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def write_immutable(path: str, raw: bytes) -> None:
    """Atomic, private, no-replace publication; identical bytes are a no-op."""
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.parent.is_symlink() or stat.S_IMODE(target.parent.stat().st_mode) & 0o077:
        raise InputError("output parent must be a private directory (mode 0700)")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".gap-", delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or stat.S_IMODE(target.stat().st_mode) & 0o077:
                raise InputError("existing output is not a private regular file")
            if target.stat().st_size != len(raw) or target.read_bytes() != raw:
                raise InputError("immutable output already exists with different contents")
        directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    # Keep the outer Hermes `command` intact for its startup gates.
    sub = parser.add_subparsers(dest="gap_action", required=True)
    validation = sub.add_parser("validate")
    validation.add_argument("--input", required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--input", required=True)
    analysis.add_argument("--output")
    analysis.add_argument("--limit", type=int, default=5)
    diff = sub.add_parser("diff")
    diff.add_argument("--before", required=True)
    diff.add_argument("--after", required=True)
    diff.add_argument("--output")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--snapshot", required=True)
    evaluation.add_argument("--labels", required=True)
    evaluation.add_argument("--output")
    review = sub.add_parser("review", help="Validate agent-authored comparisons against admitted evidence")
    review.add_argument("--input", required=True)
    review.add_argument("--proposals", required=True)
    review.add_argument("--output")
    return parser


def _validate(args: argparse.Namespace) -> dict:
    data = validate(load(args.input))
    return {"valid": True, "schema_version": data["schema_version"], "sources": len(data["sources"]),
            "authority": "adapter assertions only; no upstream authorization is granted"}


def _analyze(args: argparse.Namespace) -> dict:
    return analyze(load(args.input), limit=args.limit)


def _diff(args: argparse.Namespace) -> dict:
    return compare(load(args.before), load(args.after))


def _evaluate(args: argparse.Namespace) -> dict:
    return evaluate(load(args.snapshot), load(args.labels))


def _review(args: argparse.Namespace) -> dict:
    from .semantic_cli import review
    return review(args)


_COMMANDS = {"validate": _validate, "analyze": _analyze, "diff": _diff,
             "evaluate": _evaluate, "review": _review}


def run(args: argparse.Namespace) -> int:
    try:
        result = _COMMANDS[args.gap_action](args)
        raw = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        if getattr(args, "output", None):
            write_immutable(args.output, raw)
            print(json.dumps({"ok": True, "output_written_or_identical": True, "schema_version": result["schema_version"]}))
        else:
            sys.stdout.buffer.write(raw)
        return 0
    except (InputError, OSError, ValueError, TypeError, RecursionError):
        # Paths, source bodies and invalid values are deliberately excluded.
        print(json.dumps({"ok": False, "error": "invalid_input_or_immutable_output_refused"}), file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
