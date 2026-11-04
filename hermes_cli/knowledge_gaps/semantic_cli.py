"""Semantic review uses the common JSON CLI and source-admission contract."""
import sys


def review(args) -> dict:
    from .__main__ import load
    from .semantic import validate_comparisons
    return validate_comparisons(load(args.input), load(args.proposals))


def main(argv: list[str] | None = None) -> int:
    from .__main__ import main as common_main
    return common_main(["review", *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    raise SystemExit(main())
