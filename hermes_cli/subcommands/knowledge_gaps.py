"""``hermes knowledge-gaps`` read-only proposal command parser."""

from __future__ import annotations

from typing import Callable


def build_knowledge_gaps_parser(subparsers, *, cmd_knowledge_gaps: Callable) -> None:
    from hermes_cli.knowledge_gaps.__main__ import build_parser

    parser = subparsers.add_parser(
        "knowledge-gaps",
        help="Compare admitted evidence and review knowledge/content gap proposals",
        description="Read-only, source-neutral gap analysis with exact evidence bindings. "
        "Produces JSON proposals; never grants access or changes curated knowledge.",
    )
    build_parser(parser)
    parser.set_defaults(func=cmd_knowledge_gaps)
