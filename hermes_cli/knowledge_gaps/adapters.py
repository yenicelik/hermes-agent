"""Compose existing authorized readers; never discover or grant source access.

An adapter owns actual provider permissions, extraction and currentness. This
module binds exact requested identities and checks two observed reads, without
pretending that separate source systems provide an atomic global snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Mapping

from hermes_cli.knowledge_gaps.core import InputError, canonical, digest, validate


@dataclass(frozen=True)
class Selection:
    reader: str
    key: str
    source_id: str
    account: str
    tenant: str | None
    kind: str


def collect(readers: Mapping[str, Callable[[str], dict]], selections: list[Selection],
            *, scope: dict, as_of: str, claims: list | None = None,
            topics: list | None = None) -> tuple[dict, dict]:
    """Return a validated bounded snapshot and a body-free reader receipt.

    Readers return the records defined in docs/design/knowledge-gap-analysis.md.
    ``key`` is an
    opaque native ID; it is never fetched, executed or used as a filename here.
    Metadata-only denied observations must have empty text. All reads must be
    read-only. The caller supplies already authorized, bounded selections.
    """
    if not isinstance(selections, (list, tuple)) or not 0 < len(selections) <= 256:
        raise InputError("invalid bounded source selection")
    selections = tuple(selections)
    if any(not isinstance(s, Selection) for s in selections):
        raise InputError("invalid bounded source selection")
    # Check the domain before invoking even one reader. Invalid private-domain
    # selections must not cause reads merely to be rejected after capture.
    scope = validate({"schema_version": "hermes.gaps.input.v1", "as_of": as_of,
                      "scope": scope, "sources": []})["scope"]
    bound_readers = []
    for selection in selections:
        if any(not isinstance(value, str) or not value.strip() or len(value) > limit
               for value, limit in ((selection.reader, 256), (selection.key, 2048),
                                    (selection.source_id, 256), (selection.account, 256))):
            raise InputError("invalid source selection identity")
        if not isinstance(selection.kind, str) or selection.kind not in {"internal", "external", "own_draft", "own_published", "plan"}:
            raise InputError("invalid selected source kind")
        same_domain = selection.account == scope["account"] and selection.tenant == scope["tenant"]
        public = selection.account == "public" and selection.tenant is None and selection.kind == "external"
        if not (same_domain or (public and scope["public_external_allowed"])):
            raise InputError("source selection outside admitted scope")
        reader = readers.get(selection.reader)
        if not callable(reader):
            raise InputError("selected reader unavailable")
        bound_readers.append(reader)
    if len({s.source_id for s in selections}) != len(selections):
        raise InputError("duplicate source selection identity")
    try:
        claims = json.loads(canonical(claims if claims is not None else []))
        topics = json.loads(canonical(topics if topics is not None else []))
    except Exception:
        raise InputError("invalid capture annotations") from None
    records = []
    for selection, reader in zip(selections, bound_readers):
        try:
            source = reader(selection.key)
        except Exception:
            raise InputError("source reader failed; no snapshot published") from None
        expected = {"id": selection.source_id, "account": selection.account,
                    "tenant": selection.tenant, "kind": selection.kind,
                    "policy_revision": scope.get("policy_revision")}
        if not isinstance(source, dict) or any(source.get(k) != v for k, v in expected.items()):
            raise InputError("source identity or policy changed")
        # Round-trip makes an owned copy; callers cannot mutate a returned view.
        try:
            records.append(json.loads(canonical(source)))
        except Exception:
            raise InputError("invalid source reader record") from None
    payload = {"schema_version": "hermes.gaps.input.v1", "as_of": as_of,
               "scope": scope, "sources": records, "claims": claims, "topics": topics}
    validated = validate(payload)
    for selection, reader, before in zip(selections, bound_readers, records):
        try:
            after = reader(selection.key)
            equal = canonical(before) == canonical(after)
        except Exception:
            raise InputError("source recheck failed; no snapshot published") from None
        if not equal:
            raise InputError("source changed during bounded capture")
    receipt = {"schema_version": "hermes.gaps.capture-receipt.v1",
               "input_sha256": digest(validated), "as_of": as_of,
               "selection_count": len(selections), "all_rechecks_equal": True,
               "sources": [{"source_id": r["id"], "revision": r["revision"],
                            "text_sha256": r["sha256"], "reader": s.reader}
                           for s, r in zip(selections, records)],
               "limits": ["Reader admission is a trusted adapter assertion, not a new grant.",
                          "Two observed reads are not an atomic cross-system snapshot.",
                          "Revalidate current permissions and source revisions before live display."]}
    return validated, receipt
