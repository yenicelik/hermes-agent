"""Source-neutral admission of agent-authored, evidence-bound comparisons.

The caller supplies reasoning from an already approved execution path. This
module invokes no provider and authenticates no upstream permission or reviewer.
It proves exact citation bindings, not the semantic truth of a proposed gap.
"""
from __future__ import annotations

from typing import Any
import re

from . import core

VERSION = "hermes.gaps.semantic.v1"
DIFF_VERSION = "hermes.gaps.semantic-diff.v1"
KINDS = {"evidence_gap", "missing_connection", "content_gap", "possible_contradiction", "redundancy"}
FIELDS = {"id", "kind", "title", "observation", "why_it_matters", "next_step", "uncertainty",
          "impact", "evidence", "basis"}
REF_FIELDS = {"source_id", "start", "end", "quote", "revision", "sha256"}
LIMITS = {"id": 128, "title": 240, "observation": 2000, "why_it_matters": 1200,
          "next_step": 1200, "uncertainty": 1200}


def validate_comparisons(envelope: Any, proposals: Any, *, limit: int = 5) -> dict:
    """Mechanically validate cross-kind proposals against current scoped text.

Offsets are Unicode codepoint [start,end), and sha256 binds the full source
text. Each proposal needs at least two distinct sources and source kinds.
Unknown fields are refused, including any attempted acceptance/execution state.
"""
    if type(limit) is not int or not 1 <= limit <= 5:
        core.fail("semantic.limit")
    data = core.validate(envelope)
    sources = {row["id"]: row for row in data["sources"]}
    seen, candidates = set(), []
    for item in core.sequence(proposals, "semantic.proposals", core.MAX_ITEMS):
        row = core.obj(item, "semantic.proposal")
        if set(row) != FIELDS:
            core.fail("semantic.proposal.fields")
        for key, bound in LIMITS.items():
            core.string(row[key], "semantic.proposal." + key, limit=bound)
        if row["id"] in seen:
            core.fail("semantic.proposal.duplicate_id")
        seen.add(row["id"])
        core.oneof(row["kind"], KINDS, "semantic.proposal.kind")
        if row["basis"] != "agent_review":
            core.fail("semantic.proposal.basis")
        if type(row["impact"]) is not int or not 1 <= row["impact"] <= 5:
            core.fail("semantic.proposal.impact")
        citations, spans, source_ids, source_kinds = [], set(), set(), set()
        for item_ref in core.sequence(row["evidence"], "semantic.evidence", 12):
            ref = core.obj(item_ref, "semantic.evidence.item")
            if set(ref) != REF_FIELDS:
                core.fail("semantic.evidence.fields")
            checked = core.validate_evidence(ref, sources, "semantic.evidence")
            span = (checked["source_id"], checked["start"], checked["end"])
            if span in spans:
                core.fail("semantic.evidence.duplicate_span")
            spans.add(span)
            source = sources[checked["source_id"]]
            source_ids.add(source["id"])
            source_kinds.add(source["kind"])
            citations.append(core.from_ref(checked, sources) | {
                "source_kind": source["kind"], "source_title": source["title"]})
        if len(source_ids) < 2 or len(source_kinds) < 2:
            core.fail("semantic.evidence.cross_source_kinds")
        identity = {key: data["scope"].get(key) for key in ("account", "tenant")}
        identity.update({"kind": row["kind"], "proposal_key": row["id"]})
        candidate = {key: row[key] for key in LIMITS if key != "id"}
        candidate.update({"id": "sgap_" + core.digest(identity)[:24], "proposal_key": row["id"],
                          "kind": row["kind"], "basis": "agent_review", "impact": row["impact"],
                          "status": "proposal_only", "validation": "unvalidated",
                          "evidence_validation": "exact_current_scoped_citation_bindings_only",
                          "evidence": sorted(citations, key=lambda e: (e["source_id"], e["start"], e["end"]))})
        candidate["material_sha256"] = core.digest(candidate)
        candidates.append(candidate)
    candidates.sort(key=lambda c: (-c["impact"], c["id"]))
    result = {"schema_version": VERSION, "as_of": data["as_of"], "scope": data["scope"],
              "input_sha256": core.digest(data), "status": "proposal_only", "validation": "unvalidated",
              "admission": "External agent assertions; exact citation checks do not validate semantic truth, "
                           "upstream permission, human approval, or an editorial commitment.",
              "candidates": candidates, "queue": candidates[:limit],
              "counts": {"candidates": len(candidates), "queued": min(limit, len(candidates))}}
    result["snapshot_sha256"] = core.digest(result)
    return result


def _report(value: Any) -> dict:
    row = core.obj(value, "semantic.snapshot")
    expected = {"schema_version", "as_of", "scope", "input_sha256", "status", "validation",
                "admission", "candidates", "queue", "counts", "snapshot_sha256"}
    if set(row) != expected or row["schema_version"] != VERSION:
        core.fail("semantic.snapshot.schema")
    if row["status"] != "proposal_only" or row["validation"] != "unvalidated":
        core.fail("semantic.snapshot.authority")
    if row["snapshot_sha256"] != core.digest({k: v for k, v in row.items() if k != "snapshot_sha256"}):
        core.fail("semantic.snapshot.hash")
    core.timestamp(row["as_of"], "semantic.snapshot.as_of")
    scope = core.obj(row["scope"], "semantic.snapshot.scope")
    for key in ("account", "policy_revision"):
        core.string(scope.get(key), "semantic.snapshot.scope." + key, limit=256)
    if scope.get("tenant") is not None:
        core.string(scope["tenant"], "semantic.snapshot.scope.tenant", limit=256)
    indexed = {}
    for item in core.sequence(row["candidates"], "semantic.snapshot.candidates", core.MAX_ITEMS):
        candidate = core.obj(item, "semantic.snapshot.candidate")
        cid = candidate.get("id")
        if not isinstance(cid, str) or re.fullmatch(r"sgap_[0-9a-f]{24}", cid) is None or cid in indexed:
            core.fail("semantic.snapshot.candidate.id")
        if (candidate.get("status") != "proposal_only" or candidate.get("validation") != "unvalidated"
                or candidate.get("basis") != "agent_review"):
            core.fail("semantic.snapshot.candidate.authority")
        material = {key: value for key, value in candidate.items() if key != "material_sha256"}
        if candidate.get("material_sha256") != core.digest(material):
            core.fail("semantic.snapshot.candidate.hash")
        indexed[cid] = candidate
    queue = core.sequence(row["queue"], "semantic.snapshot.queue", 5)
    if (row["counts"] != {"candidates": len(indexed), "queued": len(queue)}
            or queue != row["candidates"][:len(queue)]):
        core.fail("semantic.snapshot.queue_or_counts")
    return indexed


def compare_semantic(before: Any, after: Any) -> dict:
    """Return ID-only changes over full sets; do not replay withdrawn evidence.

Snapshot hashes are integrity checks, not authenticated authority. The caller
must re-admit each new envelope through its current source reader.
"""
    old, new = _report(before), _report(after)
    if any(before["scope"].get(k) != after["scope"].get(k) for k in ("account", "tenant")):
        core.fail("semantic.diff.cross_scope")
    if core.timestamp(after["as_of"], "semantic.diff.after") < core.timestamp(before["as_of"], "semantic.diff.before"):
        core.fail("semantic.diff.time_order")
    return {"schema_version": DIFF_VERSION, "before_sha256": before["snapshot_sha256"],
            "after_sha256": after["snapshot_sha256"], "status": "proposal_only",
            "added": sorted(new.keys() - old.keys()),
            "material_changed": sorted(cid for cid in old.keys() & new.keys()
                                       if old[cid]["material_sha256"] != new[cid]["material_sha256"]),
            "no_longer_observed": sorted(old.keys() - new.keys()),
            "unchanged": sorted(cid for cid in old.keys() & new.keys()
                                if old[cid]["material_sha256"] == new[cid]["material_sha256"]),
            "scope_changes": sorted(k for k in set(before["scope"]) | set(after["scope"])
                                    if before["scope"].get(k) != after["scope"].get(k)),
            "interpretation": "No longer observed does not mean resolved. Withdrawal, access, sampling or "
                              "analysis changes may remove a proposal. No prior quotes or assertions are repeated."}
