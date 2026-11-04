"""Deterministic proposal generation from a bounded, adapter-admitted snapshot.

This module does not authenticate upstream permissions or human review receipts.
The upstream reader owns those checks. Every output is an unvalidated proposal;
neither lexical non-matches nor corpus coverage establish a knowledge gap.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

INPUT_VERSION = "hermes.gaps.input.v1"
OUTPUT_VERSION = "hermes.gaps.output.v1"
DIFF_VERSION = "hermes.gaps.diff.v1"
ENGINE_VERSION = "0.1.1"
MAX_SOURCES = 256
MAX_TEXT = 200_000
MAX_TOTAL_TEXT = 2_000_000
MAX_ITEMS = 128
MAX_QUOTE = 360
KINDS = {"internal", "external", "own_draft", "own_published", "plan"}
STOP = set("a an and are as at be been but by can could do does for from had has have how i in into is it its may of on or our should that the their there these they this to was we were what when where which who why will with would you your".split())


class InputError(ValueError):
    """Validation errors deliberately omit source text and input values."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fail(field: str) -> None:
    raise InputError("invalid " + field)


def obj(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        fail(field)
    return value


def string(value: Any, field: str, *, empty: bool = False, limit: int = 2048) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()) or "\x00" in value:
        fail(field)
    return value


def boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        fail(field)
    return value


def timestamp(value: Any, field: str) -> datetime:
    raw = string(value, field, limit=64)
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        fail(field)
    if result.tzinfo is None:
        fail(field)
    return result.astimezone(timezone.utc)


def sequence(value: Any, field: str, limit: int) -> list:
    if not isinstance(value, list) or len(value) > limit:
        fail(field)
    return value


def oneof(value: Any, choices: set, field: str) -> Any:
    if not isinstance(value, str) or value not in choices:
        fail(field)
    return value


def tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[\w]+", text.lower(), re.UNICODE) if len(x) > 2 and x not in STOP}


def lexical(query: str | set[str], text: str) -> float:
    query_tokens = tokens(query) if isinstance(query, str) else query
    return len(query_tokens & tokens(text)) / len(query_tokens) if query_tokens else 0.0


def eligible(source: dict) -> bool:
    return (source["authorized"] and source["current"] and not source["withdrawn"]
            and source["access"] == "accessible" and source["capture"] == "captured"
            and (source["index"] == "indexed" or source.get("text_basis") == "direct_current_source")
            and bool(source["text"].strip()))


def validate_evidence(value: Any, sources: dict[str, dict], field: str) -> dict:
    ref = obj(value, field)
    source_id = string(ref.get("source_id"), field + ".source_id", limit=256)
    source = sources.get(source_id)
    if source is None or not eligible(source):
        fail(field + ".eligible_source")
    start, end = ref.get("start"), ref.get("end")
    if (type(start) is not int or type(end) is not int or start < 0
            or start >= end or end > len(source["text"]) or end - start > MAX_QUOTE):
        fail(field + ".span")
    for key, expected in (("revision", source["revision"]), ("sha256", source["sha256"]),
                          ("quote", source["text"][start:end])):
        if key in ref and ref[key] != expected:
            fail(field + ".binding")
    return {"source_id": source_id, "start": start, "end": end}


def validate(payload: Any) -> dict:
    root = obj(payload, "input")
    if root.get("schema_version") != INPUT_VERSION:
        fail("schema_version")
    as_of = timestamp(root.get("as_of"), "as_of")
    scope = obj(root.get("scope"), "scope")
    for field in ("account", "policy_revision"):
        string(scope.get(field), "scope." + field, limit=256)
    if scope.get("tenant") is not None:
        string(scope["tenant"], "scope.tenant", limit=256)
    boolean(scope.get("public_external_allowed", False), "scope.public_external_allowed")
    oneof(scope.get("coverage"), {"bounded", "complete_for_declared_scope", "unknown"}, "scope.coverage")
    sources = {}
    total = 0
    for index, item in enumerate(sequence(root.get("sources"), "sources", MAX_SOURCES)):
        prefix = "sources[%d]" % index
        source = obj(item, prefix)
        source_id = string(source.get("id"), prefix + ".id", limit=256)
        if source_id in sources:
            fail(prefix + ".duplicate_id")
        for field in ("revision", "sha256", "policy_revision", "account"):
            string(source.get(field), prefix + "." + field, limit=256)
        for field in ("title", "uri"):
            string(source.get(field), prefix + "." + field, empty=True)
        kind = oneof(source.get("kind"), KINDS, prefix + ".kind")
        is_public = source["account"] == "public" and source.get("tenant") is None and kind == "external"
        if not ((source["account"] == scope["account"] and source.get("tenant") == scope.get("tenant"))
                or (is_public and scope.get("public_external_allowed", False))):
            fail(prefix + ".scope")
        if source["policy_revision"] != scope["policy_revision"]:
            fail(prefix + ".policy_revision")
        for field in ("authorized", "current", "withdrawn"):
            boolean(source.get(field), prefix + "." + field)
        oneof(source.get("access"), {"accessible", "not_accessible", "unknown"}, prefix + ".access")
        oneof(source.get("capture"), {"captured", "not_captured", "unknown"}, prefix + ".capture")
        oneof(source.get("index"), {"indexed", "not_indexed", "unknown"}, prefix + ".index")
        body = string(source.get("text"), prefix + ".text", empty=True, limit=MAX_TEXT)
        total += len(body)
        if total > MAX_TOTAL_TEXT or text_hash(body) != source["sha256"]:
            fail(prefix + ".sha256_or_size")
        if (source["withdrawn"] or not source["authorized"] or source["access"] != "accessible"
                or source["capture"] != "captured") and body:
            fail(prefix + ".unavailable_body")
        if timestamp(source.get("observed_at"), prefix + ".observed_at") > as_of:
            fail(prefix + ".future_timestamp")
        if "published_at" not in source:
            fail(prefix + ".published_at")
        if source["published_at"] is not None and timestamp(source["published_at"], prefix + ".published_at") > as_of:
            fail(prefix + ".future_timestamp")
        basis = oneof(source.get("text_basis", "canonical_index"),
                      {"canonical_index", "direct_current_source"}, prefix + ".text_basis")
        if basis == "direct_current_source":
            projection = obj(source.get("projection"), prefix + ".projection")
            string(projection.get("reader"), prefix + ".projection.reader", limit=256)
            if (projection.get("revision") != source["revision"]
                    or projection.get("text_sha256") != source["sha256"]):
                fail(prefix + ".projection.binding")
            if timestamp(projection.get("checked_at"), prefix + ".projection.checked_at") > as_of:
                fail(prefix + ".projection.future_timestamp")
        elif "projection" in source:
            fail(prefix + ".projection.without_direct_basis")
        sources[source_id] = dict(source)
    claims, seen = [], set()
    for index, item in enumerate(sequence(root.get("claims", []), "claims", MAX_ITEMS)):
        prefix = "claims[%d]" % index
        claim = dict(obj(item, prefix))
        for field in ("id", "subject", "predicate", "value", "context"):
            string(claim.get(field), prefix + "." + field, limit=512)
        if claim["id"] in seen:
            fail(prefix + ".duplicate_id")
        seen.add(claim["id"])
        if timestamp(claim.get("valid_at"), prefix + ".valid_at") > as_of:
            fail(prefix + ".future_timestamp")
        boolean(claim.get("corroborated", False), prefix + ".corroborated")
        claim["evidence"] = validate_evidence(claim.get("evidence"), sources, prefix + ".evidence")
        claims.append(claim)
    topics, seen = [], set()
    for index, item in enumerate(sequence(root.get("topics", []), "topics", MAX_ITEMS)):
        prefix = "topics[%d]" % index
        topic = dict(obj(item, prefix))
        for field in ("id", "title"):
            string(topic.get(field), prefix + "." + field, limit=512)
        if topic["id"] in seen:
            fail(prefix + ".duplicate_id")
        seen.add(topic["id"])
        terms = sequence(topic.get("terms"), prefix + ".terms", 32)
        if not terms:
            fail(prefix + ".terms")
        for term in terms:
            string(term, prefix + ".term", limit=80)
        if not tokens(" ".join(terms)):
            fail(prefix + ".empty_terms")
        oneof(topic.get("status"), {"proposed", "accepted"}, prefix + ".status")
        oneof(topic.get("intent", "learn"), {"learn", "write"}, prefix + ".intent")
        topic["evidence"] = validate_evidence(topic.get("evidence"), sources, prefix + ".evidence")
        if topic["status"] == "accepted":
            acceptance = obj(topic.get("acceptance"), prefix + ".acceptance")
            source = sources[topic["evidence"]["source_id"]]
            quote = source["text"][topic["evidence"]["start"]:topic["evidence"]["end"]]
            expected = {"actor_type": "human", "source_id": source["id"], "source_revision": source["revision"],
                        "evidence_sha256": text_hash(quote)}
            if source["kind"] != "plan" or any(acceptance.get(k) != v for k, v in expected.items()):
                fail(prefix + ".acceptance_binding")
            for field in ("reviewer", "review_ref"):
                string(acceptance.get(field), prefix + ".acceptance." + field)
        topics.append(topic)
    stale_days = root.get("stale_after_days", 90)
    if type(stale_days) is not int or not 1 <= stale_days <= 3650:
        fail("stale_after_days")
    # Do not reflect opaque adapter metadata (which may be private) into output.
    normalized_scope = {key: scope.get(key) for key in ("account", "tenant", "policy_revision", "coverage")}
    normalized_scope["public_external_allowed"] = scope.get("public_external_allowed", False)
    return {"schema_version": INPUT_VERSION, "as_of": root["as_of"], "scope": normalized_scope,
            "sources": sorted(sources.values(), key=lambda s: s["id"]),
            "claims": sorted(claims, key=lambda c: c["id"]),
            "topics": sorted(topics, key=lambda t: t["id"]), "stale_after_days": stale_days}


def evidence(source: dict, start: int = 0, end: int | None = None, *, metadata: bool = False) -> dict:
    end = min(len(source["text"]), start + MAX_QUOTE) if end is None else end
    if metadata:
        start, end = 0, 0
    quote = source["text"][start:end]
    return {"source_id": source["id"], "revision": source["revision"], "source_sha256": source["sha256"],
            "uri": source["uri"], "start": start, "end": end, "quote": quote,
            "quote_sha256": text_hash(quote), "basis": "metadata" if metadata else "text_span",
            "text_basis": source.get("text_basis", "canonical_index"), "index_state": source["index"]}


def from_ref(ref: dict, sources: dict) -> dict:
    return evidence(sources[ref["source_id"]], ref["start"], ref["end"])


def match_span(source: dict, query: set[str]) -> dict:
    matches = [m for m in re.finditer(r"\w+", source["text"], re.UNICODE) if m.group(0).lower() in query]
    start = max(0, matches[0].start() - 60) if matches else 0
    return evidence(source, start)


def analyze(payload: Any, *, limit: int = 5) -> dict:
    if type(limit) is not int or not 1 <= limit <= 5:
        fail("limit")
    data = validate(payload)
    sources = {s["id"]: s for s in data["sources"]}
    usable = sorted((s for s in sources.values() if eligible(s)), key=lambda s: s["id"])
    as_of = timestamp(data["as_of"], "as_of")
    found = {}

    def add(kind: str, key: Any, title: str, reason: str, refs: list, impact: int,
            confidence: float, action: str, uncertainty: str, method: str, details: dict | None = None) -> None:
        gap_id = "gap_" + digest([kind, key])[:24]
        item = {"id": gap_id, "kind": kind, "title": title, "status": "proposal_only",
                "validation": "unvalidated", "reason": reason, "evidence": refs,
                "impact": impact, "confidence": confidence, "uncertainty": uncertainty,
                "proposed_next_action": action, "method": method, "details": details or {}}
        item["priority"] = round(impact * confidence, 4)
        item["material_sha256"] = digest(item)
        found[gap_id] = item

    # Known source-state gaps are distinct from knowledge or editorial gaps.
    exclusions = Counter()
    for source in sorted(sources.values(), key=lambda s: s["id"]):
        if source["withdrawn"] or not source["authorized"] or not source["current"]:
            exclusions["withdrawn" if source["withdrawn"] else "unauthorized" if not source["authorized"] else "noncurrent"] += 1
            continue
        statuses = []
        if source["access"] != "accessible":
            statuses.append(("not_accessible" if source["access"] == "not_accessible" else "access_unknown", "Confirm legitimate source access and keep unavailable bodies excluded."))
        elif source["capture"] != "captured":
            statuses.append(("not_captured" if source["capture"] == "not_captured" else "capture_unknown", "Inspect the existing passive collector receipt and reconcile the declared source interval."))
        elif source["index"] != "indexed":
            statuses.append(("not_in_index" if source["index"] == "not_indexed" else "index_unknown", "Inspect the existing canonical indexing receipt before evaluating content coverage."))
        elif not source["text"].strip():
            statuses.append(("indexed_text_unavailable", "Check whether this record is metadata-only or has a separately authorized text projection."))
        for state, action in statuses:
            direct_available = source.get("text_basis") == "direct_current_source" and eligible(source)
            if direct_available:
                action = "Use the admitted direct source for this comparison; verify canonical indexing only if indexed discovery is required."
            add("source_coverage", source["id"], "Source coverage: " + state,
                "The admitted snapshot reports a source access, capture or indexing limitation.",
                [evidence(source, metadata=True)], 1 if direct_available else 4, 1.0, action,
                "Source state is asserted by the input adapter; it does not establish upstream completeness or a knowledge gap.",
                "source_state", {"coverage_state": state, "source_id": source["id"], "analysis_blocked": not eligible(source)})

    # Question discovery is text-based; absence of lexical matches is explicitly weak evidence.
    for source in usable:
        if source["kind"] not in {"internal", "own_draft", "plan"}:
            continue
        seen_questions = set()
        for match in re.finditer(r"(?:^|(?<=[.!?\n]))\s*([^\n?]{8,240}\?)", source["text"]):
            question = match.group(1)
            terms = tokens(question)
            if len(terms) < 2 or question in seen_questions:
                continue
            seen_questions.add(question)
            related = sorted(((lexical(terms, other["text"]), other["id"]) for other in usable if other["id"] != source["id"]), reverse=True)
            best = related[0][0] if related else 0.0
            if best < 0.6:
                add("unanswered_question", [source["id"], sorted(terms)], question,
                    "No candidate answer with sufficient lexical overlap was located in this bounded snapshot.",
                    [evidence(source, match.start(1), match.end(1))], 2, 0.35,
                    "Search the current authorized corpus and verify an actual answer with its evidence.",
                    "A paraphrased answer, inaccessible source, unindexed text or missing capture can explain the non-match.",
                    "question_pattern_and_lexical_search", {"best_lexical_overlap": round(best, 4), "threshold": 0.6})
            if len(seen_questions) >= 3:
                break

    # Explicit claim annotations stay analyst interpretations of exact quoted spans.
    groups = {}
    for claim in data["claims"]:
        effective_time = timestamp(claim["valid_at"], "valid_at").isoformat()
        group = tuple(claim[k] for k in ("subject", "predicate", "context")) + (effective_time,)
        groups.setdefault(group, []).append(claim)
        ref = from_ref(claim["evidence"], sources)
        if not claim.get("corroborated", False):
            add("weak_evidence", claim["id"], "Check support: " + claim["subject"],
                "The supplied claim annotation has no corroboration recorded in this snapshot.", [ref], 2, 0.6,
                "Find independent support or qualify the claim before reusing it.",
                "The annotation may omit existing corroboration; this is not a finding that the claim is false.",
                "analyst_annotation_support_check", {"claim_id": claim["id"]})
        source = sources[claim["evidence"]["source_id"]]
        # Unknown publication time is not the cache's first-seen or modified
        # time. Drafts may be unpublished; never manufacture an age claim.
        published_at = source["published_at"]
        age = (as_of - timestamp(published_at, "published_at")).days if published_at is not None else None
        if age is not None and age > data["stale_after_days"]:
            add("stale_coverage", claim["id"], "Refresh evidence: " + claim["subject"],
                "The cited source predates the configured review horizon.", [ref], 3, 0.7,
                "Check a current source and record whether the claim still holds.",
                "Age is a review trigger, not evidence that a claim changed or became false.",
                "source_age_policy", {"claim_id": claim["id"], "published_at": published_at,
                                      "date_basis": "declared_publication_time", "horizon_days": data["stale_after_days"]})
    for group, claims in sorted(groups.items()):
        if len({c["value"].strip().casefold() for c in claims}) < 2:
            continue
        add("possible_contradiction", list(group), "Reconcile claims: " + group[0],
            "Claim annotations assign different values to the same subject, predicate, context and effective time.",
            [from_ref(c["evidence"], sources) for c in sorted(claims, key=lambda c: c["id"])], 5, 0.8,
            "Verify the quoted wording, attribution and scope; retain both versions until a human resolves the difference.",
            "Structured claim annotations are analyst interpretations. Differing meanings or qualifiers can remove the apparent conflict.",
            "scoped_claim_value_comparison", {"claim_ids": sorted(c["id"] for c in claims)})

    # External perspectives must relate to a declared review target with evidence.
    for topic in sorted(data["topics"], key=lambda t: t["id"]):
        terms = tokens(" ".join(topic["terms"]))
        # A target can be anchored in substantive internal/own writing. That
        # source still counts as coverage; excluding every anchor fabricated a
        # gap when the only internal match was the target's own source. Plan
        # records are naturally excluded from the evidence categories below.
        matches = [s for s in usable if lexical(terms, s["text"]) >= 0.5]
        external = [s for s in matches if s["kind"] == "external"]
        internal = [s for s in matches if s["kind"] in {"internal", "own_draft", "own_published"}]
        own = [s for s in matches if s["kind"] in {"own_draft", "own_published"}]
        details = {"topic_id": topic["id"], "target_status": topic["status"], "target_intent": topic.get("intent", "learn"),
                   "match_threshold": 0.5, "internal_matches": [s["id"] for s in internal], "own_matches": [s["id"] for s in own]}
        if external and not internal:
            add("external_perspective", topic["id"], "Investigate perspective: " + topic["title"],
                "An external source matches the declared review target; no matching internal evidence was located in the bounded snapshot.",
                [from_ref(topic["evidence"], sources)] + [match_span(s, terms) for s in external[:2]], 3, 0.4,
                "Review the cited external perspective and search the full authorized internal corpus before proposing research.",
                "Lexical coverage is not understanding. The target may be proposed, and relevant internal material may be omitted or phrased differently.",
                "declared_topic_lexical_comparison", details)
        elif topic.get("intent", "learn") == "write" and not own:
            supporting = internal[:1] + external[:1]
            if not supporting:
                continue
            add("editorial_opportunity", topic["id"], "Review writing opportunity: " + topic["title"],
                "Evidence matches a declared writing target, but no matching own draft or published article was located in this bounded snapshot.",
                [from_ref(topic["evidence"], sources)] + [match_span(s, terms) for s in supporting], 3, 0.45,
                "Check the authorized writing inventory, intended audience and accepted content plan before making an editorial commitment.",
                "A missing sampled draft does not prove missing writing. This is a proposal, including when its target is marked accepted.",
                "declared_writing_target_comparison", details)

    published = [s for s in usable if s["kind"] == "own_published"]
    for draft in (s for s in usable if s["kind"] == "own_draft"):
        draft_words = re.findall(r"\w+", draft["text"].lower())
        draft_shingles = {tuple(draft_words[i:i + 5]) for i in range(len(draft_words) - 4)}
        for article in published:
            words = re.findall(r"\w+", article["text"].lower())
            shingles = {tuple(words[i:i + 5]) for i in range(len(words) - 4)}
            overlap = len(draft_shingles & shingles) / max(1, len(draft_shingles | shingles))
            exact = draft["sha256"] == article["sha256"]
            if exact or (len(draft_shingles) >= 10 and overlap >= 0.85):
                add("redundant_writing", [draft["id"], article["id"]], "Review draft overlap: " + draft["title"],
                    "This draft duplicates or closely overlaps the text of an own published article in the sample.",
                    [evidence(draft), evidence(article)], 4, 0.95 if exact else 0.8,
                    "Decide whether the draft is a deliberate update, syndication or redundant work; preserve both source records.",
                    "Text overlap alone cannot establish editorial redundancy, intent, originality or ownership.",
                    "exact_hash_or_five_token_shingles", {"exact_text_match": exact, "shingle_jaccard": round(overlap, 4)})

    candidates = sorted(found.values(), key=lambda item: (-item["priority"], item["kind"], item["id"]))
    result = {"schema_version": OUTPUT_VERSION, "engine_version": ENGINE_VERSION,
              "as_of": data["as_of"], "scope": data["scope"],
              "detector_config": {"stale_after_days": data["stale_after_days"]},
              "input_sha256": digest(data), "proposal_only": True, "validated_gap_count": 0,
              "admission": "input_adapter_assertion; upstream authorization and currentness must be verified by the reader",
              "coverage_claim": "bounded_observations_only; no upstream or knowledge completeness claim",
              "candidates": candidates, "queue": candidates[:limit],
              "counts": {"sources": len(sources), "searchable_sources": len(usable), "excluded": dict(sorted(exclusions.items())),
                         "candidates": len(candidates), "queued": min(limit, len(candidates)),
                         "by_kind": dict(sorted(Counter(c["kind"] for c in candidates).items()))}}
    result["snapshot_sha256"] = digest(result)
    return result


def compare(before: Any, after: Any) -> dict:
    """Compare full candidate sets; never repeat old evidence after withdrawal."""
    maps = []
    for index, value in enumerate((before, after)):
        snap = obj(value, "snapshot")
        if (snap.get("schema_version") != OUTPUT_VERSION or snap.get("proposal_only") is not True
                or type(snap.get("validated_gap_count")) is not int or snap["validated_gap_count"] != 0):
            fail("snapshot.schema")
        expected = dict(snap)
        supplied_hash = expected.pop("snapshot_sha256", None)
        if digest(expected) != supplied_hash:
            fail("snapshot.integrity")
        scope = obj(snap.get("scope"), "snapshot.scope")
        for key in ("account", "policy_revision"):
            string(scope.get(key), "snapshot.scope." + key, limit=256)
        if scope.get("tenant") is not None:
            string(scope["tenant"], "snapshot.scope.tenant", limit=256)
        oneof(scope.get("coverage"), {"bounded", "complete_for_declared_scope", "unknown"}, "snapshot.scope.coverage")
        boolean(scope.get("public_external_allowed", False), "snapshot.scope.public_external_allowed")
        timestamp(snap.get("as_of"), "snapshot.as_of")
        string(snap.get("engine_version"), "snapshot.engine_version", limit=64)
        if "detector_config" in snap:
            config = obj(snap["detector_config"], "snapshot.detector_config")
            horizon = config.get("stale_after_days")
            if set(config) != {"stale_after_days"} or type(horizon) is not int or not 1 <= horizon <= 3650:
                fail("snapshot.detector_config")
        mapping = {}
        for item in sequence(snap.get("candidates"), "snapshot.candidates", 10000):
            obj(item, "snapshot.candidate")
            if item.get("status") != "proposal_only" or item.get("validation") != "unvalidated":
                fail("snapshot.candidate.authority")
            refs = sequence(item.get("evidence"), "snapshot.candidate.evidence", MAX_ITEMS + 1)
            if not refs:
                fail("snapshot.candidate.evidence")
            for ref in refs:
                obj(ref, "snapshot.evidence")
                quote = string(ref.get("quote"), "snapshot.evidence.quote", empty=True, limit=MAX_QUOTE)
                start, end = ref.get("start"), ref.get("end")
                if (type(start) is not int or type(end) is not int or start < 0 or end < start
                        or end - start != len(quote) or text_hash(quote) != ref.get("quote_sha256")):
                    fail("snapshot.evidence.binding")
            gap_id = string(item.get("id"), "snapshot.candidate.id", limit=128)
            material = dict(item)
            material_hash = material.pop("material_sha256", None)
            if gap_id in mapping or digest(material) != material_hash:
                fail("snapshot.candidate.integrity")
            mapping[gap_id] = item
        queue = sequence(snap.get("queue"), "snapshot.queue", 5)
        if queue != snap["candidates"][:len(queue)] or (mapping and not queue):
            fail("snapshot.queue_binding")
        counts = obj(snap.get("counts"), "snapshot.counts")
        if (type(counts.get("queued")) is not int or type(counts.get("candidates")) is not int
                or counts["queued"] != len(queue) or counts["candidates"] != len(mapping)):
            fail("snapshot.counts_binding")
        maps.append(mapping)
        if index and any(before["scope"].get(k) != scope.get(k) for k in ("account", "tenant")):
            fail("snapshot.cross_scope")
    old, new = maps
    mechanism_changes = []
    for key, old_value, new_value in (
            ("engine_version", before["engine_version"], after["engine_version"]),
            ("detector_config", before.get("detector_config"), after.get("detector_config")),
            *[("scope." + key, before["scope"].get(key), after["scope"].get(key))
              for key in ("policy_revision", "coverage", "public_external_allowed")]):
        if old_value != new_value:
            mechanism_changes.append({"field": key, "before": old_value, "after": new_value})
    changes = []
    for gap_id in sorted(set(old) | set(new)):
        if gap_id not in old:
            status = "added"
        elif gap_id not in new:
            status = "no_longer_observed"
        elif old[gap_id]["material_sha256"] != new[gap_id]["material_sha256"]:
            status = "material_changed"
        else:
            continue
        changes.append({"id": gap_id, "change": status,
                        "before_material_sha256": old.get(gap_id, {}).get("material_sha256"),
                        "after_material_sha256": new.get(gap_id, {}).get("material_sha256")})
    return {"schema_version": DIFF_VERSION, "before_snapshot_sha256": before["snapshot_sha256"],
            "after_snapshot_sha256": after["snapshot_sha256"], "changes": changes,
            "mechanism_changes": mechanism_changes,
            "content_only_comparison": not bool(mechanism_changes),
            "unchanged_count": sum(gap_id in old and old[gap_id]["material_sha256"] == item["material_sha256"] for gap_id, item in new.items()),
            "interpretation": "No longer observed never means resolved. Removal can reflect withdrawal, scope, sampling or detector changes. Prior evidence is deliberately omitted."}


def evaluate(snapshot: Any, labels: Any) -> dict:
    """Evaluate explicitly judged queue items without treating unlabeled as wrong."""
    compare(snapshot, snapshot)  # Check snapshot and candidate integrity first.
    labels = obj(labels, "labels")
    if (labels.get("schema_version") != "hermes.gaps.labels.v1"
            or labels.get("snapshot_sha256") != snapshot["snapshot_sha256"]):
        fail("labels.snapshot_binding")
    label_kind = oneof(labels.get("label_kind"), {"human", "provisional", "synthetic"}, "labels.label_kind")
    candidates = {c["id"]: c for c in snapshot["candidates"]}
    judgments = {}
    for judgment in sequence(labels.get("judgments"), "labels.judgments", 10000):
        judgment = obj(judgment, "judgment")
        gap_id = string(judgment.get("gap_id"), "judgment.gap_id", limit=128)
        if gap_id not in candidates or gap_id in judgments:
            fail("judgment.candidate_binding")
        boolean(judgment.get("useful"), "judgment.useful")
        boolean(judgment.get("correct_kind"), "judgment.correct_kind")
        string(judgment.get("reviewer"), "judgment.reviewer", limit=256)
        judgments[gap_id] = judgment
    queue = snapshot["queue"]
    judged = [judgments[c["id"]] for c in queue if c["id"] in judgments]
    complete = len(judged) == len(queue)
    useful = sum(j["useful"] for j in judged)
    result = {"schema_version": "hermes.gaps.evaluation.v1", "snapshot_sha256": snapshot["snapshot_sha256"],
              "labels_sha256": digest(labels), "label_kind": label_kind, "queue_size": len(queue),
              "judged_queue_count": len(judged), "unjudged_queue_count": len(queue) - len(judged),
              "useful_precision_at_queue": useful / len(queue) if complete and queue else None,
              "useful_precision_among_judged": useful / len(judged) if judged else None,
              "correct_kind_rate_among_judged": sum(j["correct_kind"] for j in judged) / len(judged) if judged else None,
              "validated_gap_claims": snapshot["validated_gap_count"],
              "interpretation": "Labels are input assertions, not authenticated decisions. Provisional and synthetic judgments are not human acceptance; unjudged items are not failures."}
    return result
