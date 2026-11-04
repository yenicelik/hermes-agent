from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermes_cli.knowledge_gaps import InputError, analyze, compare, evaluate, validate
from hermes_cli.knowledge_gaps.core import digest, text_hash
from hermes_cli.knowledge_gaps.__main__ import load, write_immutable

ROOT = Path(__file__).resolve().parents[3]


def source(identifier, text, kind="internal", **updates):
    record = {"id": identifier, "revision": "r1", "sha256": text_hash(text),
              "uri": "https://example.test/" + identifier, "title": identifier, "kind": kind,
              "account": "example-account", "tenant": "personal", "policy_revision": "p1",
              "authorized": True, "current": True, "withdrawn": False,
              "access": "accessible", "capture": "captured", "index": "indexed",
              "observed_at": "2026-10-27T12:00:00Z", "published_at": "2026-10-26T12:00:00Z", "text": text}
    record.update(updates)
    return record


def envelope(*records, **updates):
    result = {"schema_version": "hermes.gaps.input.v1", "as_of": "2026-10-27T12:00:00Z",
              "scope": {"account": "example-account", "tenant": "personal", "policy_revision": "p1",
                        "public_external_allowed": True, "coverage": "bounded"},
              "sources": list(records)}
    result.update(updates)
    return result


def ref(record, start=0, end=None):
    return {"source_id": record["id"], "start": start, "end": len(record["text"]) if end is None else end}


def claim(identifier, record, value="1", **updates):
    result = {"id": identifier, "subject": "embedding dimension", "predicate": "equals", "value": value,
              "context": "active profile", "valid_at": "2026-10-26T12:00:00Z", "corroborated": False,
              "evidence": ref(record)}
    result.update(updates)
    return result


def direct(record):
    record.update(text_basis="direct_current_source", index="not_indexed",
                  projection={"reader": "existing-provider.read_current", "revision": record["revision"],
                              "text_sha256": record["sha256"], "checked_at": "2026-10-27T12:00:00Z"})
    return record


def rehash_snapshot(snapshot):
    for candidate in snapshot["candidates"]:
        candidate.pop("material_sha256", None)
        candidate["material_sha256"] = digest(candidate)
    snapshot.pop("snapshot_sha256", None)
    snapshot["snapshot_sha256"] = digest(snapshot)
    return snapshot


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = source("s1", "How can the retrieval baseline be evaluated?")
        self.data = envelope(self.source)

    def reject(self, data):
        with self.assertRaises(InputError):
            analyze(data)

    def test_hash_mismatch(self):
        self.source["text"] += "changed"
        self.reject(self.data)

    def test_same_source_duplicate_denied(self):
        self.data["sources"].append(copy.deepcopy(self.source))
        self.reject(self.data)

    def test_cross_account_denied(self):
        self.source["account"] = "other-private-account"
        self.reject(self.data)

    def test_cross_tenant_denied(self):
        self.source["tenant"] = "company"
        self.reject(self.data)

    def test_policy_drift_denied(self):
        self.source["policy_revision"] = "p2"
        self.reject(self.data)

    def test_public_blending_requires_explicit_flag(self):
        self.source.update(account="public", tenant=None, kind="external")
        validate(self.data)
        self.data["scope"]["public_external_allowed"] = False
        self.reject(self.data)

    def test_public_internal_source_denied(self):
        self.source.update(account="public", tenant=None)
        self.reject(self.data)

    def test_denied_or_withdrawn_body_refused(self):
        for updates in ({"authorized": False}, {"withdrawn": True}, {"access": "not_accessible"}, {"capture": "unknown"}):
            with self.subTest(updates=updates):
                data = copy.deepcopy(self.data)
                data["sources"][0].update(updates)
                self.reject(data)

    def test_future_timestamps_denied(self):
        for field in ("observed_at", "published_at"):
            data = copy.deepcopy(self.data)
            data["sources"][0][field] = "2026-10-28T00:00:00Z"
            self.reject(data)

    def test_timezone_required(self):
        self.data["as_of"] = "2026-10-27T12:00:00"
        self.reject(self.data)

    def test_observation_required_and_unknown_publication_explicit(self):
        self.source["published_at"] = None
        validate(self.data)
        del self.source["observed_at"]
        self.reject(self.data)

    def test_actual_boolean_required(self):
        self.source["authorized"] = "false"
        self.reject(self.data)

    def test_claim_wrong_quote_or_revision_denied(self):
        c = claim("c1", self.source)
        self.data["claims"] = [c]
        for key, value in (("quote", "fabricated quote"), ("revision", "old"), ("sha256", "wrong")):
            data = copy.deepcopy(self.data)
            data["claims"][0]["evidence"][key] = value
            self.reject(data)

    def test_claim_unavailable_evidence_denied(self):
        self.data["claims"] = [claim("c1", self.source)]
        self.source["index"] = "not_indexed"
        self.reject(self.data)

    def test_invalid_span_denied(self):
        for start, end in ((-1, 2), (0, 999), (2, 2), (True, 4), (0, False)):
            data = copy.deepcopy(self.data)
            c = claim("c1", self.source)
            c["evidence"].update(start=start, end=end)
            data["claims"] = [c]
            self.reject(data)

    def test_accepted_target_requires_bound_human_marker(self):
        self.source["kind"] = "plan"
        t = {"id": "t1", "title": "Retrieval", "terms": ["retrieval"], "status": "accepted", "evidence": ref(self.source)}
        self.data["topics"] = [t]
        self.reject(self.data)
        t["acceptance"] = {"actor_type": "human", "reviewer": "human-reviewer", "review_ref": "review:1", "source_id": "s1", "source_revision": "r1", "evidence_sha256": text_hash(self.source["text"])}
        validate(self.data)
        t["acceptance"]["source_revision"] = "old"
        self.reject(self.data)

    def test_source_order_is_semantically_irrelevant(self):
        self.data["sources"].append(source("s2", "A second source without questions."))
        first = analyze(self.data)
        self.data["sources"].reverse()
        self.assertEqual(first, analyze(self.data))

    def test_unvalidated_scope_metadata_is_not_reflected(self):
        self.data["scope"]["adapter_private_debug"] = "DO-NOT-REFLECT"
        self.assertNotIn("DO-NOT-REFLECT", json.dumps(analyze(self.data)))

    def test_direct_current_source_requires_exact_projection(self):
        direct(self.source)
        validate(self.data)
        for field, value in (("reader", ""), ("revision", "old"), ("text_sha256", "bad"),
                             ("checked_at", "2026-10-28T12:00:00Z")):
            data = copy.deepcopy(self.data)
            data["sources"][0]["projection"][field] = value
            self.reject(data)

    def test_direct_basis_never_bypasses_access_or_currentness(self):
        direct(self.source)
        for field, value in (("access", "not_accessible"), ("capture", "unknown"), ("authorized", False)):
            data = copy.deepcopy(self.data)
            data["sources"][0][field] = value
            self.reject(data)
        self.source["current"] = False
        self.assertEqual(analyze(self.data)["counts"]["searchable_sources"], 0)

    def test_projection_with_default_basis_is_not_silently_accepted(self):
        self.source["projection"] = {"reader": "accidental-direct-reader"}
        self.reject(self.data)


class DiscoveryTests(unittest.TestCase):
    def test_direct_api_articles_and_drafts_need_no_parallel_index(self):
        article = direct(source("rss:any-publisher/item:42", "Evaluation datasets need realistic questions.", kind="external"))
        draft = direct(source("docs:any-document-provider/item:7", "Evaluation datasets cover realistic questions.", kind="own_draft"))
        plan = source("plan:any-provider/item:6", "Write about evaluation datasets.", kind="plan")
        topic = {"id": "eval", "title": "Evaluation datasets", "terms": ["evaluation", "datasets"],
                 "status": "proposed", "intent": "write", "evidence": ref(plan)}
        out = analyze(envelope(article, draft, plan, topics=[topic]))
        self.assertEqual(out["counts"]["searchable_sources"], 3)
        self.assertEqual({c["kind"] for c in out["candidates"]}, {"source_coverage"})
        self.assertTrue(all(c["details"]["coverage_state"] == "not_in_index" for c in out["candidates"]))
        self.assertTrue(all(e["text_basis"] == "direct_current_source" and e["index_state"] == "not_indexed"
                            for c in out["candidates"] for e in c["evidence"]))

    def test_internal_topic_anchor_counts_as_real_internal_coverage(self):
        internal = source("internal", "Evaluation datasets need realistic questions.")
        article = source("article", "Evaluation datasets need realistic questions.", kind="external")
        topic = {"id": "eval", "title": "Evaluation datasets", "terms": ["evaluation", "datasets"],
                 "status": "proposed", "evidence": ref(internal)}
        out = analyze(envelope(internal, article, topics=[topic]))
        self.assertNotIn("external_perspective", {c["kind"] for c in out["candidates"]})

    def test_own_writing_anchor_counts_as_existing_writing(self):
        own = source("own", "Evaluation datasets need realistic questions.", kind="own_published")
        internal = source("internal", "Evaluation datasets need realistic questions.")
        topic = {"id": "eval", "title": "Evaluation datasets", "terms": ["evaluation", "datasets"],
                 "status": "proposed", "intent": "write", "evidence": ref(own)}
        out = analyze(envelope(own, internal, topics=[topic]))
        self.assertNotIn("editorial_opportunity", {c["kind"] for c in out["candidates"]})

    def test_timezone_equivalent_claims_compare_at_same_instant(self):
        s1, s2 = source("a", "Dimension is 1024."), source("b", "Dimension is 4096.")
        c1, c2 = claim("c1", s1, "1024"), claim("c2", s2, "4096", valid_at="2026-10-26T14:00:00+02:00")
        out = analyze(envelope(s1, s2, claims=[c1, c2]))
        self.assertIn("possible_contradiction", {c["kind"] for c in out["candidates"]})

    def test_denied_title_and_uri_never_enter_output(self):
        for updates in ({"authorized": False}, {"withdrawn": True}, {"current": False}):
            hidden = source("hidden", "", title="DO-NOT-LEAK", uri="https://private.test/DO-NOT-LEAK", **updates)
            out = analyze(envelope(hidden))
            self.assertNotIn("DO-NOT-LEAK", json.dumps(out))

    def test_sponsor_question_in_external_article_is_not_personal_question(self):
        ad = source("generic-article", "Sponsor message. How can business owners get cheaper insurance?", kind="external")
        self.assertEqual(analyze(envelope(ad))["candidates"], [])

    def test_source_states_are_distinct_not_validated_gaps(self):
        records = [source("access", "", access="not_accessible", capture="unknown", index="unknown"),
                   source("capture", "", capture="not_captured", index="unknown"),
                   source("index", "captured content", index="not_indexed"),
                   source("unknown", "", capture="unknown", index="unknown")]
        out = analyze(envelope(*records))
        states = {c["details"]["coverage_state"] for c in out["candidates"]}
        self.assertEqual(states, {"not_accessible", "not_captured", "not_in_index", "capture_unknown"})
        self.assertEqual(out["validated_gap_count"], 0)
        self.assertEqual(out["counts"]["searchable_sources"], 0)
        self.assertTrue(all(not e["quote"] for c in out["candidates"] for e in c["evidence"]))

    def test_unanswered_question_is_exactly_evidenced(self):
        s = source("s1", "Background. How can embedding retrieval benchmarks detect regressions?")
        out = analyze(envelope(s))
        candidate = out["candidates"][0]
        self.assertEqual(candidate["kind"], "unanswered_question")
        evidence = candidate["evidence"][0]
        self.assertEqual(evidence["quote"], s["text"][evidence["start"]:evidence["end"]])
        self.assertEqual(evidence["quote_sha256"], text_hash(evidence["quote"]))
        self.assertIn("paraphrased", candidate["uncertainty"])

    def test_related_text_does_not_create_unanswered_candidate(self):
        s1 = source("s1", "How can embedding retrieval benchmarks detect regressions?")
        s2 = source("s2", "Embedding retrieval benchmarks detect regressions by comparing labeled queries across versions.")
        self.assertEqual(analyze(envelope(s1, s2))["candidates"], [])

    def test_external_questions_not_treated_as_internal_unanswered(self):
        out = analyze(envelope(source("s1", "How can embedding retrieval benchmarks detect regressions?", kind="external")))
        self.assertEqual(out["candidates"], [])

    def test_claim_conflict_same_context_and_time_only(self):
        s1, s2 = source("a", "The active embedding dimension is 1024."), source("b", "The active embedding dimension is 4096.")
        c1, c2 = claim("c1", s1, "1024"), claim("c2", s2, "4096")
        data = envelope(s1, s2, claims=[c1, c2])
        self.assertIn("possible_contradiction", {c["kind"] for c in analyze(data)["candidates"]})
        for field, value in (("context", "old profile"), ("valid_at", "2026-10-25T12:00:00Z")):
            changed = copy.deepcopy(data)
            changed["claims"][1][field] = value
            self.assertNotIn("possible_contradiction", {c["kind"] for c in analyze(changed)["candidates"]})

    def test_stale_claim_is_not_marked_false(self):
        s = source("a", "Retention policy is pending.", published_at="2026-01-01T00:00:00Z")
        out = analyze(envelope(s, claims=[claim("c1", s)]))
        stale = next(c for c in out["candidates"] if c["kind"] == "stale_coverage")
        self.assertIn("not evidence", stale["uncertainty"])
        self.assertIn("weak_evidence", {c["kind"] for c in out["candidates"]})
        self.assertEqual(stale["details"]["date_basis"], "declared_publication_time")

    def test_unknown_publication_time_never_inferred_from_old_observation(self):
        for kind in ("external", "own_draft", "internal"):
            s = source("source", "Retention policy needs current support.", kind=kind,
                       published_at=None, observed_at="2026-01-01T00:00:00Z")
            out = analyze(envelope(s, claims=[claim("c1", s)]))
            self.assertNotIn("stale_coverage", {c["kind"] for c in out["candidates"]})
            self.assertIn("weak_evidence", {c["kind"] for c in out["candidates"]})

    def test_target_required_for_external_perspective(self):
        external = source("e", "Evaluation datasets need representative query diversity.", kind="external", account="public", tenant=None)
        plan = source("p", "Review representative evaluation datasets.", kind="plan")
        data = envelope(external, plan)
        self.assertEqual(analyze(data)["candidates"], [])
        data["topics"] = [{"id": "eval", "title": "Representative evaluation", "terms": ["evaluation", "datasets"], "status": "proposed", "evidence": ref(plan)}]
        out = analyze(data)
        self.assertEqual(out["candidates"][0]["kind"], "external_perspective")
        self.assertEqual(out["candidates"][0]["details"]["target_status"], "proposed")

    def test_editorial_target_is_explicit_and_existing_draft_suppresses(self):
        plan = source("p", "Write about evaluation datasets.", kind="plan")
        internal = source("i", "Our evaluation datasets measure task performance.")
        topic = {"id": "eval", "title": "Evaluation datasets", "terms": ["evaluation", "datasets"], "status": "proposed", "intent": "write", "evidence": ref(plan)}
        data = envelope(plan, internal, topics=[topic])
        self.assertEqual(analyze(data)["candidates"][0]["kind"], "editorial_opportunity")
        data["sources"].append(source("d", "Draft about evaluation datasets.", kind="own_draft"))
        self.assertNotIn("editorial_opportunity", {c["kind"] for c in analyze(data)["candidates"]})

    def test_exact_redundancy_and_no_false_external_ownership(self):
        text = "An intentionally repeated draft body."
        draft = source("d", text, kind="own_draft")
        published = source("p", text, kind="own_published")
        self.assertEqual(analyze(envelope(draft, published))["candidates"][0]["kind"], "redundant_writing")
        published["kind"] = "external"
        self.assertEqual(analyze(envelope(draft, published))["candidates"], [])

    def test_private_or_old_sources_never_enter_search_or_output(self):
        hidden = source("hidden", "SENSITIVE-PHRASE How are retrieval regressions measured?", current=False)
        out = analyze(envelope(hidden))
        self.assertEqual(out["counts"]["searchable_sources"], 0)
        self.assertNotIn("SENSITIVE-PHRASE", json.dumps(out))

    def test_top_queue_bound_full_candidates_retained(self):
        records = [source("m%d" % i, "", capture="not_captured", index="unknown") for i in range(8)]
        out = analyze(envelope(*records))
        self.assertEqual(len(out["queue"]), 5)
        self.assertEqual(len(out["candidates"]), 8)
        self.assertEqual(out, analyze(envelope(*records)))


class DiffTests(unittest.TestCase):
    def setUp(self):
        self.s = source("s", "The retrieval model has 1024 dimensions.")
        self.data = envelope(self.s, claims=[claim("c1", self.s)])

    def test_no_change(self):
        out = analyze(self.data)
        self.assertEqual(compare(out, out)["changes"], [])

    def test_same_gap_id_new_revision_material_change(self):
        before = analyze(self.data)
        self.s["revision"] = "r2"
        after = analyze(self.data)
        self.assertEqual(before["candidates"][0]["id"], after["candidates"][0]["id"])
        self.assertEqual(compare(before, after)["changes"][0]["change"], "material_changed")

    def test_withdrawal_omits_old_quotes_never_claims_resolved(self):
        before = analyze(self.data)
        self.s.update(withdrawn=True, text="", sha256=text_hash(""), revision="r2")
        self.data["claims"] = []
        diff = compare(before, analyze(self.data))
        self.assertEqual(diff["changes"][0]["change"], "no_longer_observed")
        self.assertNotIn("1024", json.dumps(diff))
        self.assertIn("never means resolved", diff["interpretation"])

    def test_diff_uses_full_set_not_top_queue(self):
        records = [source("m%d" % i, "", capture="not_captured", index="unknown") for i in range(8)]
        data = envelope(*records)
        before = analyze(data)
        excluded = next(s for s in records if s["id"] not in {c["details"]["source_id"] for c in before["queue"]})
        excluded.update(withdrawn=True)
        after = analyze(data)
        self.assertEqual(len(compare(before, after)["changes"]), 1)

    def test_snapshot_integrity_and_scope_refused(self):
        before = analyze(self.data)
        changed = copy.deepcopy(before)
        changed["candidates"][0]["reason"] = "tampered"
        with self.assertRaises(InputError):
            compare(before, changed)
        other_data = copy.deepcopy(self.data)
        other_data["scope"]["tenant"] = "other"
        other_data["sources"][0]["tenant"] = "other"
        with self.assertRaises(InputError):
            compare(before, analyze(other_data))

    def test_rehashed_queue_duplicates_and_authority_escalation_refused(self):
        original = analyze(self.data)
        changed = copy.deepcopy(original)
        changed["queue"] *= 2
        changed["counts"]["queued"] = 2
        rehash_snapshot(changed)
        with self.assertRaises(InputError):
            compare(original, changed)
        changed = copy.deepcopy(original)
        changed["candidates"][0]["validation"] = "validated"
        rehash_snapshot(changed)
        with self.assertRaises(InputError):
            compare(original, changed)

    def test_rehashed_wrong_quote_binding_refused(self):
        original = analyze(self.data)
        changed = copy.deepcopy(original)
        changed["candidates"][0]["evidence"][0]["quote"] = "different"
        rehash_snapshot(changed)
        with self.assertRaises(InputError):
            compare(original, changed)

    def test_policy_and_detector_changes_are_explicit_mechanism_changes(self):
        before = analyze(self.data)
        self.data["scope"]["policy_revision"] = "p2"
        self.s["policy_revision"] = "p2"
        self.data["stale_after_days"] = 45
        result = compare(before, analyze(self.data))
        self.assertFalse(result["content_only_comparison"])
        self.assertEqual({c["field"] for c in result["mechanism_changes"]}, {"scope.policy_revision", "detector_config"})


class FileAndCliTests(unittest.TestCase):
    def test_fifo_input_refused_without_waiting_for_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipe"
            os.mkfifo(path)
            result = subprocess.run([sys.executable, "-m", "hermes_cli.knowledge_gaps", "validate", "--input", str(path)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)

    def test_immutable_atomic_private_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_immutable(str(path), b"first")
            write_immutable(str(path), b"first")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(InputError):
                write_immutable(str(path), b"second")
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual(list(Path(directory).glob(".gap-*")), [])

    def test_symlink_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            original.write_bytes(b"body")
            link = Path(directory) / "link"
            link.symlink_to(original)
            with self.assertRaises(InputError):
                write_immutable(str(link), b"body")

    def test_duplicate_key_and_nonfinite_json_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "in.json"
            for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(InputError):
                    load(str(path))

    def test_real_cli_round_trip_and_generic_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            out = Path(directory) / "output.json"
            data = envelope(source("s", "How can retrieval benchmarks detect semantic regressions?"))
            path.write_text(json.dumps(data), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "hermes_cli.knowledge_gaps", "analyze", "--input", str(path), "--output", str(out)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["counts"]["candidates"], 1)
            diff = subprocess.run([sys.executable, "-m", "hermes_cli.knowledge_gaps", "diff", "--before", str(out), "--after", str(out)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(diff.returncode, 0, diff.stderr)
            self.assertEqual(json.loads(diff.stdout)["changes"], [])
            data["sources"][0]["text"] = "PRIVATE-DO-NOT-PRINT"
            path.write_text(json.dumps(data), encoding="utf-8")
            failed = subprocess.run([sys.executable, "-m", "hermes_cli.knowledge_gaps", "analyze", "--input", str(path)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertNotIn("PRIVATE-DO-NOT-PRINT", failed.stderr + failed.stdout)


class EvaluationTests(unittest.TestCase):
    def test_unjudged_is_not_failure_or_complete_precision(self):
        output = analyze(envelope(source("s", "How can semantic retrieval benchmarks detect regressions?")))
        labels = {"schema_version": "hermes.gaps.labels.v1", "snapshot_sha256": output["snapshot_sha256"], "label_kind": "provisional", "judgments": []}
        result = evaluate(output, labels)
        self.assertEqual(result["unjudged_queue_count"], 1)
        self.assertIsNone(result["useful_precision_at_queue"])
        labels["judgments"] = [{"gap_id": output["queue"][0]["id"], "useful": True, "correct_kind": True, "reviewer": "analyst"}]
        result = evaluate(output, labels)
        self.assertEqual(result["useful_precision_at_queue"], 1.0)
        self.assertEqual(result["label_kind"], "provisional")

    def test_label_binding_and_duplicates_refused(self):
        output = analyze(envelope(source("s", "How can semantic retrieval benchmarks detect regressions?")))
        labels = {"schema_version": "hermes.gaps.labels.v1", "snapshot_sha256": "wrong", "label_kind": "human", "judgments": []}
        with self.assertRaises(InputError):
            evaluate(output, labels)
        labels["snapshot_sha256"] = output["snapshot_sha256"]
        judgment = {"gap_id": output["queue"][0]["id"], "useful": True, "correct_kind": True, "reviewer": "human"}
        labels["judgments"] = [judgment, judgment]
        with self.assertRaises(InputError):
            evaluate(output, labels)


if __name__ == "__main__":
    unittest.main()
