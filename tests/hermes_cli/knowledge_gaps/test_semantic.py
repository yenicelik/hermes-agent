"""Citation admission is mechanical and never semantic or human authority."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermes_cli.knowledge_gaps import core
from hermes_cli.knowledge_gaps.semantic import compare_semantic, validate_comparisons


def fixture():
    scope = {"account": "owner", "tenant": "personal", "policy_revision": "policy-v1",
             "public_external_allowed": False, "coverage": "bounded"}
    rows = []
    for sid, kind, text in [("i", "internal", "We have no measured deployment success rate."),
                            ("e", "external", "The author describes reliable agent deployments."),
                            ("w", "own_draft", "Our draft says autonomous systems are reliable.")]:
        rows.append({"id": sid, "kind": kind, "text": text, "revision": "r1", "sha256": core.text_hash(text),
                     "account": "owner", "tenant": "personal", "policy_revision": "policy-v1",
                     "authorized": True, "current": True, "withdrawn": False, "access": "accessible",
                     "capture": "captured", "index": "indexed", "title": sid, "uri": "https://example.org/" + sid,
                     "observed_at": "2026-11-01T00:00:00Z", "published_at": "2026-11-01T00:00:00Z"})
    envelope = {"schema_version": core.INPUT_VERSION, "as_of": "2026-11-03T00:00:00Z", "scope": scope,
                "sources": rows}
    refs = [{"source_id": r["id"], "start": 0, "end": len(r["text"]), "quote": r["text"],
             "revision": r["revision"], "sha256": r["sha256"]} for r in rows]
    proposal = {"id": "deployment-evidence", "kind": "evidence_gap", "title": "Measure deployment reliability",
                "observation": "The draft's claim lacks a measured local baseline in this sample.",
                "why_it_matters": "A measured case would support the intended article.",
                "next_step": "Propose a bounded experiment for review.",
                "uncertainty": "Only three records were inspected; evidence may exist elsewhere.",
                "impact": 4, "basis": "agent_review", "evidence": refs}
    return envelope, [proposal]


class SemanticTests(unittest.TestCase):
    def test_deterministic_cross_source_output_does_not_mutate_inputs(self):
        envelope, proposals = fixture(); original = copy.deepcopy((envelope, proposals))
        report = validate_comparisons(envelope, proposals)
        self.assertEqual(report, validate_comparisons(envelope, proposals))
        self.assertEqual((envelope, proposals), original)
        self.assertEqual(report["queue"][0]["status"], "proposal_only")
        self.assertEqual(report["queue"][0]["validation"], "unvalidated")
        self.assertEqual({e["source_kind"] for e in report["queue"][0]["evidence"]},
                         {"internal", "external", "own_draft"})

    def test_hallucinated_quote_revision_hash_source_and_span_refused(self):
        for field, value in [("quote", "invented"), ("revision", "r2"), ("sha256", "0" * 64),
                             ("source_id", "missing"), ("start", -1), ("end", 9999), ("start", True)]:
            with self.subTest(field=field):
                envelope, proposals = fixture(); proposals[0]["evidence"][0][field] = value
                with self.assertRaises(core.InputError):
                    validate_comparisons(envelope, proposals)

    def test_all_citation_bindings_are_mandatory(self):
        for field in ("quote", "revision", "sha256"):
            envelope, proposals = fixture(); del proposals[0]["evidence"][0][field]
            with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)

    def test_cross_scope_and_policy_drift_refused(self):
        for field, value in [("account", "another"), ("tenant", "company"), ("policy_revision", "p2")]:
            envelope, proposals = fixture(); envelope["sources"][0][field] = value
            with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)

    def test_same_kind_or_single_source_cannot_fabricate_comparison(self):
        envelope, proposals = fixture()
        for source in envelope["sources"]:source["kind"] = "internal"
        with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)
        envelope, proposals = fixture(); proposals[0]["evidence"] = proposals[0]["evidence"][:1]
        with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)

    def test_noncurrent_withdrawn_and_denied_sources_cannot_support_proposal(self):
        for field, value in [("current", False), ("withdrawn", True), ("authorized", False),
                             ("access", "not_accessible"), ("capture", "not_captured"), ("index", "not_indexed")]:
            envelope, proposals = fixture(); envelope["sources"][0][field] = value
            if field in {"withdrawn", "authorized", "access", "capture"}:
                envelope["sources"][0]["text"] = ""; envelope["sources"][0]["sha256"] = core.text_hash("")
            with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)

    def test_source_content_hash_drift_refused_before_review(self):
        envelope, proposals = fixture(); envelope["sources"][0]["text"] += "changed"
        with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)

    def test_unknown_fields_categories_and_authority_are_refused(self):
        for key, value in [("accepted", True), ("status", "accepted"), ("execute", {"tool": "publish"})]:
            envelope, proposals = fixture(); proposals[0][key] = value
            with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)
        for key, value in [("kind", "validated_gap"), ("basis", "human_review"), ("impact", True),
                           ("impact", 6), ("title", "x" * 241), ("uncertainty", "")]:
            envelope, proposals = fixture(); proposals[0][key] = value
            with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)

    def test_duplicate_proposals_and_spans_refused(self):
        envelope, proposals = fixture(); proposals.append(copy.deepcopy(proposals[0]))
        with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)
        envelope, proposals = fixture(); proposals[0]["evidence"].append(copy.deepcopy(proposals[0]["evidence"][0]))
        with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals)

    def test_queue_cap_diff_full_sets_and_material_change(self):
        envelope, proposals = fixture()
        proposals = [dict(proposals[0], id="proposal-" + str(i)) for i in range(8)]
        before = validate_comparisons(envelope, proposals)
        self.assertEqual(before["counts"], {"candidates": 8, "queued": 5})
        after = validate_comparisons(envelope, proposals[:7])
        self.assertEqual(len(compare_semantic(before, after)["no_longer_observed"]), 1)
        changed = copy.deepcopy(proposals); changed[0]["next_step"] = "Propose another bounded check."
        delta = compare_semantic(before, validate_comparisons(envelope, changed))
        self.assertEqual(len(delta["material_changed"]), 1)
        self.assertEqual(len(delta["unchanged"]), 7)
        for limit in (0, 6, True):
            with self.assertRaises(core.InputError):validate_comparisons(envelope, proposals, limit=limit)

    def test_withdrawal_diff_contains_no_old_quotes_or_claims(self):
        envelope, proposals = fixture(); before = validate_comparisons(envelope, proposals)
        envelope["sources"][0].update(withdrawn=True, text="", sha256=core.text_hash(""))
        after = validate_comparisons(envelope, [])
        delta = compare_semantic(before, after); encoded = json.dumps(delta)
        self.assertEqual(len(delta["no_longer_observed"]), 1)
        self.assertNotIn("Our draft", encoded); self.assertNotIn("Measure deployment", encoded)
        self.assertIn("does not mean resolved", delta["interpretation"])

    def test_diff_refuses_corruption_cross_scope_and_backwards_time(self):
        envelope, proposals = fixture(); before = validate_comparisons(envelope, proposals)
        altered = copy.deepcopy(before); altered["queue"][0]["title"] = "tampered"
        with self.assertRaises(core.InputError):compare_semantic(before, altered)
        other = copy.deepcopy(envelope); other["scope"]["account"] = "other"
        for source in other["sources"]:source["account"] = "other"
        with self.assertRaises(core.InputError):compare_semantic(before, validate_comparisons(other, proposals))
        other = copy.deepcopy(envelope); other["as_of"] = "2026-11-02T00:00:00Z"
        with self.assertRaises(core.InputError):compare_semantic(before, validate_comparisons(other, proposals))

    def test_error_does_not_reflect_source_text(self):
        envelope, proposals = fixture(); proposals[0]["evidence"][0]["quote"] = "private sentinel 7183"
        try:validate_comparisons(envelope, proposals)
        except core.InputError as error:self.assertNotIn("private sentinel", str(error))
        else:self.fail("bad evidence accepted")

    def test_source_kind_agnostic_and_new_observation_time_is_not_material_change(self):
        envelope, proposals = fixture()
        envelope["sources"][1].update(title="A technical standard", uri="https://example.org/standard")
        envelope["sources"][2]["kind"] = "own_published"
        before = validate_comparisons(envelope, proposals)
        envelope["as_of"] = "2026-11-04T00:00:00Z"
        after = validate_comparisons(envelope, proposals)
        delta = compare_semantic(before, after)
        self.assertEqual(len(delta["unchanged"]), 1)
        self.assertEqual(delta["material_changed"], [])
        self.assertNotEqual(before["snapshot_sha256"], after["snapshot_sha256"])


if __name__ == "__main__":unittest.main()
