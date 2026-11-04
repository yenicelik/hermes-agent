"""Offline geometry experiment contracts and counterexamples."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pytest

np = pytest.importorskip("numpy", reason="geometry experiments require the optional NumPy dependency")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hermes_cli.knowledge_gaps.experiments.geometry_benchmark import (  # noqa: E402
    INPUT_SCHEMA,
    InputError,
    geometry,
    run_benchmark,
)


def fixture():
    train = [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3],
             [-1, 1, 0, 0], [1, -1, 0, 0]]
    evaluation = [[1, .1, .2, .3], [.1, 1, .3, .2], [.2, .3, 1, .1], [.3, .2, .1, 1]]
    return {
        "schema_version": INPUT_SCHEMA,
        "embedding": {"model": "test-only-model", "dimension": 4, "profile_sha256": "a" * 64},
        "records": [{"id": f"t{i}", "source_group": "training-source", "split": "train", "vector": v}
                    for i, v in enumerate(train)] +
                   [{"id": f"e{i}", "source_group": f"eval-source-{i // 2}", "split": "eval", "vector": v}
                    for i, v in enumerate(evaluation)],
        "queries": [
            {"id": "q0", "source_group": "eval-source-0", "vector": [1, .11, .21, .31],
             "relevance": {"e0": 2, "e1": 1, "e2": 0, "e3": 0}, "label_kind": "human",
             "judgments_complete": True},
            {"id": "q1", "source_group": "eval-source-0", "vector": [.11, 1, .31, .21],
             "relevance": {"e1": 2}, "label_kind": "weak_adjacent", "judgments_complete": False},
        ],
        "config": {"k": [1, 3, 10], "whitening_rank": 2, "regularizer": .001},
    }


class GeometryTests(unittest.TestCase):
    def test_input_arrays_remain_unchanged_and_repeat_is_deterministic(self):
        payload = fixture()
        before = copy.deepcopy(payload)
        first = run_benchmark(payload)
        self.assertEqual(payload, before)
        self.assertEqual(first, run_benchmark(payload))
        self.assertEqual(first[0]["decision"], "experiment_only_baseline_unchanged")

    def test_fit_uses_only_train_even_when_eval_changes(self):
        first = fixture()
        second = copy.deepcopy(first)
        second["records"][-1]["vector"] = [7, 8, 2, 3]
        second["queries"][0]["vector"] = [3, 1, 2, 1]
        _, fit1 = run_benchmark(first)
        _, fit2 = run_benchmark(second)
        self.assertEqual(fit1["fit_vectors_sha256"], fit2["fit_vectors_sha256"])
        self.assertEqual(fit1["transforms"], fit2["transforms"])
        self.assertNotEqual(fit1["input_canonical_sha256"], fit2["input_canonical_sha256"])

    def test_source_group_leakage_denied(self):
        payload = fixture()
        payload["records"][-1]["source_group"] = "training-source"
        with self.assertRaisesRegex(InputError, "source_group overlap"):
            run_benchmark(payload)

    def test_query_source_group_leakage_denied(self):
        payload = fixture()
        payload["queries"][0]["source_group"] = "training-source"
        with self.assertRaisesRegex(InputError, "query source_group"):
            run_benchmark(payload)

    def test_exact_duplicate_vector_across_split_denied(self):
        payload = fixture()
        payload["records"][-1]["vector"] = payload["records"][0]["vector"][:]
        with self.assertRaisesRegex(InputError, "duplicate vector crosses"):
            run_benchmark(payload)

    def test_query_exact_train_vector_denied_even_with_another_group(self):
        payload = fixture()
        payload["queries"][0]["vector"] = payload["records"][0]["vector"][:]
        with self.assertRaisesRegex(InputError, "query exact vector occurs in train"):
            run_benchmark(payload)

    def test_query_self_retrieval_denied_unless_explicitly_excluded(self):
        payload = fixture()
        payload["queries"][0]["vector"] = payload["records"][6]["vector"][:]
        with self.assertRaisesRegex(InputError, "unexcluded eval"):
            run_benchmark(payload)
        payload["queries"][0]["exclude_ids"] = ["e0"]
        del payload["queries"][0]["relevance"]["e0"]
        report, _ = run_benchmark(payload)
        ranking = report["candidates"]["baseline_cosine"]["retrieval"]["queries"][0]["ranking"]
        self.assertNotIn("e0", {row["id"] for row in ranking})

    def test_query_ids_cannot_masquerade_as_records(self):
        payload = fixture()
        payload["queries"][0]["id"] = "e0"
        with self.assertRaisesRegex(InputError, "distinct from record ids"):
            run_benchmark(payload)

    def test_nonfinite_zero_dimension_and_boolean_vectors_rejected(self):
        for vector in ([1, 2, float("nan"), 4], [1, 2, float("inf"), 4],
                       [0, 0, 0, 0], [1, 2, 3], [True, 2, 3, 4]):
            with self.subTest(vector=vector):
                payload = fixture()
                payload["records"][0]["vector"] = vector
                with self.assertRaises(InputError):
                    run_benchmark(payload)

    def test_profile_identity_required(self):
        payload = fixture()
        payload["embedding"]["profile_sha256"] = "unknown"
        with self.assertRaisesRegex(InputError, "profile_sha256"):
            run_benchmark(payload)

    def test_joint_matrix_size_bound_precedes_vector_allocation(self):
        payload = fixture()
        payload["embedding"]["dimension"] = 16384
        # Deliberately invalid short vector bodies: the joint bound must be
        # checked first, without constructing any matrix or starting an SVD.
        payload["records"] = [payload["records"][0]] * 123
        payload["queries"] = []
        with self.assertRaisesRegex(InputError, "scalar bound"):
            run_benchmark(payload)

    def test_complete_judgments_must_cover_actual_candidates(self):
        payload = fixture()
        payload["queries"][1]["judgments_complete"] = True
        with self.assertRaisesRegex(InputError, "complete judgments"):
            run_benchmark(payload)

    def test_unknown_relevance_record_rejected(self):
        payload = fixture()
        payload["queries"][0]["relevance"]["missing"] = 2
        with self.assertRaisesRegex(InputError, "eligible"):
            run_benchmark(payload)

    def test_baseline_metrics_and_small_corpus_cutoff(self):
        report, _ = run_benchmark(fixture())
        query = report["candidates"]["baseline_cosine"]["retrieval"]["queries"][0]
        self.assertEqual(query["ranking"][0]["id"], "e0")
        self.assertEqual(query["mrr"], 1)
        self.assertEqual(query["at_k"]["1"]["recall"], .5)
        self.assertEqual(query["at_k"]["1"]["direct_evidence_recall"], 1)
        self.assertEqual(query["at_k"]["1"]["ndcg"], 1)
        self.assertEqual(query["at_k"]["10"]["effective_k"], 4)
        self.assertEqual(query["at_k"]["10"]["recall"], 1)

    def test_weak_and_human_labels_never_blended(self):
        report, _ = run_benchmark(fixture())
        retrieval = report["candidates"]["baseline_cosine"]["retrieval"]
        self.assertEqual(set(retrieval["by_label_kind"]), {"human", "weak_adjacent"})
        self.assertEqual(retrieval["by_label_kind"]["human"]["query_count"], 1)
        weak = retrieval["by_label_kind"]["weak_adjacent"]
        self.assertFalse(weak["all_judgments_complete"])
        self.assertEqual(weak["at_k"]["10"]["unjudged_count"], 3)
        self.assertTrue(any(r["grade"] is None for r in retrieval["queries"][1]["ranking"]))

    def test_no_positive_is_undefined_not_zero_success_or_failure(self):
        payload = fixture()
        payload["queries"][0]["relevance"] = {"e0": 0, "e1": 0, "e2": 0, "e3": 0}
        report, _ = run_benchmark(payload)
        scores = report["candidates"]["baseline_cosine"]["retrieval"]["by_label_kind"]["human"]
        self.assertIsNone(scores["mrr"])
        self.assertEqual(scores["mrr_denominator"], 0)
        self.assertIsNone(scores["at_k"]["1"]["ndcg"])
        self.assertIsNone(scores["at_k"]["1"]["recall"])

    def test_whitening_artifact_preserves_original_coordinates_and_matches_formula(self):
        report, fit = run_benchmark(fixture())
        transform = fit["transforms"]["regularized_whitening_then_l2"]
        self.assertEqual(transform["output_dimension"], 4)
        self.assertEqual(transform["retained_rank"], 2)
        self.assertEqual(transform["discarded_directions"], 2)
        basis = np.array(transform["basis"])
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
        mean, scales = np.array(transform["mean"]), np.array(transform["scales"])
        payload = fixture()
        docs = np.array([r["vector"] for r in payload["records"] if r["split"] == "eval"])
        query = np.array(payload["queries"][0]["vector"])
        transformed = ((docs - mean) @ basis * scales) @ basis.T
        q = ((query - mean) @ basis * scales) @ basis.T
        cosines = (transformed @ q) / np.linalg.norm(transformed, axis=1) / np.linalg.norm(q)
        ranking = report["candidates"]["regularized_whitening_then_l2"]["retrieval"]["queries"][0]["ranking"]
        for row in ranking:
            self.assertAlmostEqual(row["score"], cosines[int(row["id"][1:])])
        self.assertIn("pca_projection_then_l2", report["candidates"])

    def test_rank_is_bounded_by_samples_and_reported(self):
        payload = fixture()
        payload["records"] = payload["records"][:2] + payload["records"][6:]
        payload["config"]["whitening_rank"] = 64
        report, fit = run_benchmark(payload)
        transform = fit["transforms"]["regularized_whitening_then_l2"]
        self.assertEqual(transform["retained_rank"], 1)
        self.assertFalse(transform["full_covariance_identifiable"])
        self.assertEqual(report["counts"]["train"], 2)

    def test_repeated_train_vectors_skip_whitening_and_preserve_baseline(self):
        payload = fixture()
        for row in payload["records"][:6]:
            row["vector"] = [1, 2, 3, 4]
        report, _ = run_benchmark(payload)
        self.assertEqual(report["candidates"]["regularized_whitening_then_l2"]["reason"],
                         "zero_fit_covariance_rank")
        self.assertEqual(report["candidates"]["baseline_cosine"]["status"], "measured")
        self.assertEqual(report["counts"]["distinct_train_vectors"], 1)

    def test_zero_centered_eval_skips_entire_candidate_without_dropping_rows(self):
        payload = fixture()
        payload["records"][6]["vector"] = np.mean(
            [r["vector"] for r in payload["records"][:6]], axis=0).tolist()
        report, _ = run_benchmark(payload)
        self.assertEqual(report["candidates"]["center_then_l2"]["status"], "skipped")
        self.assertEqual(report["candidates"]["baseline_cosine"]["status"], "measured")
        self.assertEqual(report["counts"]["eval"], 4)

    def test_mean_cosine_does_not_prove_high_dimensional_isotropy(self):
        # Balanced mass on one axis has mean zero but remains rank one.
        diagnostics = geometry(np.array([[1, 0, 0], [-1, 0, 0], [2, 0, 0], [-2, 0, 0]], dtype=float))
        self.assertEqual(diagnostics["mean_vector_norm"], 0)
        self.assertEqual(diagnostics["covariance_numeric_rank"], 1)
        self.assertAlmostEqual(diagnostics["participation_ratio"], 1)
        self.assertAlmostEqual(diagnostics["top1_variance_share"], 1)

    def test_empty_queries_is_diagnostics_only(self):
        payload = fixture()
        payload["queries"] = []
        report, _ = run_benchmark(payload)
        self.assertEqual(report["candidates"]["baseline_cosine"]["retrieval"]["by_label_kind"], {})
        self.assertEqual(report["counts"]["queries"], 0)

    def test_cli_round_trip_exclusive_outputs_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, fit_path = root / "input.json", root / "result.json", root / "fit.json"
            original = json.dumps(fixture()).encode()
            source.write_bytes(original)
            command = [sys.executable, "-m", "hermes_cli.knowledge_gaps.experiments.geometry_benchmark", "--input", str(source),
                       "--output", str(output), "--fit-output", str(fit_path)]
            cwd = Path(__file__).resolve().parents[3]
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["counts"]["eval"], 4)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            second = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("must not already exist", second.stderr)


if __name__ == "__main__":
    unittest.main()
