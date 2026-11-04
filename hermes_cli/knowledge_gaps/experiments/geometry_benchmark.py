"""Bounded, offline cosine/centering/whitening comparison over cached vectors.

Run: python -m experiments.geometry_benchmark --input sample.json --output report.json
Optional --fit-output preserves fitted transforms without changing source embeddings.
No model calls, network calls, database access, or production configuration changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


INPUT_SCHEMA = "hermes.geometry-benchmark.v1"
OUTPUT_SCHEMA = "hermes.geometry-benchmark-result.v1"
FIT_SCHEMA = "hermes.geometry-fit.v1"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 4096
MAX_DIMENSION = 16384
MAX_SCALARS = 2_000_000
MAX_RANK = 64
EPS = 1e-12
LABEL_KINDS = ("human", "provisional", "weak_adjacent")


class InputError(ValueError):
    """An invalid or leakage-prone experiment input."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise InputError(f"{name} must be a nonempty bounded string")
    return value


def _vector(value: Any, dimension: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != dimension:
        raise InputError(f"{name} has an unexpected dimension")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise InputError(f"{name} must contain numbers")
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (ValueError, OverflowError) as exc:
        raise InputError(f"{name} is not representable as float64") from exc
    if not np.isfinite(array).all() or not math.isfinite(float(np.linalg.norm(array))):
        raise InputError(f"{name} must be finite")
    if float(np.linalg.norm(array)) <= EPS:
        raise InputError(f"{name} has zero norm")
    return array


def _normalize(array: np.ndarray) -> np.ndarray:
    if not np.isfinite(array).all():
        raise InputError("transformed_nonfinite_vector")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= EPS):
        raise InputError("transformed_zero_or_nonfinite_norm")
    return array / norms


def _vector_key(array: np.ndarray) -> bytes:
    # Exact numeric equality, including canonicalized signed zero. No approximate
    # vector matching can reliably establish source independence.
    copy = array.astype("<f8", copy=True)
    copy[copy == 0] = 0
    return hashlib.sha256(copy.tobytes()).digest()


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != INPUT_SCHEMA:
        raise InputError(f"schema_version must be {INPUT_SCHEMA}")
    embedding = payload.get("embedding")
    if not isinstance(embedding, dict):
        raise InputError("embedding must be an object")
    model = _string(embedding.get("model"), "embedding.model")
    dimension = embedding.get("dimension")
    if type(dimension) is not int or not 1 <= dimension <= MAX_DIMENSION:
        raise InputError("embedding.dimension is outside experiment bounds")
    profile = embedding.get("profile_sha256")
    if not isinstance(profile, str) or re.fullmatch(r"[a-f0-9]{64}", profile) is None:
        raise InputError("embedding.profile_sha256 must be a lowercase SHA256")
    rows = payload.get("records")
    queries = payload.get("queries", [])
    if not isinstance(rows, list) or not 2 <= len(rows) <= MAX_RECORDS:
        raise InputError("records must contain 2..4096 cached vectors")
    if not isinstance(queries, list) or len(queries) > 512:
        raise InputError("queries must be a list of at most 512 cases")
    # Cardinality and dimension limits alone still allow a very large dense SVD.
    # Reject before allocating even the first array; cached public samples are
    # small and this prototype is not a broad corpus analysis worker.
    if (len(rows) + len(queries)) * dimension > MAX_SCALARS:
        raise InputError("combined vector matrix exceeds 2000000 scalar bound")

    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    groups: dict[str, set[str]] = {"train": set(), "eval": set()}
    vector_splits: dict[bytes, set[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InputError("record must be an object")
        rid = _string(row.get("id"), f"records[{index}].id")
        if rid in ids:
            raise InputError("record ids must be unique")
        ids.add(rid)
        group = _string(row.get("source_group"), "record.source_group")
        split = row.get("split")
        if not isinstance(split, str) or split not in groups:
            raise InputError("record.split must be train or eval")
        vector = _vector(row.get("vector"), dimension, "record.vector")
        groups[split].add(group)
        vector_splits.setdefault(_vector_key(vector), set()).add(split)
        records.append({"id": rid, "source_group": group, "split": split, "vector": vector})
    if groups["train"] & groups["eval"]:
        raise InputError("source_group overlap between train and eval")
    if any(len(splits) > 1 for splits in vector_splits.values()):
        raise InputError("exact duplicate vector crosses train/eval split")
    records.sort(key=lambda r: r["id"])
    train = [r for r in records if r["split"] == "train"]
    evaluation = [r for r in records if r["split"] == "eval"]
    if len(evaluation) < 2:
        raise InputError("at least two eval records are required")
    eval_ids = {r["id"] for r in evaluation}
    train_vector_keys = {_vector_key(r["vector"]) for r in train}
    eval_vector_keys: dict[bytes, set[str]] = {}
    for row in evaluation:
        eval_vector_keys.setdefault(_vector_key(row["vector"]), set()).add(row["id"])
    query_ids: set[str] = set()
    parsed_queries: list[dict[str, Any]] = []
    for query in queries:
        if not isinstance(query, dict):
            raise InputError("query must be an object")
        qid = _string(query.get("id"), "query.id")
        if qid in ids or qid in query_ids:
            raise InputError("query ids must be unique and distinct from record ids")
        query_ids.add(qid)
        group = _string(query.get("source_group"), "query.source_group")
        if group in groups["train"]:
            raise InputError("query source_group occurs in train")
        vector = _vector(query.get("vector"), dimension, "query.vector")
        if _vector_key(vector) in train_vector_keys:
            raise InputError("query exact vector occurs in train")
        excluded = query.get("exclude_ids", [])
        if not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded):
            raise InputError("query.exclude_ids must be a list of record ids")
        if len(set(excluded)) != len(excluded) or not set(excluded) <= eval_ids:
            raise InputError("query.exclude_ids must contain unique eval ids")
        if eval_vector_keys.get(_vector_key(vector), set()) - set(excluded):
            raise InputError("query exact vector occurs in unexcluded eval records")
        if len(excluded) == len(evaluation):
            raise InputError("query excludes the entire eval corpus")
        relevance = query.get("relevance")
        if not isinstance(relevance, dict):
            raise InputError("query.relevance must be an object")
        if not set(relevance) <= eval_ids or set(relevance) & set(excluded):
            raise InputError("query relevance must reference eligible, unexcluded eval ids")
        if any(type(x) is not int or x not in (0, 1, 2) for x in relevance.values()):
            raise InputError("relevance grades must be integers 0, 1, or 2")
        kind = query.get("label_kind")
        if kind not in LABEL_KINDS:
            raise InputError("label_kind must be human, provisional, or weak_adjacent")
        complete = query.get("judgments_complete", False)
        if type(complete) is not bool:
            raise InputError("judgments_complete must be boolean")
        if complete and set(relevance) != eval_ids - set(excluded):
            raise InputError("complete judgments must grade every unexcluded eval record")
        parsed_queries.append({"id": qid, "source_group": group, "vector": vector,
                               "exclude_ids": set(excluded), "relevance": dict(relevance),
                               "label_kind": kind, "judgments_complete": complete})
    parsed_queries.sort(key=lambda q: q["id"])
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise InputError("config must be an object")
    k_values = config.get("k", [1, 5, 10])
    if not isinstance(k_values, list) or not k_values or len(k_values) > 10:
        raise InputError("config.k must contain 1..10 cutoffs")
    if any(type(k) is not int or not 1 <= k <= 100 for k in k_values):
        raise InputError("config.k cutoffs must be integers 1..100")
    rank = config.get("whitening_rank", 8)
    if type(rank) is not int or not 1 <= rank <= MAX_RANK:
        raise InputError("whitening_rank must be an integer 1..64")
    regularizer = config.get("regularizer", 0.001)
    if (isinstance(regularizer, bool) or not isinstance(regularizer, (int, float))
            or not math.isfinite(regularizer) or not 1e-8 <= regularizer <= 1):
        raise InputError("regularizer must be finite in [1e-8, 1]")
    return {"embedding": {"model": model, "dimension": dimension, "profile_sha256": profile},
            "train": train, "eval": evaluation, "queries": parsed_queries,
            "config": {"k": sorted(set(k_values)), "whitening_rank": rank,
                       "regularizer": float(regularizer)}}


def _matrix(rows: list[dict[str, Any]], dimension: int) -> np.ndarray:
    return np.stack([r["vector"] for r in rows]) if rows else np.empty((0, dimension))


def geometry(vectors: np.ndarray) -> dict[str, Any]:
    """Diagnostics on the evaluated representation, never a knowledge-gap score."""
    n, dimension = vectors.shape
    if n < 2:
        return {"count": n, "dimension": dimension, "status": "insufficient_rows"}
    unit = _normalize(vectors)
    centered = vectors - vectors.mean(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular * singular / (n - 1)
    total = float(eigenvalues.sum())
    if total > EPS:
        probabilities = eigenvalues[eigenvalues > EPS * max(1.0, float(eigenvalues[0]))] / total
        entropy_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
        participation = float(total * total / np.square(eigenvalues).sum())
        top_one = float(eigenvalues[0] / total)
        top_five = float(eigenvalues[:5].sum() / total)
    else:
        entropy_rank = participation = top_one = top_five = 0.0
    # The bounded dataset has <=4096 vectors. Deterministically sample at most
    # 8192 distinct unordered pairs, avoiding an n*n allocation.
    pairs: list[tuple[int, int]] = []
    target = min(8192, n * (n - 1) // 2)
    if target == n * (n - 1) // 2:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        rng = np.random.default_rng(0)
        selected: set[tuple[int, int]] = set()
        while len(selected) < target:
            i, j = (int(v) for v in rng.integers(0, n, 2))
            if i != j:
                selected.add((min(i, j), max(i, j)))
        pairs = sorted(selected)
    cosines = np.array([float(unit[i] @ unit[j]) for i, j in pairs])
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "count": n, "dimension": dimension, "status": "measured",
        "sample_rank_bound": min(dimension, n - 1),
        "covariance_numeric_rank": int(np.sum(eigenvalues > EPS * max(1.0, float(eigenvalues[0])))),
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "mean_vector_norm": float(np.linalg.norm(vectors.mean(axis=0))),
        "pair_cosine": {"count": len(pairs), "mean": float(cosines.mean()),
                        "p05": float(np.quantile(cosines, .05)),
                        "p50": float(np.quantile(cosines, .50)),
                        "p95": float(np.quantile(cosines, .95))},
        "top1_variance_share": top_one, "top5_variance_share": top_five,
        "entropy_effective_rank": entropy_rank, "participation_ratio": participation,
        "warning": "Geometry diagnostics are not retrieval quality or knowledge-gap validity.",
    }


def _dcg(grades: list[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def retrieval(evaluation: list[dict[str, Any]], queries: list[dict[str, Any]],
              documents: np.ndarray, query_vectors: np.ndarray, k_values: list[int]) -> dict[str, Any]:
    results = []
    counts = {k: {row["id"]: 0 for row in evaluation} for k in k_values}
    for query, vector in zip(queries, query_vectors):
        scores = documents @ vector
        ordered = sorted(((row["id"], float(scores[index])) for index, row in enumerate(evaluation)
                          if row["id"] not in query["exclude_ids"]), key=lambda p: (-p[1], p[0]))
        relevance = query["relevance"]
        positive = {rid for rid, grade in relevance.items() if grade > 0}
        direct = {rid for rid, grade in relevance.items() if grade == 2}
        reciprocal_rank = next((1 / (i + 1) for i, (rid, _) in enumerate(ordered) if rid in positive), 0.0)
        metrics: dict[str, Any] = {}
        for k in k_values:
            top = ordered[:k]
            for rid, _ in top:
                counts[k][rid] += 1
            top_ids = {rid for rid, _ in top}
            ideal = _dcg(sorted(relevance.values(), reverse=True)[:len(top)])
            metrics[str(k)] = {
                "effective_k": len(top),
                "recall": len(positive & top_ids) / len(positive) if positive else None,
                "direct_evidence_recall": len(direct & top_ids) / len(direct) if direct else None,
                "ndcg": _dcg([relevance.get(rid, 0) for rid, _ in top]) / ideal if ideal > 0 else None,
                "unjudged_count": sum(rid not in relevance for rid, _ in top),
            }
        results.append({"id": query["id"], "source_group": query["source_group"],
                        "label_kind": query["label_kind"],
                        "judgments_complete": query["judgments_complete"],
                        "known_positive_count": len(positive), "known_direct_evidence_count": len(direct),
                        "mrr": reciprocal_rank if positive else None, "at_k": metrics,
                        "ranking": [{"id": rid, "score": score,
                                     "grade": relevance.get(rid),
                                     "judgment": "judged" if rid in relevance else "unjudged"}
                                    for rid, score in ordered[:max(k_values)]]})
    aggregates: dict[str, Any] = {}
    for kind in LABEL_KINDS:
        selected = [q for q in results if q["label_kind"] == kind]
        if not selected:
            continue
        mr = [q["mrr"] for q in selected if q["mrr"] is not None]
        by_k: dict[str, Any] = {}
        for k in k_values:
            bucket: dict[str, Any] = {}
            for name in ("recall", "direct_evidence_recall", "ndcg"):
                values = [q["at_k"][str(k)][name] for q in selected
                          if q["at_k"][str(k)][name] is not None]
                bucket[name] = float(np.mean(values)) if values else None
                bucket[name + "_denominator"] = len(values)
            bucket["unjudged_count"] = sum(q["at_k"][str(k)]["unjudged_count"] for q in selected)
            by_k[str(k)] = bucket
        aggregates[kind] = {"query_count": len(selected), "mrr": float(np.mean(mr)) if mr else None,
                            "mrr_denominator": len(mr), "at_k": by_k,
                            "all_judgments_complete": all(q["judgments_complete"] for q in selected)}
    hubness: dict[str, Any] = {}
    for k, occurrences in counts.items():
        total = sum(occurrences.values())
        top = sorted(occurrences.items(), key=lambda p: (-p[1], p[0]))[:10]
        hubness[str(k)] = {"total_occurrences": total,
                           "max_occurrences": max(occurrences.values(), default=0),
                           "top_document_share": max(occurrences.values(), default=0) / total if total else None,
                           "occurrences": [{"id": rid, "count": count} for rid, count in top]}
    return {"queries": results, "by_label_kind": aggregates, "query_top_k_hubness": hubness,
            "metric_convention": "Recall and MRR count grades 1/2 as relevant; direct_evidence_recall uses grade 2. "
                                 "nDCG uses gain 2^grade-1. Unjudged items receive zero gain only for pooled metric "
                                 "calculation and remain explicitly unjudged, not proven irrelevant. No-positive "
                                 "queries have null relevance metrics and are excluded from those means.",
            "interpretation": "Weak adjacency measures local source proximity only; provisional labels and "
                              "incomplete pooled judgments are not independent semantic-quality validation."}


def run_benchmark(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = validate(payload)
    dimension = parsed["embedding"]["dimension"]
    train = _matrix(parsed["train"], dimension)
    evaluation = _matrix(parsed["eval"], dimension)
    queries = _matrix(parsed["queries"], dimension)
    config = parsed["config"]
    fit = {"schema_version": FIT_SCHEMA, "declared_embedding": parsed["embedding"],
           "input_canonical_sha256": _sha(payload),
           "fit_ids": [r["id"] for r in parsed["train"]],
           "fit_source_groups": sorted({r["source_group"] for r in parsed["train"]}),
           "fit_vectors_sha256": _sha(train.tolist()), "config": config,
           "transform_input": "raw_cached_vectors", "transforms": {}}
    report: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA, "input_canonical_sha256": _sha(payload),
        "declared_embedding": parsed["embedding"], "config": config,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "dtype": "float64",
                    "pair_cosine_sample_seed": 0, "pair_cosine_max_samples": 8192},
        "counts": {"train": len(train), "eval": len(evaluation), "queries": len(queries),
                   "train_source_groups": len({r["source_group"] for r in parsed["train"]}),
                   "eval_source_groups": len({r["source_group"] for r in parsed["eval"]}),
                   "distinct_train_vectors": len({_vector_key(v) for v in train}),
                   "distinct_eval_vectors": len({_vector_key(v) for v in evaluation})},
        "split": {"fit_ids": fit["fit_ids"], "fit_source_groups": fit["fit_source_groups"],
                  "eval_ids": [r["id"] for r in parsed["eval"]],
                  "eval_source_groups": sorted({r["source_group"] for r in parsed["eval"]}),
                  "query_ids": [q["id"] for q in parsed["queries"]]},
        "decision": "experiment_only_baseline_unchanged",
        "limitations": [
            "Declared model/profile and source groups are input provenance, not live attestation.",
            "Exact bounded cosine search isolates representation effects; it is not the installed HNSW/reranker path.",
            "No source capture completeness, knowledge gap, or human acceptance follows from geometry/retrieval metrics.",
            "No human-labeled holdout or release recommendation is inferred from weak/provisional labels.",
            "Raw cached vectors are copied and never overwritten; transformations are separate fitted artifacts.",
            "Rank-limited whitening also removes directions; PCA projection control separates that effect.",
            "Source-group partition and exact-duplicate checks cannot detect semantic paraphrase leakage.",
            "Tiny source-group counts are a smoke experiment, not generalization evidence.",
        ],
        "raw_geometry": geometry(evaluation), "candidates": {},
    }

    def measure(name: str, docs: np.ndarray, qvecs: np.ndarray, details: dict[str, Any]) -> None:
        try:
            normalized_docs = _normalize(docs)
            normalized_queries = _normalize(qvecs) if len(qvecs) else qvecs.copy()
            report["candidates"][name] = {
                "status": "measured", "details": details,
                "geometry": geometry(normalized_docs),
                "retrieval": retrieval(parsed["eval"], parsed["queries"], normalized_docs,
                                       normalized_queries, config["k"]),
            }
        except InputError as exc:
            # Never drop only difficult rows: a failed transform is not evaluated
            # against a smaller, artificially easier corpus.
            report["candidates"][name] = {"status": "skipped", "reason": str(exc), "details": details}

    measure("baseline_cosine", evaluation.copy(), queries.copy(),
            {"operation": "L2_then_exact_cosine", "original_dimension": dimension,
             "output_dimension": dimension, "fitted": False})
    if len(train) < 2:
        for name in ("center_then_l2", "pca_projection_then_l2", "regularized_whitening_then_l2"):
            report["candidates"][name] = {"status": "skipped", "reason": "insufficient_train_rows"}
    else:
        mean = train.mean(axis=0)
        centered_train = train - mean
        centered_eval, centered_queries = evaluation - mean, queries - mean
        fit["transforms"]["center_then_l2"] = {"mean": mean.tolist()}
        measure("center_then_l2", centered_eval, centered_queries,
                {"operation": "subtract_train_mean_then_L2", "fit_count": len(train),
                 "original_dimension": dimension, "output_dimension": dimension})
        _, singular, basis_t = np.linalg.svd(centered_train, full_matrices=False)
        eigenvalues = singular * singular / (len(train) - 1)
        tolerance = EPS * max(1.0, float(eigenvalues[0]))
        available_rank = int(np.sum(eigenvalues > tolerance))
        retained = min(config["whitening_rank"], available_rank, len(train) - 1, dimension)
        if retained == 0:
            for name in ("pca_projection_then_l2", "regularized_whitening_then_l2"):
                report["candidates"][name] = {"status": "skipped", "reason": "zero_fit_covariance_rank"}
        else:
            basis = basis_t[:retained].T.copy()
            # Canonicalize basis signs; the projection was already sign-invariant.
            for column in range(retained):
                if basis[int(np.argmax(np.abs(basis[:, column]))), column] < 0:
                    basis[:, column] *= -1
            ridge = config["regularizer"] * float(eigenvalues[:available_rank].mean())
            scales = 1 / np.sqrt(eigenvalues[:retained] + ridge)
            details = {"fit_count": len(train), "fit_covariance_rank": available_rank,
                       "requested_rank": config["whitening_rank"], "retained_rank": retained,
                       "original_dimension": dimension, "output_dimension": dimension,
                       "discarded_directions": dimension - retained,
                       "full_covariance_identifiable": len(train) > dimension and available_rank == dimension,
                       "regularizer": config["regularizer"], "ridge_absolute": ridge,
                       "regularizer_convention": "ridge = regularizer * mean(nonzero_fit_eigenvalues)",
                       "fit_rank_tolerance": tolerance}
            projected_eval, projected_queries = centered_eval @ basis, centered_queries @ basis
            measure("pca_projection_then_l2", projected_eval @ basis.T, projected_queries @ basis.T,
                    {**details, "operation": "train_mean_then_rank_projection_then_L2"})
            measure("regularized_whitening_then_l2", (projected_eval * scales) @ basis.T,
                    (projected_queries * scales) @ basis.T,
                    {**details, "operation": "train_mean_then_U_diag(1/sqrt(lambda+ridge))_U_transpose_then_L2"})
            fit["transforms"]["regularized_whitening_then_l2"] = {
                **details, "mean": mean.tolist(), "basis": basis.tolist(),
                "eigenvalues": eigenvalues[:retained].tolist(), "scales": scales.tolist(),
                "formula": "normalize(((x-mean) @ basis * scales) @ basis.T)",
            }
    baseline = report["candidates"]["baseline_cosine"]["retrieval"]["by_label_kind"]
    for name, candidate in report["candidates"].items():
        if candidate["status"] != "measured" or name == "baseline_cosine":
            continue
        delta: dict[str, Any] = {}
        for kind, summary in candidate["retrieval"]["by_label_kind"].items():
            ref = baseline[kind]
            delta[kind] = {"mrr": summary["mrr"] - ref["mrr"] if summary["mrr"] is not None else None,
                           "at_k": {}}
            for k in config["k"]:
                delta[kind]["at_k"][str(k)] = {
                    metric: summary["at_k"][str(k)][metric] - ref["at_k"][str(k)][metric]
                    if summary["at_k"][str(k)][metric] is not None else None
                    for metric in ("recall", "direct_evidence_recall", "ndcg")}
        candidate["delta_from_baseline_by_label_kind"] = delta
    report["fit_artifact_sha256"] = _sha(fit)
    return report, fit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-output", type=Path)
    args = parser.parse_args(argv)
    targets = [args.output.resolve()] + ([args.fit_output.resolve()] if args.fit_output else [])
    if args.input.resolve() in targets or len(set(targets)) != len(targets):
        parser.error("input/report/fit paths must be distinct")
    if any(path.exists() for path in targets):
        parser.error("output files must not already exist; use a new experiment artifact path")
    try:
        if args.input.stat().st_size > MAX_INPUT_BYTES:
            raise InputError("input exceeds 64 MiB bound")
        raw = args.input.read_bytes()
        payload = json.loads(raw)
        report, fit = run_benchmark(payload)
        report["input_file_sha256"] = hashlib.sha256(raw).hexdigest()
        report["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for path, value in ((args.fit_output, fit), (args.output, report)):
            if path is not None:
                # Exclusive creation prevents destroying earlier evidence. Files
                # can contain private derived data; publish only via the parent.
                import os
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "complete", "decision": report["decision"],
                          "counts": report["counts"]}, sort_keys=True))
        return 0
    except (InputError, ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
