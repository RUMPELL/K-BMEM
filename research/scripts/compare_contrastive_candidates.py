#!/usr/bin/env python3
"""TASK-08E: reproducible dev-only comparison of the four accepted one-epoch
contrastive candidates and provisional development-candidate selection.

Evidence synthesis only. Reads tracked summary.json / per_query.tsv files
under results/training/contrastive_v1/<model_key>/ and produces a
deterministic comparison report. No GPU, model construction, training,
inference, package mutation, checkpoint modification, or test-split access.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

EXAM_METRIC_KEYS = [
    ("exam_accuracy_diff", "exam_accuracy"),
    ("exam_mrr_diff", "exam_mrr"),
]
AIHUB_METRIC_KEYS = [
    ("aihub_ndcg_diff", "aihub_ndcg"),
    ("aihub_mrr_diff", "aihub_mrr"),
    ("aihub_recall_diff", "aihub_recall"),
]


class ValidationError(Exception):
    """Raised when an input fails a fail-closed validation check."""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValidationError(f"expected 'True'/'False' boolean cell, got {value!r}")


# ---------------------------------------------------------------------------
# Loading candidate evidence
# ---------------------------------------------------------------------------


class CandidateInputs:
    __slots__ = (
        "model_key",
        "summary_path",
        "exam_path",
        "passage_path",
        "summary",
        "exam_rows",
        "passage_rows",
        "summary_sha256",
        "exam_sha256",
        "passage_sha256",
    )

    def __init__(self, model_key: str, cfg: dict[str, Any], root: Path) -> None:
        self.model_key = model_key
        self.summary_path = root / cfg["summary_path"]
        self.exam_path = root / cfg["exam_per_query_path"]
        self.passage_path = root / cfg["passage_per_query_path"]
        self.summary = read_json(self.summary_path)
        self.exam_rows = read_tsv(self.exam_path)
        self.passage_rows = read_tsv(self.passage_path)
        self.summary_sha256 = sha256_file(self.summary_path)
        self.exam_sha256 = sha256_file(self.exam_path)
        self.passage_sha256 = sha256_file(self.passage_path)


def load_all_candidates(config: dict[str, Any], root: Path) -> dict[str, CandidateInputs]:
    out: dict[str, CandidateInputs] = {}
    for model_key, cfg in config["candidates"].items():
        out[model_key] = CandidateInputs(model_key, cfg, root)
    return out


# ---------------------------------------------------------------------------
# Validation (fail-closed)
# ---------------------------------------------------------------------------


def validate_all(config: dict[str, Any], candidates: dict[str, CandidateInputs]) -> None:
    issues: list[str] = []

    expected_keys = set(config["candidates"].keys())
    actual_keys = list(candidates.keys())
    if len(actual_keys) != len(set(actual_keys)):
        issues.append("duplicate model keys present in candidate set")
    if set(actual_keys) != expected_keys:
        issues.append(
            f"model key set mismatch: expected {sorted(expected_keys)}, got {sorted(set(actual_keys))}"
        )

    expected_digest = config["expected_batch_plan_digest"]
    expected_steps = config["expected_optimizer_steps"]
    frozen = config["frozen_contract"]

    for key, cand in candidates.items():
        s = cand.summary
        prov = s.get("provenance", {})
        digest = prov.get("batch_plan_digest")
        if digest != expected_digest:
            issues.append(f"{key}: batch_plan_digest mismatch ({digest!r} != {expected_digest!r})")

        training = s.get("training", {})
        for field in ("global_step", "planned_optimizer_steps", "logged_steps"):
            val = training.get(field)
            if val != expected_steps:
                issues.append(f"{key}: training.{field}={val!r}, expected {expected_steps}")

        fc = s.get("frozen_contract", {})
        for field, expected_val in frozen.items():
            actual_val = fc.get(field)
            if actual_val != expected_val:
                issues.append(
                    f"{key}: frozen_contract.{field}={actual_val!r}, expected {expected_val!r}"
                )

        final_export = s.get("final_export", {})
        identity = final_export.get("identity", {})
        if not identity or "identity_sha256" not in final_export:
            issues.append(f"{key}: final_export identity/linkage missing")
        dev_eval = s.get("dev_evaluation", {})
        if not dev_eval.get("command"):
            issues.append(f"{key}: dev_evaluation linkage (command) missing")

    # ID-set consistency across all four models, no duplicates/missing.
    exam_id_sets: dict[str, list[str]] = {}
    passage_id_sets: dict[str, list[str]] = {}
    for key, cand in candidates.items():
        exam_ids = [row["query_id"] for row in cand.exam_rows]
        passage_ids = [row["query_id"] for row in cand.passage_rows]
        if len(exam_ids) != len(set(exam_ids)):
            issues.append(f"{key}: duplicate exam query_id values")
        if len(passage_ids) != len(set(passage_ids)):
            issues.append(f"{key}: duplicate AIHub passage query_id values")
        exam_id_sets[key] = exam_ids
        passage_id_sets[key] = passage_ids

    if exam_id_sets:
        base_key = next(iter(exam_id_sets))
        base_exam = set(exam_id_sets[base_key])
        base_passage = set(passage_id_sets[base_key])
        for key in exam_id_sets:
            if set(exam_id_sets[key]) != base_exam:
                issues.append(f"{key}: exam question-ID set does not match {base_key}")
            if set(passage_id_sets[key]) != base_passage:
                issues.append(f"{key}: AIHub query-ID set does not match {base_key}")

    # Upskyy license label must be present, non-empty, and identical wherever
    # it is recorded in the consumed aggregate evidence.
    upskyy_key = None
    for key in config.get("ineligible_for_external_selection", []):
        if key in candidates:
            upskyy_key = key
    if upskyy_key is not None:
        required_label = config["required_upskyy_license_label"]
        s = candidates[upskyy_key].summary
        top_label = s.get("license_label")
        export_label = s.get("final_export", {}).get("identity", {}).get("license_label")
        if top_label != required_label:
            issues.append(
                f"{upskyy_key}: top-level license_label={top_label!r}, expected {required_label!r}"
            )
        if export_label != required_label:
            issues.append(
                f"{upskyy_key}: final_export.identity.license_label={export_label!r}, "
                f"expected {required_label!r}"
            )

    # Corrected Jina zero-shot reference must be used; historical erratum
    # values must never be the ones consumed.
    jina_key = config.get("jina_model_key")
    if jina_key in candidates:
        s = candidates[jina_key].summary
        exam = s.get("dev_evaluation", {}).get("exam_mcq", {})
        aihub = s.get("dev_evaluation", {}).get("ko_passage_strict", {})
        corrected_exam = exam.get("corrected_zero_shot")
        corrected_aihub = aihub.get("corrected_zero_shot")
        expected = config["jina_corrected_zero_shot"]
        forbidden = config["jina_forbidden_historical_zero_shot"]
        if not corrected_exam or not corrected_aihub:
            issues.append(f"{jina_key}: corrected_zero_shot block missing for exam or AIHub track")
        else:
            checks = [
                ("exam_accuracy_at_1", corrected_exam.get("accuracy@1")),
                ("exam_mrr", corrected_exam.get("MRR")),
                ("aihub_ndcg_at_10", corrected_aihub.get("nDCG@10")),
                ("aihub_mrr_at_10", corrected_aihub.get("MRR@10")),
                ("aihub_recall_at_20", corrected_aihub.get("Recall@20")),
            ]
            for field, actual_val in checks:
                expected_val = expected[field]
                if actual_val is None or abs(actual_val - expected_val) > 1e-9:
                    issues.append(
                        f"{jina_key}: corrected_zero_shot.{field}={actual_val!r}, "
                        f"expected {expected_val!r}"
                    )
            for field, forbidden_val in forbidden.items():
                actual_val = dict(checks).get(field)
                if actual_val is not None and abs(actual_val - forbidden_val) < 1e-9:
                    issues.append(
                        f"{jina_key}: corrected_zero_shot.{field} equals the forbidden "
                        f"historical erratum value {forbidden_val!r}"
                    )

    if issues:
        raise ValidationError("; ".join(issues))


# ---------------------------------------------------------------------------
# Per-candidate metric extraction
# ---------------------------------------------------------------------------


def exam_exact_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    correct_flags = [parse_bool(row["correct"]) for row in rows]
    reciprocal_ranks = [float(row["reciprocal_rank"]) for row in rows]
    correct_count = sum(1 for flag in correct_flags if flag)
    accuracy = correct_count / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    by_id = {
        row["query_id"]: {
            "correct": parse_bool(row["correct"]),
            "reciprocal_rank": float(row["reciprocal_rank"]),
        }
        for row in rows
    }
    return {
        "n": n,
        "correct_count": correct_count,
        "accuracy_at_1_exact": accuracy,
        "mrr_exact": mrr,
        "by_id": by_id,
    }


def passage_exact_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    ndcgs = [float(row["ndcg"]) for row in rows]
    mrrs = [float(row["mrr"]) for row in rows]
    recalls = [float(row["recall"]) for row in rows]
    by_id = {
        row["query_id"]: {
            "ndcg": float(row["ndcg"]),
            "mrr": float(row["mrr"]),
            "recall": float(row["recall"]),
        }
        for row in rows
    }
    return {
        "n": n,
        "ndcg_at_10_exact": sum(ndcgs) / n if n else 0.0,
        "mrr_at_10_exact": sum(mrrs) / n if n else 0.0,
        "recall_at_20_exact": sum(recalls) / n if n else 0.0,
        "by_id": by_id,
    }


def zero_shot_exam(summary: dict[str, Any], model_key: str, config: dict[str, Any]) -> dict[str, float]:
    exam = summary["dev_evaluation"]["exam_mcq"]
    if model_key == config.get("jina_model_key"):
        block = exam["corrected_zero_shot"]
    else:
        block = exam["zero_shot"]
    return {"accuracy_at_1": block["accuracy@1"], "mrr": block["MRR"]}


def zero_shot_aihub(summary: dict[str, Any], model_key: str, config: dict[str, Any]) -> dict[str, float]:
    aihub = summary["dev_evaluation"]["ko_passage_strict"]
    if model_key == config.get("jina_model_key"):
        block = aihub["corrected_zero_shot"]
    else:
        block = aihub["zero_shot"]
    return {
        "ndcg_at_10": block["nDCG@10"],
        "mrr_at_10": block["MRR@10"],
        "recall_at_20": block["Recall@20"],
    }


def build_candidate_report(
    model_key: str, cand: CandidateInputs, config: dict[str, Any]
) -> dict[str, Any]:
    s = cand.summary
    exam_exact = exam_exact_metrics(cand.exam_rows)
    passage_exact = passage_exact_metrics(cand.passage_rows)
    zs_exam = zero_shot_exam(s, model_key, config)
    zs_aihub = zero_shot_aihub(s, model_key, config)

    training = s.get("training", {})
    vram_note = config.get("resource_caveats", {}).get(model_key)
    vram_evidence: dict[str, Any] = {}
    if "gpu_snapshot_mid_training" in training:
        snap = training["gpu_snapshot_mid_training"]
        vram_evidence = {
            "kind": "single_sample",
            "memory_used_mib": snap.get("memory_used_mib"),
            "caveat": vram_note or snap.get("_note"),
        }
    elif "monitoring" in s:
        mon = s["monitoring"]
        vram_evidence = {
            "kind": "continuous_peak",
            "peak_vram_mib": mon.get("peak_vram_mib"),
            "total_samples": mon.get("total_samples"),
        }
    else:
        vram_evidence = {"kind": "not_recorded"}

    provenance = s.get("provenance", {})
    frozen_contract = s.get("frozen_contract", {})

    return {
        "model_id": s.get("model_id"),
        "license_label": s.get("license_label"),
        "eligible_for_external_selection": model_key
        not in config.get("ineligible_for_external_selection", []),
        "exam_dev": {
            "n": exam_exact["n"],
            "correct_count": exam_exact["correct_count"],
            "accuracy_at_1": exam_exact["accuracy_at_1_exact"],
            "mrr": exam_exact["mrr_exact"],
            "zero_shot": zs_exam,
            "delta_accuracy_at_1": exam_exact["accuracy_at_1_exact"] - zs_exam["accuracy_at_1"],
            "delta_mrr": exam_exact["mrr_exact"] - zs_exam["mrr"],
            "accuracy_ci": s["dev_evaluation"]["exam_mcq"].get("accuracy_ci"),
            "code_switch_strata": s["dev_evaluation"]["exam_mcq"].get("code_switch_strata"),
        },
        "aihub_dev": {
            "n": passage_exact["n"],
            "ndcg_at_10": passage_exact["ndcg_at_10_exact"],
            "mrr_at_10": passage_exact["mrr_at_10_exact"],
            "recall_at_20": passage_exact["recall_at_20_exact"],
            "zero_shot": zs_aihub,
            "delta_ndcg_at_10": passage_exact["ndcg_at_10_exact"] - zs_aihub["ndcg_at_10"],
            "delta_mrr_at_10": passage_exact["mrr_at_10_exact"] - zs_aihub["mrr_at_10"],
            "delta_recall_at_20": passage_exact["recall_at_20_exact"] - zs_aihub["recall_at_20"],
            "ndcg_ci": s["dev_evaluation"]["ko_passage_strict"].get("ndcg_ci"),
        },
        "resource": {
            "elapsed_seconds": training.get("elapsed_seconds"),
            "final_export_bytes": s.get("final_export", {}).get("total_bytes"),
            "checkpoint_disk_usage_bytes": training.get("checkpoint_disk_usage_bytes"),
            "vram": vram_evidence,
        },
        "operational_risk": {
            "trust_remote_code": bool(provenance.get("trust_remote_code", False)),
            "code_revision": provenance.get("code_revision"),
            "pooling": frozen_contract.get("pooling"),
            "runtime_lock_package_count": provenance.get("runtime_lock_package_count"),
        },
        "provenance": {
            "git_commit_at_start": provenance.get("git_commit")
            or provenance.get("execution_start_commit"),
            "executed_source_content_commit": provenance.get("executed_source_content_commit"),
            "accepted_task_commits": config["candidates"][model_key].get("accepted_task_commits", []),
        },
        "_exact_by_id": {
            "exam": exam_exact["by_id"],
            "aihub": passage_exact["by_id"],
        },
    }


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def apply_selection_rule(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible_keys = [k for k, r in reports.items() if r["eligible_for_external_selection"]]

    def sort_key(key: str) -> tuple[int, float, float]:
        r = reports[key]
        return (
            r["exam_dev"]["correct_count"],
            r["exam_dev"]["mrr"],
            r["aihub_dev"]["ndcg_at_10"],
        )

    ranked = sorted(eligible_keys, key=sort_key, reverse=True)

    trace: list[dict[str, Any]] = []
    trace.append(
        {
            "step": "eligibility",
            "eligible": ranked,
            "excluded": sorted(set(reports.keys()) - set(eligible_keys)),
            "reason": "Upskyy participates in internal comparison only; ineligible for release, "
            "publication, redistribution, deployment, or externally reported selection "
            "while its license remains unknown.",
        }
    )

    correct_counts = {k: reports[k]["exam_dev"]["correct_count"] for k in ranked}
    top_correct = correct_counts[ranked[0]] if ranked else None
    tied_on_correct = [k for k in ranked if correct_counts[k] == top_correct]
    trace.append(
        {
            "step": "primary_correct_count_and_accuracy",
            "correct_counts": correct_counts,
            "tied": tied_on_correct if len(tied_on_correct) > 1 else [],
        }
    )

    if len(tied_on_correct) > 1:
        mrrs = {k: reports[k]["exam_dev"]["mrr"] for k in tied_on_correct}
        top_mrr = max(mrrs.values())
        tied_on_mrr = [k for k in tied_on_correct if mrrs[k] == top_mrr]
        trace.append(
            {
                "step": "tie_break_exact_exam_mrr",
                "mrr_values": mrrs,
                "tied": tied_on_mrr if len(tied_on_mrr) > 1 else [],
            }
        )
        if len(tied_on_mrr) > 1:
            ndcgs = {k: reports[k]["aihub_dev"]["ndcg_at_10"] for k in tied_on_mrr}
            top_ndcg = max(ndcgs.values())
            tied_on_ndcg = [k for k in tied_on_mrr if ndcgs[k] == top_ndcg]
            trace.append(
                {
                    "step": "tie_break_aihub_ndcg_at_10",
                    "ndcg_values": ndcgs,
                    "tied": tied_on_ndcg if len(tied_on_ndcg) > 1 else [],
                }
            )
            if len(tied_on_ndcg) > 1:
                risk_order = sorted(
                    tied_on_ndcg,
                    key=lambda k: (
                        reports[k]["operational_risk"]["trust_remote_code"],
                        reports[k]["resource"]["elapsed_seconds"] or 0.0,
                    ),
                )
                trace.append(
                    {
                        "step": "tie_break_operational_risk",
                        "order_lowest_risk_first": risk_order,
                    }
                )

    return {
        "eligible_ranking": ranked,
        "selected_development_candidate": ranked[0] if ranked else None,
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "excluded_from_external_selection": sorted(set(reports.keys()) - set(eligible_keys)),
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Deterministic paired bootstrap and exact paired test
# ---------------------------------------------------------------------------


def paired_bootstrap_diff(
    values_a: list[float],
    values_b: list[float],
    iterations: int,
    confidence: float,
    seed_key: str,
) -> dict[str, Any]:
    n = len(values_a)
    if n == 0 or n != len(values_b):
        return {"mean_diff": 0.0, "low": 0.0, "high": 0.0, "n": n, "crosses_zero": True}
    rng = random.Random(seed_key)
    diffs: list[float] = []
    for _ in range(iterations):
        total_a = 0.0
        total_b = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            total_a += values_a[idx]
            total_b += values_b[idx]
        diffs.append((total_a - total_b) / n)
    diffs.sort()
    tail = (1.0 - confidence) / 2.0
    low = diffs[max(int(tail * iterations) - 1, 0)]
    high = diffs[min(int((1.0 - tail) * iterations), iterations - 1)]
    point_estimate = (sum(values_a) - sum(values_b)) / n
    return {
        "mean_diff": point_estimate,
        "low": low,
        "high": high,
        "n": n,
        "crosses_zero": low <= 0.0 <= high,
    }


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Exact two-sided binomial (sign) test p-value for McNemar's test."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_half_n = 0.5**n
    lower_tail = sum(math.comb(n, i) for i in range(0, k + 1)) * p_half_n
    upper_tail = sum(math.comb(n, i) for i in range(k, n + 1)) * p_half_n
    return min(1.0, 2.0 * min(lower_tail, upper_tail))


def build_pairwise(
    reports: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    keys = sorted(reports.keys())
    bootstrap_cfg = config["bootstrap"]
    iterations = int(bootstrap_cfg["iterations"])
    confidence = float(bootstrap_cfg["confidence"])
    base_seed = bootstrap_cfg["seed"]

    pairwise: dict[str, Any] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
            pair_name = f"{a_key}__vs__{b_key}"
            a_exam = reports[a_key]["_exact_by_id"]["exam"]
            b_exam = reports[b_key]["_exact_by_id"]["exam"]
            a_aihub = reports[a_key]["_exact_by_id"]["aihub"]
            b_aihub = reports[b_key]["_exact_by_id"]["aihub"]

            exam_ids = sorted(a_exam.keys())
            aihub_ids = sorted(a_aihub.keys())

            a_correct = [1.0 if a_exam[q]["correct"] else 0.0 for q in exam_ids]
            b_correct = [1.0 if b_exam[q]["correct"] else 0.0 for q in exam_ids]
            a_rr = [a_exam[q]["reciprocal_rank"] for q in exam_ids]
            b_rr = [b_exam[q]["reciprocal_rank"] for q in exam_ids]

            a_ndcg = [a_aihub[q]["ndcg"] for q in aihub_ids]
            b_ndcg = [b_aihub[q]["ndcg"] for q in aihub_ids]
            a_mrr = [a_aihub[q]["mrr"] for q in aihub_ids]
            b_mrr = [b_aihub[q]["mrr"] for q in aihub_ids]
            a_recall = [a_aihub[q]["recall"] for q in aihub_ids]
            b_recall = [b_aihub[q]["recall"] for q in aihub_ids]

            intervals = {
                "exam_accuracy_diff": paired_bootstrap_diff(
                    a_correct, b_correct, iterations, confidence, f"{base_seed}:{pair_name}:exam_accuracy"
                ),
                "exam_mrr_diff": paired_bootstrap_diff(
                    a_rr, b_rr, iterations, confidence, f"{base_seed}:{pair_name}:exam_mrr"
                ),
                "aihub_ndcg_diff": paired_bootstrap_diff(
                    a_ndcg, b_ndcg, iterations, confidence, f"{base_seed}:{pair_name}:aihub_ndcg"
                ),
                "aihub_mrr_diff": paired_bootstrap_diff(
                    a_mrr, b_mrr, iterations, confidence, f"{base_seed}:{pair_name}:aihub_mrr"
                ),
                "aihub_recall_diff": paired_bootstrap_diff(
                    a_recall, b_recall, iterations, confidence, f"{base_seed}:{pair_name}:aihub_recall"
                ),
            }

            a_correct_b_incorrect = sum(
                1 for q in exam_ids if a_exam[q]["correct"] and not b_exam[q]["correct"]
            )
            b_correct_a_incorrect = sum(
                1 for q in exam_ids if b_exam[q]["correct"] and not a_exam[q]["correct"]
            )
            p_value = exact_mcnemar_p_value(a_correct_b_incorrect, b_correct_a_incorrect)

            pairwise[pair_name] = {
                "a": a_key,
                "b": b_key,
                "bootstrap_intervals": intervals,
                "exam_discordant": {
                    f"{a_key}_correct_{b_key}_incorrect": a_correct_b_incorrect,
                    f"{b_key}_correct_{a_key}_incorrect": b_correct_a_incorrect,
                    "exact_mcnemar_two_sided_p_value": p_value,
                },
            }
    return pairwise


# ---------------------------------------------------------------------------
# Assembly and rendering
# ---------------------------------------------------------------------------


STATEMENTS = {
    "aihub_strict_passage_diagnostic": (
        "AIHub strict-passage dev nDCG@10 is a QA-pair-matching diagnostic (the corpus is the "
        "answer set itself), not open-corpus RAG evidence."
    ),
    "intervals_do_not_prove_superiority": (
        "Overlapping or zero-crossing paired 95% bootstrap intervals do not prove superiority; "
        "they identify a selection signal, not a proven ranking."
    ),
    "test_locked": "The exam and AIHub passage test splits remain locked; this comparison used dev only.",
    "upskyy_internal_only": (
        "upskyy_bge_m3_korean is internal-only and license-unknown. It participates in the "
        "internal comparison but cannot become the externally eligible release candidate, and its "
        "metrics support no publication, redistribution, deployment, or external claim."
    ),
}


def build_comparison(config: dict[str, Any], candidates: dict[str, CandidateInputs]) -> dict[str, Any]:
    validate_all(config, candidates)

    reports = {
        key: build_candidate_report(key, cand, config) for key, cand in candidates.items()
    }
    selection = apply_selection_rule(reports)
    pairwise = build_pairwise(reports, config)

    selected = selection["selected_development_candidate"]
    runner_up = selection["runner_up"]
    selected_vs_runner_up_key = None
    selection_label = None
    if selected and runner_up:
        a, b = sorted([selected, runner_up])
        selected_vs_runner_up_key = f"{a}__vs__{b}"
        crosses_zero = pairwise[selected_vs_runner_up_key]["bootstrap_intervals"][
            "exam_accuracy_diff"
        ]["crosses_zero"]
        selection_label = (
            "provisional development selection; superiority not established"
            if crosses_zero
            else "provisional development selection; interval excludes zero for the primary metric"
        )

    input_hashes = {
        key: {
            "summary_sha256": cand.summary_sha256,
            "exam_per_query_sha256": cand.exam_sha256,
            "passage_per_query_sha256": cand.passage_sha256,
            "accepted_task_commits": config["candidates"][key].get("accepted_task_commits", []),
        }
        for key, cand in candidates.items()
    }

    public_reports = {
        key: {k: v for k, v in r.items() if k != "_exact_by_id"} for key, r in reports.items()
    }

    return {
        "schema_version": 1,
        "task_id": config["task_id"],
        "expected_starting_commit": config["expected_starting_commit"],
        "batch_plan_digest": config["expected_batch_plan_digest"],
        "input_hashes": input_hashes,
        "candidates": public_reports,
        "selection": {
            **selection,
            "selected_vs_runner_up_pair": selected_vs_runner_up_key,
            "selection_label": selection_label,
        },
        "pairwise": pairwise,
        "statements": STATEMENTS,
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(comparison: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# TASK-08E — 4개 대조학습 후보 비교 및 개발 후보 선정")
    lines.append("")
    lines.append(f"- 작업: {comparison['task_id']}")
    lines.append(f"- 시작 커밋: `{comparison['expected_starting_commit']}`")
    lines.append(f"- 배치 다이제스트: `{comparison['batch_plan_digest']}`")
    lines.append("")
    lines.append(
        "본 문서는 학습·추론·GPU 사용 없이, 이미 승인된 4개 후보의 dev 평가 산출물만으로 "
        "재현 가능하게 생성되었습니다."
    )
    lines.append("")

    lines.append("## 1. 4-모델 지표 표 (dev)")
    lines.append("")
    lines.append(
        "| 모델 | exam n | exam 정답수 | exam acc@1 | exam MRR | exam acc@1 Δ(zero-shot) | "
        "AIHub nDCG@10 | AIHub nDCG@10 Δ | 선정 자격 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for key in sorted(comparison["candidates"].keys()):
        c = comparison["candidates"][key]
        e = c["exam_dev"]
        a = c["aihub_dev"]
        eligible = "예" if c["eligible_for_external_selection"] else "아니오(내부 전용)"
        lines.append(
            f"| {key} | {e['n']} | {e['correct_count']} | {format_pct(e['accuracy_at_1'])} | "
            f"{e['mrr']:.4f} | {e['delta_accuracy_at_1']:+.4f} | {format_pct(a['ndcg_at_10'])} | "
            f"{a['delta_ndcg_at_10']:+.4f} | {eligible} |"
        )
    lines.append("")

    lines.append("## 2. 선정 규칙 적용 경로")
    lines.append("")
    for step in comparison["selection"]["trace"]:
        lines.append(f"- **{step['step']}**: {json.dumps(step, ensure_ascii=False)}")
    lines.append("")
    lines.append(f"- 선정된 개발 후보: **{comparison['selection']['selected_development_candidate']}**")
    lines.append(f"- 차점 후보: **{comparison['selection']['runner_up']}**")
    lines.append(
        f"- 대외 선정 제외: {', '.join(comparison['selection']['excluded_from_external_selection']) or '없음'}"
    )
    lines.append(f"- 선정 라벨: {comparison['selection']['selection_label']}")
    lines.append("")

    lines.append("## 3. 쌍별(pairwise) 비교 — 6쌍")
    lines.append("")
    lines.append("| 쌍 | exam acc@1 Δ (95% CI) | 교차 0 | AIHub nDCG@10 Δ (95% CI) | 교차 0 | McNemar p |")
    lines.append("|---|---|---|---|---|---:|")
    for pair_name in sorted(comparison["pairwise"].keys()):
        p = comparison["pairwise"][pair_name]
        acc = p["bootstrap_intervals"]["exam_accuracy_diff"]
        ndcg = p["bootstrap_intervals"]["aihub_ndcg_diff"]
        mcnemar_p = p["exam_discordant"]["exact_mcnemar_two_sided_p_value"]
        lines.append(
            f"| {p['a']} vs {p['b']} | {acc['mean_diff']:+.4f} "
            f"[{acc['low']:+.4f}, {acc['high']:+.4f}] | {'예' if acc['crosses_zero'] else '아니오'} | "
            f"{ndcg['mean_diff']:+.4f} [{ndcg['low']:+.4f}, {ndcg['high']:+.4f}] | "
            f"{'예' if ndcg['crosses_zero'] else '아니오'} | {mcnemar_p:.4f} |"
        )
    lines.append("")

    lines.append("## 4. 혼용(code-switch) 층화 — exam MCQ, 임계 5/10/20")
    lines.append("")
    for threshold in (5, 10, 20):
        lines.append(f"### 임계 {threshold}")
        lines.append("")
        lines.append("| 모델 | ko acc@1 (n) | mixed acc@1 (n) | drop |")
        lines.append("|---|---|---|---:|")
        for key in sorted(comparison["candidates"].keys()):
            strata = comparison["candidates"][key]["exam_dev"].get("code_switch_strata") or {}
            block = strata.get(f"code_switch@{threshold}")
            if not block:
                lines.append(f"| {key} | - | - | - |")
                continue
            lines.append(
                f"| {key} | {format_pct(block['ko'])} (n={block['ko_n']}) | "
                f"{format_pct(block['mixed'])} (n={block['mixed_n']}) | {block['drop']:+.4f} |"
            )
        lines.append("")

    lines.append("## 5. 학습 시간·체크포인트 저장·VRAM 증거")
    lines.append("")
    lines.append("| 모델 | 학습 시간(초) | 최종 export 크기(byte) | VRAM 근거 |")
    lines.append("|---|---:|---:|---|")
    for key in sorted(comparison["candidates"].keys()):
        r = comparison["candidates"][key]["resource"]
        vram = r["vram"]
        if vram["kind"] == "single_sample":
            vram_desc = f"단일 샘플 {vram.get('memory_used_mib')} MiB — 실제 peak 아님, {vram.get('caveat')}"
        elif vram["kind"] == "continuous_peak":
            vram_desc = f"연속 모니터링 peak {vram.get('peak_vram_mib')} MiB (표본 {vram.get('total_samples')}개)"
        else:
            vram_desc = "기록 없음"
        lines.append(
            f"| {key} | {r['elapsed_seconds']} | {r['final_export_bytes']} | {vram_desc} |"
        )
    lines.append("")
    lines.append(
        "**주의**: TASK-08C(BAAI)의 VRAM 값은 학습 중 단일 `nvidia-smi` 스냅샷이며 연속 모니터링 "
        "peak가 아닙니다. BAAI의 자원 효율성을 실제 peak인 것처럼 순위화하지 않습니다."
    )
    lines.append("")

    lines.append("## 6. 신뢰 코드·런타임·라이선스·운영 리스크")
    lines.append("")
    lines.append("| 모델 | trust_remote_code | pooling | 라이선스 라벨 |")
    lines.append("|---|---|---|---|")
    for key in sorted(comparison["candidates"].keys()):
        c = comparison["candidates"][key]
        risk = c["operational_risk"]
        license_display = c["license_label"] if c["license_label"] else "—"
        lines.append(
            f"| {key} | {risk['trust_remote_code']} | {risk['pooling']} | {license_display} |"
        )
    lines.append("")

    lines.append("## 7. 필수 명시 사항")
    lines.append("")
    for text in comparison["statements"].values():
        lines.append(f"- {text}")
    lines.append("")

    lines.append("## 8. 입력 해시 및 생산 커밋")
    lines.append("")
    lines.append("| 모델 | summary.json sha256 | exam per_query sha256 | passage per_query sha256 |")
    lines.append("|---|---|---|---|")
    for key in sorted(comparison["input_hashes"].keys()):
        h = comparison["input_hashes"][key]
        lines.append(
            f"| {key} | `{h['summary_sha256']}` | `{h['exam_per_query_sha256']}` | "
            f"`{h['passage_per_query_sha256']}` |"
        )

    return "\n".join(lines) + "\n"


def render_pairwise_csv(comparison: dict[str, Any]) -> str:
    fieldnames = [
        "pair",
        "a",
        "b",
        "exam_accuracy_diff_mean",
        "exam_accuracy_diff_low",
        "exam_accuracy_diff_high",
        "exam_accuracy_diff_crosses_zero",
        "exam_mrr_diff_mean",
        "exam_mrr_diff_low",
        "exam_mrr_diff_high",
        "exam_mrr_diff_crosses_zero",
        "aihub_ndcg_diff_mean",
        "aihub_ndcg_diff_low",
        "aihub_ndcg_diff_high",
        "aihub_ndcg_diff_crosses_zero",
        "aihub_mrr_diff_mean",
        "aihub_mrr_diff_low",
        "aihub_mrr_diff_high",
        "aihub_mrr_diff_crosses_zero",
        "aihub_recall_diff_mean",
        "aihub_recall_diff_low",
        "aihub_recall_diff_high",
        "aihub_recall_diff_crosses_zero",
        "exam_discordant_a_correct_b_incorrect",
        "exam_discordant_b_correct_a_incorrect",
        "exam_mcnemar_two_sided_p_value",
    ]
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for pair_name in sorted(comparison["pairwise"].keys()):
        p = comparison["pairwise"][pair_name]
        bi = p["bootstrap_intervals"]
        disc = p["exam_discordant"]
        a_key, b_key = p["a"], p["b"]
        row = {
            "pair": pair_name,
            "a": a_key,
            "b": b_key,
            "exam_accuracy_diff_mean": bi["exam_accuracy_diff"]["mean_diff"],
            "exam_accuracy_diff_low": bi["exam_accuracy_diff"]["low"],
            "exam_accuracy_diff_high": bi["exam_accuracy_diff"]["high"],
            "exam_accuracy_diff_crosses_zero": bi["exam_accuracy_diff"]["crosses_zero"],
            "exam_mrr_diff_mean": bi["exam_mrr_diff"]["mean_diff"],
            "exam_mrr_diff_low": bi["exam_mrr_diff"]["low"],
            "exam_mrr_diff_high": bi["exam_mrr_diff"]["high"],
            "exam_mrr_diff_crosses_zero": bi["exam_mrr_diff"]["crosses_zero"],
            "aihub_ndcg_diff_mean": bi["aihub_ndcg_diff"]["mean_diff"],
            "aihub_ndcg_diff_low": bi["aihub_ndcg_diff"]["low"],
            "aihub_ndcg_diff_high": bi["aihub_ndcg_diff"]["high"],
            "aihub_ndcg_diff_crosses_zero": bi["aihub_ndcg_diff"]["crosses_zero"],
            "aihub_mrr_diff_mean": bi["aihub_mrr_diff"]["mean_diff"],
            "aihub_mrr_diff_low": bi["aihub_mrr_diff"]["low"],
            "aihub_mrr_diff_high": bi["aihub_mrr_diff"]["high"],
            "aihub_mrr_diff_crosses_zero": bi["aihub_mrr_diff"]["crosses_zero"],
            "aihub_recall_diff_mean": bi["aihub_recall_diff"]["mean_diff"],
            "aihub_recall_diff_low": bi["aihub_recall_diff"]["low"],
            "aihub_recall_diff_high": bi["aihub_recall_diff"]["high"],
            "aihub_recall_diff_crosses_zero": bi["aihub_recall_diff"]["crosses_zero"],
            "exam_discordant_a_correct_b_incorrect": disc[f"{a_key}_correct_{b_key}_incorrect"],
            "exam_discordant_b_correct_a_incorrect": disc[f"{b_key}_correct_{a_key}_incorrect"],
            "exam_mcnemar_two_sided_p_value": disc["exact_mcnemar_two_sided_p_value"],
        }
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "compare_contrastive_task08e.json",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run validation and computation without writing output files.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    candidates = load_all_candidates(config, args.root)

    try:
        comparison = build_comparison(config, candidates)
    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2

    if args.check_only:
        print("validation and computation succeeded", file=sys.stderr)
        return 0

    output_cfg = config["output"]
    json_path = args.root / output_cfg["json_path"]
    md_path = args.root / output_cfg["markdown_path"]
    csv_path = args.root / output_cfg["pairwise_csv_path"]

    json_text = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")

    md_text = render_markdown(comparison)
    md_path.write_text(md_text, encoding="utf-8")

    csv_text = render_pairwise_csv(comparison)
    csv_path.write_text(csv_text, encoding="utf-8")

    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {md_path}", file=sys.stderr)
    print(f"wrote {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
