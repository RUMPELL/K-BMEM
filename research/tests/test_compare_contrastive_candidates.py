import copy
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "compare_contrastive_candidates.py"
SPEC = importlib.util.spec_from_file_location("compare_contrastive_candidates", MODULE_PATH)
CMP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CMP
SPEC.loader.exec_module(CMP)


BATCH_DIGEST = "28809c25fb0898a6a682c9bcfb0f7a9654e4291c9f0c3aed3ebc0fafe554e708"
STEPS = 4
EXAM_IDS = ["examq_1", "examq_2", "examq_3", "examq_4"]
AIHUB_IDS = ["qa_1", "qa_2", "qa_3"]
LICENSE_LABEL = "LICENSE UNKNOWN — INTERNAL EXPERIMENT ONLY"

JINA_CORRECTED = {
    "exam_accuracy_at_1": 0.25,
    "exam_mrr": 0.5,
    "aihub_ndcg_at_10": 0.7,
    "aihub_mrr_at_10": 0.6,
    "aihub_recall_at_20": 0.9,
}
JINA_HISTORICAL_FORBIDDEN = {
    "exam_accuracy_at_1": 0.9999,
    "aihub_ndcg_at_10": 0.9998,
}

# Per-model exam correctness pattern: (correct flags, reciprocal ranks)
EXAM_PATTERNS = {
    "kure_v1": ([True, True, False, False], [1.0, 1.0, 0.2, 0.25]),
    "baai_bge_m3": ([True, True, False, False], [1.0, 0.5, 0.2, 0.25]),
    "jina_embeddings_v3": ([True, False, False, False], [1.0, 0.2, 0.2, 0.25]),
    "upskyy_bge_m3_korean": ([True, True, True, False], [1.0, 1.0, 1.0, 0.25]),
}
AIHUB_PATTERNS = {
    "kure_v1": ([0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [1.0, 1.0, 1.0]),
    "baai_bge_m3": ([0.85, 0.75, 0.65], [0.85, 0.75, 0.65], [1.0, 1.0, 1.0]),
    "jina_embeddings_v3": ([0.6, 0.5, 0.4], [0.6, 0.5, 0.4], [1.0, 1.0, 1.0]),
    "upskyy_bge_m3_korean": ([0.95, 0.9, 0.85], [0.95, 0.9, 0.85], [1.0, 1.0, 1.0]),
}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_exam_tsv(path: Path, ids, correct_flags, reciprocal_ranks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "correct", "rank", "reciprocal_rank", "longest_option_hit", "random_hit"])
        for qid, correct, rr in zip(ids, correct_flags, reciprocal_ranks):
            rank = int(round(1.0 / rr)) if rr else 99
            writer.writerow([qid, str(correct), rank, rr, "1.0", "0.2"])


def _write_passage_tsv(path: Path, ids, ndcgs, mrrs, recalls) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "ndcg", "mrr", "recall", "unjudged_in_top"])
        for qid, ndcg, mrr, recall in zip(ids, ndcgs, mrrs, recalls):
            writer.writerow([qid, ndcg, mrr, recall, 0])


def _base_summary(model_key: str, model_id: str) -> dict:
    correct_flags, rrs = EXAM_PATTERNS[model_key]
    n_exam = len(correct_flags)
    accuracy = sum(correct_flags) / n_exam
    mrr = sum(rrs) / n_exam

    ndcgs, mrrs_p, recalls = AIHUB_PATTERNS[model_key]
    n_aihub = len(ndcgs)

    summary = {
        "task_id": "TASK-TEST",
        "model_key": model_key,
        "model_id": model_id,
        "license_label": None,
        "provenance": {
            "git_commit": "0" * 40,
            "batch_plan_digest": BATCH_DIGEST,
            "trust_remote_code": False,
        },
        "frozen_contract": {
            "objective": "multiple_negatives_ranking_loss",
            "precision": "bf16",
            "max_seq_length_applied": 512,
            "seed": 20260810,
            "epochs": 1,
            "pooling": "cls",
        },
        "training": {
            "global_step": STEPS,
            "planned_optimizer_steps": STEPS,
            "logged_steps": STEPS,
            "elapsed_seconds": 100.0,
        },
        "final_export": {
            "total_bytes": 12345,
            "identity_sha256": "a" * 64,
            "identity": {"model_key": model_key},
        },
        "dev_evaluation": {
            "command": "frozen-python scripts/evaluate_retrieval.py --config x.json",
            "exam_mcq": {
                "n": n_exam,
                "accuracy@1": accuracy,
                "MRR": mrr,
                "zero_shot": {"accuracy@1": 0.2, "MRR": 0.4},
                "code_switch_strata": {
                    "code_switch@5": {"ko": 0.3, "ko_n": 2, "mixed": 0.2, "mixed_n": 2, "drop": 0.1}
                },
            },
            "ko_passage_strict": {
                "n": n_aihub,
                "nDCG@10": sum(ndcgs) / n_aihub,
                "MRR@10": sum(mrrs_p) / n_aihub,
                "Recall@20": sum(recalls) / n_aihub,
                "zero_shot": {"nDCG@10": 0.5, "MRR@10": 0.4, "Recall@20": 0.8},
            },
            "locked_splits": ["test"],
        },
    }

    if model_key == "jina_embeddings_v3":
        summary["dev_evaluation"]["exam_mcq"]["corrected_zero_shot"] = {
            "accuracy@1": JINA_CORRECTED["exam_accuracy_at_1"],
            "MRR": JINA_CORRECTED["exam_mrr"],
        }
        summary["dev_evaluation"]["exam_mcq"]["_historical_zero_shot_for_reference_only"] = {
            "accuracy@1": JINA_HISTORICAL_FORBIDDEN["exam_accuracy_at_1"],
            "MRR": 0.6,
        }
        del summary["dev_evaluation"]["exam_mcq"]["zero_shot"]
        summary["dev_evaluation"]["ko_passage_strict"]["corrected_zero_shot"] = {
            "nDCG@10": JINA_CORRECTED["aihub_ndcg_at_10"],
            "MRR@10": JINA_CORRECTED["aihub_mrr_at_10"],
            "Recall@20": JINA_CORRECTED["aihub_recall_at_20"],
        }
        del summary["dev_evaluation"]["ko_passage_strict"]["zero_shot"]
        summary["provenance"]["trust_remote_code"] = True
        summary["frozen_contract"]["pooling"] = "mean"

    if model_key == "baai_bge_m3":
        summary["training"]["gpu_snapshot_mid_training"] = {
            "memory_used_mib": 7157,
            "_note": "Single nvidia-smi snapshot, not a continuous-polling peak.",
        }

    if model_key == "upskyy_bge_m3_korean":
        summary["license_label"] = LICENSE_LABEL
        summary["final_export"]["identity"]["license_label"] = LICENSE_LABEL
        summary["monitoring"] = {"peak_vram_mib": 7369, "total_samples": 45}
        summary["frozen_contract"]["pooling"] = "mean"

    return summary


MODEL_IDS = {
    "kure_v1": "nlpai-lab/KURE-v1",
    "jina_embeddings_v3": "jinaai/jina-embeddings-v3",
    "baai_bge_m3": "BAAI/bge-m3",
    "upskyy_bge_m3_korean": "upskyy/bge-m3-korean",
}


def build_fixture_tree(root: Path) -> dict:
    """Writes a synthetic 4-candidate results tree and returns a matching config dict."""
    config = {
        "task_id": "TASK-TEST",
        "expected_starting_commit": "0" * 40,
        "expected_batch_plan_digest": BATCH_DIGEST,
        "expected_optimizer_steps": STEPS,
        "frozen_contract": {
            "seed": 20260810,
            "objective": "multiple_negatives_ranking_loss",
            "epochs": 1,
            "precision": "bf16",
            "max_seq_length_applied": 512,
        },
        "candidates": {},
        "ineligible_for_external_selection": ["upskyy_bge_m3_korean"],
        "required_upskyy_license_label": LICENSE_LABEL,
        "jina_model_key": "jina_embeddings_v3",
        "jina_corrected_zero_shot": JINA_CORRECTED,
        "jina_forbidden_historical_zero_shot": JINA_HISTORICAL_FORBIDDEN,
        "code_switch_thresholds": [5, 10, 20],
        "bootstrap": {"seed": 20260810, "iterations": 200, "confidence": 0.95},
        "resource_caveats": {"baai_bge_m3": "single nvidia-smi snapshot; not a measured peak"},
        "output": {
            "json_path": "out/four_model_comparison.json",
            "markdown_path": "out/four_model_comparison_ko.md",
            "pairwise_csv_path": "out/four_model_pairwise.csv",
        },
    }

    for model_key, model_id in MODEL_IDS.items():
        summary_rel = f"results/{model_key}/summary.json"
        exam_rel = f"results/{model_key}/dev_eval/exam_mcq/per_query.tsv"
        passage_rel = f"results/{model_key}/dev_eval/ko_passage_strict/per_query.tsv"
        config["candidates"][model_key] = {
            "model_id": model_id,
            "summary_path": summary_rel,
            "exam_per_query_path": exam_rel,
            "passage_per_query_path": passage_rel,
            "accepted_task_commits": [{"task_id": "TASK-TEST", "commit": "0" * 40}],
        }
        _write_json(root / summary_rel, _base_summary(model_key, model_id))
        correct_flags, rrs = EXAM_PATTERNS[model_key]
        _write_exam_tsv(root / exam_rel, EXAM_IDS, correct_flags, rrs)
        ndcgs, mrrs_p, recalls = AIHUB_PATTERNS[model_key]
        _write_passage_tsv(root / passage_rel, AIHUB_IDS, ndcgs, mrrs_p, recalls)

    return config


class FixtureBackedTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = build_fixture_tree(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def load_candidates(self, config=None):
        config = config or self.config
        return CMP.load_all_candidates(config, self.root)


class ValidationPassesOnCleanFixtureTest(FixtureBackedTestCase):
    def test_build_comparison_succeeds_on_clean_fixture(self):
        candidates = self.load_candidates()
        comparison = CMP.build_comparison(self.config, candidates)
        self.assertEqual(
            set(comparison["candidates"].keys()), set(MODEL_IDS.keys())
        )


class IdSetMismatchAndDuplicatesFailClosedTest(FixtureBackedTestCase):
    def test_exam_id_mismatch_fails_closed(self):
        bad_ids = EXAM_IDS[:-1] + ["examq_other"]
        correct_flags, rrs = EXAM_PATTERNS["kure_v1"]
        exam_path = self.root / self.config["candidates"]["kure_v1"]["exam_per_query_path"]
        _write_exam_tsv(exam_path, bad_ids, correct_flags, rrs)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_aihub_id_mismatch_fails_closed(self):
        bad_ids = AIHUB_IDS[:-1] + ["qa_other"]
        ndcgs, mrrs_p, recalls = AIHUB_PATTERNS["baai_bge_m3"]
        passage_path = self.root / self.config["candidates"]["baai_bge_m3"]["passage_per_query_path"]
        _write_passage_tsv(passage_path, bad_ids, ndcgs, mrrs_p, recalls)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_duplicate_exam_query_id_fails_closed(self):
        dup_ids = [EXAM_IDS[0]] + EXAM_IDS[1:]
        dup_ids[-1] = EXAM_IDS[0]
        correct_flags, rrs = EXAM_PATTERNS["kure_v1"]
        exam_path = self.root / self.config["candidates"]["kure_v1"]["exam_per_query_path"]
        _write_exam_tsv(exam_path, dup_ids, correct_flags, rrs)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)


class CorrectedJinaReferenceEnforcementTest(FixtureBackedTestCase):
    def test_wrong_corrected_zero_shot_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["jina_embeddings_v3"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["dev_evaluation"]["exam_mcq"]["corrected_zero_shot"]["accuracy@1"] = 0.4321
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_missing_corrected_zero_shot_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["jina_embeddings_v3"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["dev_evaluation"]["exam_mcq"]["corrected_zero_shot"]
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)


class HistoricalFlawedJinaValuesRejectedTest(FixtureBackedTestCase):
    def test_historical_value_substituted_into_corrected_field_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["jina_embeddings_v3"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Simulate the historical erratum value leaking into the corrected field.
        summary["dev_evaluation"]["exam_mcq"]["corrected_zero_shot"]["accuracy@1"] = (
            JINA_HISTORICAL_FORBIDDEN["exam_accuracy_at_1"]
        )
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_historical_value_never_used_for_delta_on_clean_fixture(self):
        candidates = self.load_candidates()
        comparison = CMP.build_comparison(self.config, candidates)
        jina_zero_shot = comparison["candidates"]["jina_embeddings_v3"]["exam_dev"]["zero_shot"]
        self.assertNotAlmostEqual(
            jina_zero_shot["accuracy_at_1"], JINA_HISTORICAL_FORBIDDEN["exam_accuracy_at_1"]
        )
        self.assertAlmostEqual(jina_zero_shot["accuracy_at_1"], JINA_CORRECTED["exam_accuracy_at_1"])


class UpskyyLicenseLabelGuardTest(FixtureBackedTestCase):
    def test_missing_license_label_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["upskyy_bge_m3_korean"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["license_label"] = None
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_changed_license_label_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["upskyy_bge_m3_korean"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["final_export"]["identity"]["license_label"] = "SOMETHING ELSE"
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)


class UpskyyExternalEligibilityTest(FixtureBackedTestCase):
    def test_upskyy_never_selected_despite_best_raw_metrics(self):
        # Upskyy's fixture pattern (3/4 correct) beats every eligible candidate.
        candidates = self.load_candidates()
        comparison = CMP.build_comparison(self.config, candidates)
        selection = comparison["selection"]
        self.assertNotIn("upskyy_bge_m3_korean", selection["eligible_ranking"])
        self.assertNotEqual(selection["selected_development_candidate"], "upskyy_bge_m3_korean")
        self.assertNotEqual(selection["runner_up"], "upskyy_bge_m3_korean")
        self.assertIn("upskyy_bge_m3_korean", selection["excluded_from_external_selection"])


class ExactTieBreakOrderingTest(FixtureBackedTestCase):
    def test_kure_wins_tie_over_baai_by_exact_mrr(self):
        candidates = self.load_candidates()
        comparison = CMP.build_comparison(self.config, candidates)
        selection = comparison["selection"]
        # kure_v1 and baai_bge_m3 both have 2/4 correct; kure's exact MRR is higher.
        self.assertEqual(selection["selected_development_candidate"], "kure_v1")
        self.assertEqual(selection["runner_up"], "baai_bge_m3")
        tie_step = next(s for s in selection["trace"] if s["step"] == "primary_correct_count_and_accuracy")
        self.assertEqual(set(tie_step["tied"]), {"kure_v1", "baai_bge_m3"})
        mrr_step = next(s for s in selection["trace"] if s["step"] == "tie_break_exact_exam_mrr")
        self.assertGreater(mrr_step["mrr_values"]["kure_v1"], mrr_step["mrr_values"]["baai_bge_m3"])


class BaaiSampledVramCaveatTest(FixtureBackedTestCase):
    def test_baai_vram_labeled_as_sample_not_peak(self):
        candidates = self.load_candidates()
        comparison = CMP.build_comparison(self.config, candidates)
        vram = comparison["candidates"]["baai_bge_m3"]["resource"]["vram"]
        self.assertEqual(vram["kind"], "single_sample")
        self.assertIn("not", vram["caveat"].lower())

    def test_upskyy_vram_labeled_as_continuous_peak(self):
        candidates = self.load_candidates()
        comparison = CMP.build_comparison(self.config, candidates)
        vram = comparison["candidates"]["upskyy_bge_m3_korean"]["resource"]["vram"]
        self.assertEqual(vram["kind"], "continuous_peak")

    def test_kure_has_no_vram_evidence_recorded(self):
        candidates = self.load_candidates()
        comparison = CMP.build_comparison(self.config, candidates)
        vram = comparison["candidates"]["kure_v1"]["resource"]["vram"]
        self.assertEqual(vram["kind"], "not_recorded")


class FrozenContractAndStepValidationTest(FixtureBackedTestCase):
    def test_wrong_step_count_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["kure_v1"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["training"]["global_step"] = STEPS - 1
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_wrong_batch_digest_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["baai_bge_m3"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["provenance"]["batch_plan_digest"] = "0" * 64
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_wrong_precision_fails_closed(self):
        summary_path = self.root / self.config["candidates"]["jina_embeddings_v3"]["summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["frozen_contract"]["precision"] = "fp32"
        _write_json(summary_path, summary)
        candidates = self.load_candidates()
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)

    def test_missing_model_key_fails_closed(self):
        candidates = self.load_candidates()
        del candidates["jina_embeddings_v3"]
        with self.assertRaises(CMP.ValidationError):
            CMP.build_comparison(self.config, candidates)


class DeterministicPairedBootstrapTest(unittest.TestCase):
    def test_same_seed_key_is_byte_identical(self):
        values_a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        values_b = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        r1 = CMP.paired_bootstrap_diff(values_a, values_b, 500, 0.95, "seed-key-a")
        r2 = CMP.paired_bootstrap_diff(values_a, values_b, 500, 0.95, "seed-key-a")
        self.assertEqual(r1, r2)

    def test_different_seed_key_can_differ(self):
        values_a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        values_b = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        r1 = CMP.paired_bootstrap_diff(values_a, values_b, 500, 0.95, "seed-key-a")
        r2 = CMP.paired_bootstrap_diff(values_a, values_b, 500, 0.95, "seed-key-b")
        self.assertNotEqual((r1["low"], r1["high"]), (r2["low"], r2["high"]))

    def test_identical_paired_values_interval_crosses_zero(self):
        values = [0.3, 0.7, 0.5, 0.9, 0.1]
        result = CMP.paired_bootstrap_diff(values, list(values), 300, 0.95, "identical")
        self.assertEqual(result["mean_diff"], 0.0)
        self.assertTrue(result["crosses_zero"])

    def test_clearly_separated_values_interval_excludes_zero(self):
        values_a = [1.0] * 40
        values_b = [0.0] * 40
        result = CMP.paired_bootstrap_diff(values_a, values_b, 300, 0.95, "separated")
        self.assertFalse(result["crosses_zero"])
        self.assertEqual(result["mean_diff"], 1.0)


class ExactMcNemarTest(unittest.TestCase):
    def test_no_discordant_pairs_gives_p_one(self):
        self.assertEqual(CMP.exact_mcnemar_p_value(0, 0), 1.0)

    def test_symmetric_discordant_pairs_gives_p_one(self):
        self.assertAlmostEqual(CMP.exact_mcnemar_p_value(3, 3), 1.0)

    def test_all_discordant_one_direction_is_significant(self):
        p = CMP.exact_mcnemar_p_value(5, 0)
        self.assertAlmostEqual(p, 2 * (0.5**5))


class ParseBoolTest(unittest.TestCase):
    def test_accepts_true_false_literals(self):
        self.assertTrue(CMP.parse_bool("True"))
        self.assertFalse(CMP.parse_bool("False"))

    def test_rejects_other_values(self):
        with self.assertRaises(CMP.ValidationError):
            CMP.parse_bool("yes")


class DeterministicFullRunTest(FixtureBackedTestCase):
    def test_two_invocations_produce_identical_comparison(self):
        candidates_1 = self.load_candidates()
        comparison_1 = CMP.build_comparison(self.config, candidates_1)
        candidates_2 = self.load_candidates()
        comparison_2 = CMP.build_comparison(copy.deepcopy(self.config), candidates_2)
        self.assertEqual(
            json.dumps(comparison_1, sort_keys=True), json.dumps(comparison_2, sort_keys=True)
        )


if __name__ == "__main__":
    unittest.main()
