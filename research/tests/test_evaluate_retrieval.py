import csv
import gzip
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "evaluate_retrieval.py"
SPEC = importlib.util.spec_from_file_location("evaluate_retrieval", MODULE_PATH)
EV = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = EV
SPEC.loader.exec_module(EV)

LIVEQA_QRELS = PROJECT_ROOT / "data" / "processed_medquad_v1" / "liveqa" / "qrels_aggregated.tsv"


class MetricTest(unittest.TestCase):
    """Hand-computed cases. If these drift, every reported score is wrong."""

    def test_ndcg_perfect_ranking_is_one(self):
        self.assertAlmostEqual(EV.ndcg_at_k(["a", "b", "c"], {"a": 1.0}, 10), 1.0)

    def test_ndcg_with_gold_at_rank_three(self):
        # DCG = 1/log2(4); IDCG = 1/log2(2) = 1
        self.assertAlmostEqual(EV.ndcg_at_k(["x", "y", "a"], {"a": 1.0}, 10), 1 / math.log2(4))

    def test_ndcg_is_zero_when_gold_is_outside_the_cutoff(self):
        ranked = [f"d{i}" for i in range(10)] + ["gold"]
        self.assertEqual(EV.ndcg_at_k(ranked, {"gold": 1.0}, 10), 0.0)

    def test_ndcg_without_any_relevant_document(self):
        self.assertEqual(EV.ndcg_at_k(["a", "b"], {}, 10), 0.0)

    def test_graded_ndcg_prefers_the_higher_grade_first(self):
        relevance = {"a": 3.0, "b": 1.0}
        better = EV.ndcg_at_k(["a", "b"], relevance, 10)
        worse = EV.ndcg_at_k(["b", "a"], relevance, 10)
        self.assertAlmostEqual(better, 1.0)
        self.assertLess(worse, better)

    def test_graded_ndcg_matches_hand_computation(self):
        # gains 1 then 3 -> DCG = 1/1 + 3/log2(3); ideal = 3/1 + 1/log2(3)
        relevance = {"a": 3.0, "b": 1.0}
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(EV.ndcg_at_k(["b", "a"], relevance, 10), expected)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(EV.reciprocal_rank_at_k(["x", "y", "a"], {"a"}, 10), 1 / 3)
        self.assertEqual(EV.reciprocal_rank_at_k(["x", "y"], {"a"}, 10), 0.0)

    def test_reciprocal_rank_respects_the_cutoff(self):
        ranked = [f"d{i}" for i in range(9)] + ["gold"]
        self.assertAlmostEqual(EV.reciprocal_rank_at_k(ranked, {"gold"}, 10), 1 / 10)
        self.assertEqual(EV.reciprocal_rank_at_k(ranked, {"gold"}, 9), 0.0)

    def test_recall_counts_only_within_the_cutoff(self):
        ranked = [f"d{i}" for i in range(20)] + ["gold"]
        self.assertEqual(EV.recall_at_k(ranked, {"gold"}, 20), 0.0)
        self.assertEqual(EV.recall_at_k(["gold", "x"], {"gold"}, 20), 1.0)

    def test_recall_with_multiple_positives(self):
        self.assertAlmostEqual(EV.recall_at_k(["a", "x", "y"], {"a", "b"}, 20), 0.5)

    def test_recall_without_positives_is_zero(self):
        self.assertEqual(EV.recall_at_k(["a"], set(), 20), 0.0)


class GradedFixtureTest(unittest.TestCase):
    """LiveQA carries 1-4 human grades, which exercises the non-binary path."""

    @unittest.skipUnless(LIVEQA_QRELS.exists(), "LiveQA qrels not present")
    def test_graded_qrels_produce_a_bounded_score(self):
        relevance = {}
        with LIVEQA_QRELS.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["query_id"] != "1":
                    continue
                relevance[row["answer_id"]] = float(row["relevance_int_half_up"])
        self.assertGreater(len(relevance), 1)
        self.assertGreater(len(set(relevance.values())), 1)

        ideal = [d for d, _ in sorted(relevance.items(), key=lambda kv: -kv[1])]
        self.assertAlmostEqual(EV.ndcg_at_k(ideal, relevance, 10), 1.0)
        reversed_score = EV.ndcg_at_k(list(reversed(ideal)), relevance, 10)
        self.assertLess(reversed_score, 1.0)
        self.assertGreater(reversed_score, 0.0)


class TokenizerTest(unittest.TestCase):
    def test_hangul_becomes_bigrams(self):
        tokens = EV.tokenize("폐렴", "char_bigram")
        self.assertIn("폐렴", tokens)

    def test_longer_hangul_run_yields_overlapping_bigrams(self):
        tokens = EV.tokenize("급성심근경색", "char_bigram")
        self.assertIn("급성", tokens)
        self.assertIn("성심", tokens)

    def test_latin_stays_whole_and_lowercased(self):
        tokens = EV.tokenize("COPD 환자", "char_bigram")
        self.assertIn("copd", tokens)

    def test_mixed_script_produces_both_forms(self):
        tokens = EV.tokenize("MRI 검사에서", "char_bigram")
        self.assertIn("mri", tokens)
        self.assertIn("검사", tokens)

    def test_single_hangul_character_is_kept(self):
        self.assertIn("철", EV.tokenize("철", "char_bigram"))

    def test_whitespace_mode_splits_on_non_word_characters(self):
        self.assertEqual(EV.tokenize("COPD, 환자!", "whitespace"), ["copd", "환자"])


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.retriever = EV.BM25Retriever({})
        self.retriever.index([
            ("d1", "급성 심근경색 환자의 초기 치료"),
            ("d2", "폐렴 환자의 항생제 선택"),
            ("d3", "당뇨병 환자의 혈당 조절"),
        ])

    def test_query_term_ranks_its_document_first(self):
        ranked = self.retriever.search("폐렴 항생제", 3)
        self.assertEqual(ranked[0][0], "d2")

    def test_unmatched_query_returns_nothing(self):
        self.assertEqual(self.retriever.search("zzzz", 3), [])

    def test_top_k_is_respected(self):
        self.assertLessEqual(len(self.retriever.search("환자", 2)), 2)

    def test_ranking_is_deterministic(self):
        first = self.retriever.search("환자", 3)
        second = self.retriever.search("환자", 3)
        self.assertEqual(first, second)

    def test_empty_index_returns_nothing(self):
        empty = EV.BM25Retriever({})
        empty.index([])
        self.assertEqual(empty.search("폐렴", 5), [])


class TaskLoadingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        base = self.root / "data"
        (base / "passage").mkdir(parents=True)
        with gzip.open(base / "passage" / "queries.jsonl.gz", "wt", encoding="utf-8") as handle:
            for qid, split, text in [
                ("q1", "dev", "폐렴의 치료는?"),
                ("q2", "train", "당뇨병의 치료는?"),
            ]:
                handle.write(json.dumps(
                    {"query_id": qid, "split": split, "text": text, "metadata": {"domain": "7"}},
                    ensure_ascii=False) + "\n")
        with gzip.open(base / "passage" / "corpus.jsonl.gz", "wt", encoding="utf-8") as handle:
            for did, text in [("d1", "폐렴은 항생제로 치료한다."), ("d2", "당뇨병은 혈당을 조절한다.")]:
                handle.write(json.dumps({"doc_id": did, "text": text}, ensure_ascii=False) + "\n")
        with (base / "passage" / "qrels.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["query_id", "q0", "doc_id", "relevance", "split"])
            writer.writerow(["q1", "Q0", "d1", 1, "dev"])
            writer.writerow(["q2", "Q0", "d2", 1, "train"])

        self.task = {
            "split": "dev",
            "queries": "data/passage/queries.jsonl.gz",
            "qrels": "data/passage/qrels.tsv",
            "qrels_split_field": "split",
            "corpus": [{"path": "data/passage/corpus.jsonl.gz"}],
        }

    def test_only_the_requested_split_is_loaded(self):
        data = EV.load_retrieval_task(self.root, self.task)
        self.assertEqual([q["query_id"] for q in data["queries"]], ["q1"])
        self.assertIn("q1", data["relevance"])
        self.assertNotIn("q2", data["relevance"])

    def test_whole_corpus_is_indexed_regardless_of_split(self):
        data = EV.load_retrieval_task(self.root, self.task)
        self.assertEqual(len(data["documents"]), 2)

    def test_end_to_end_scores_a_findable_answer(self):
        data = EV.load_retrieval_task(self.root, self.task)
        retriever = EV.BM25Retriever({})
        result = EV.evaluate_retrieval(self.task, data, retriever, {"ndcg": 10, "mrr": 10, "recall": 20})
        self.assertEqual(result["summary"]["queries_scored"], 1)
        self.assertAlmostEqual(result["summary"]["nDCG@10"], 1.0)

    def test_unjudged_distractors_are_tracked_not_scored(self):
        task = dict(self.task)
        task["corpus"] = [
            {"path": "data/passage/corpus.jsonl.gz"},
            {"path": "data/passage/corpus.jsonl.gz", "unjudged_distractors": True},
        ]
        data = EV.load_retrieval_task(self.root, task)
        self.assertEqual(len(data["documents"]), 4)
        self.assertEqual(len(data["judged"]), 2)


class MCQTest(unittest.TestCase):
    def items(self):
        # The stem repeats the answer's wording so the ranking mechanism itself
        # is under test. Real questions rarely share vocabulary with the correct
        # option, which is precisely why a lexical baseline is expected to sit
        # near the random floor on this track.
        return [{
            "question_id": "e1",
            "split": "dev",
            "query": "세균성 폐렴 환자에게 항생제를 투여하려 한다. 적절한 처치는?",
            "options": ["항생제 투여", "인슐린 투여", "혈액 투석", "방사선 치료", "수술"],
            "answer_index": 1,
            "metadata": {"source": "kmle", "year": 2022},
        }]

    def test_correct_option_is_ranked_first(self):
        result = EV.evaluate_mcq(self.items(), {})
        self.assertEqual(result["summary"]["accuracy@1"], 1.0)
        self.assertEqual(result["rows"][0]["rank"], 1)

    def test_option_without_shared_vocabulary_is_not_favoured(self):
        items = self.items()
        items[0]["answer_index"] = 5  # "수술" shares nothing with the stem
        result = EV.evaluate_mcq(items, {})
        self.assertEqual(result["summary"]["accuracy@1"], 0.0)

    def test_random_baseline_reflects_option_count(self):
        result = EV.evaluate_mcq(self.items(), {})
        self.assertAlmostEqual(result["summary"]["random_baseline_accuracy"], 0.2)

    def test_every_question_gets_a_rank_even_without_overlap(self):
        items = self.items()
        items[0]["query"] = "zzzz"
        result = EV.evaluate_mcq(items, {})
        self.assertEqual(len(result["rows"]), 1)
        self.assertGreaterEqual(result["rows"][0]["rank"], 1)
        self.assertLessEqual(result["rows"][0]["rank"], 5)


class SubgroupTest(unittest.TestCase):
    def test_group_counts_sum_to_the_input(self):
        rows = [
            {"record": {"metadata": {"domain": "1"}}, "ndcg": 1.0},
            {"record": {"metadata": {"domain": "1"}}, "ndcg": 0.0},
            {"record": {"metadata": {"domain": "2"}}, "ndcg": 0.5},
        ]
        report = EV.subgroup_report(rows, ["metadata.domain"], ["ndcg"])
        buckets = report["metadata.domain"]
        self.assertEqual(sum(b["n"] for b in buckets.values()), 3)
        self.assertAlmostEqual(buckets["1"]["ndcg"], 0.5)


class TaskPolicyTest(unittest.TestCase):
    def test_expanded_unjudged_task_is_disabled_in_dev_config(self):
        config_path = PROJECT_ROOT / "configs" / "eval_bm25_dev.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expanded = next(task for task in config["tasks"] if task["name"] == "ko_passage_expanded")
        self.assertFalse(expanded["enabled"])
        self.assertIn("pooling gold", expanded["_disabled_note"])


if __name__ == "__main__":
    unittest.main()


class BootstrapTest(unittest.TestCase):
    def test_constant_values_give_a_zero_width_interval(self):
        result = EV.bootstrap_interval([0.5] * 50, 200, 0.95, 1)
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertAlmostEqual(result["half_width"], 0.0)

    def test_interval_brackets_the_mean(self):
        values = [1.0] * 50 + [0.0] * 50
        result = EV.bootstrap_interval(values, 500, 0.95, 1)
        self.assertLessEqual(result["low"], result["mean"])
        self.assertGreaterEqual(result["high"], result["mean"])

    def test_smaller_sample_widens_the_interval(self):
        values = [1.0, 0.0] * 100
        wide = EV.bootstrap_interval(values[:20], 500, 0.95, 1)
        narrow = EV.bootstrap_interval(values, 500, 0.95, 1)
        self.assertGreater(wide["half_width"], narrow["half_width"])

    def test_seed_makes_the_interval_reproducible(self):
        values = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
        self.assertEqual(
            EV.bootstrap_interval(values, 300, 0.95, 7),
            EV.bootstrap_interval(values, 300, 0.95, 7),
        )

    def test_empty_input_is_safe(self):
        self.assertEqual(EV.bootstrap_interval([], 100, 0.95, 1)["n"], 0)


class CodeSwitchLevelTest(unittest.TestCase):
    def record(self, hangul, latin):
        return {"code_switch": {"hangul_chars": hangul, "latin_chars": latin}}

    def test_threshold_moves_the_boundary(self):
        record = self.record(50, 7)
        self.assertEqual(EV.code_switch_level(record, 5), "mixed")
        self.assertEqual(EV.code_switch_level(record, 20), "ko")

    def test_english_dominant(self):
        self.assertEqual(EV.code_switch_level(self.record(0, 40), 20), "en")

    def test_missing_field_falls_back_to_other(self):
        self.assertEqual(EV.code_switch_level({}, 20), "other")


class MCQBaselineTest(unittest.TestCase):
    def item(self, options, answer_index):
        return {
            "question_id": f"q{answer_index}",
            "split": "dev",
            "query": "환자에게 필요한 처치는?",
            "options": options,
            "answer_index": answer_index,
            "metadata": {"source": "kmle"},
        }

    def test_longest_option_baseline_fires_when_gold_is_longest(self):
        result = EV.evaluate_mcq([self.item(["aaaaaa", "bb", "c", "d", "e"], 1)], {})
        self.assertAlmostEqual(result["summary"]["longest_option_baseline"], 1.0)

    def test_baseline_is_zero_when_gold_is_shortest(self):
        result = EV.evaluate_mcq([self.item(["aaaaaa", "bb", "c", "d", "e"], 3)], {})
        self.assertAlmostEqual(result["summary"]["longest_option_baseline"], 0.0)

    def test_ties_split_the_credit(self):
        result = EV.evaluate_mcq([self.item(["aaaa", "aaaa", "b", "c", "d"], 1)], {})
        self.assertAlmostEqual(result["summary"]["longest_option_baseline"], 0.5)


class SubgroupStratificationTest(unittest.TestCase):
    def rows(self):
        return [
            {"record": {"code_switch": {"hangul_chars": 50, "latin_chars": 30}}, "ndcg": 0.4},
            {"record": {"code_switch": {"hangul_chars": 50, "latin_chars": 7}}, "ndcg": 0.8},
            {"record": {"code_switch": {"hangul_chars": 50, "latin_chars": 0}}, "ndcg": 0.9},
        ]

    def test_group_membership_changes_with_the_threshold(self):
        report = EV.subgroup_report(self.rows(), [], ["ndcg"], [5, 20])
        self.assertEqual(report["code_switch@5"]["mixed"]["n"], 2)
        self.assertEqual(report["code_switch@20"]["mixed"]["n"], 1)

    def test_counts_sum_to_the_input_at_every_threshold(self):
        report = EV.subgroup_report(self.rows(), [], ["ndcg"], [5, 20])
        for key, buckets in report.items():
            self.assertEqual(sum(b["n"] for b in buckets.values()), 3, key)

    def test_datasets_without_code_switch_are_skipped(self):
        rows = [{"record": {"metadata": {"domain": "1"}}, "ndcg": 0.5}]
        report = EV.subgroup_report(rows, [], ["ndcg"], [20])
        self.assertNotIn("code_switch@20", report)

    def test_bootstrap_is_attached_when_requested(self):
        report = EV.subgroup_report(self.rows(), [], ["ndcg"], [5],
                                    {"iterations": 100, "confidence": 0.95, "seed": 1})
        self.assertIn("ndcg_ci", report["code_switch@5"]["mixed"])


class MixedDropTest(unittest.TestCase):
    def test_drop_is_relative_to_the_korean_group(self):
        subgroups = {"code_switch@20": {
            "ko": {"n": 100, "ndcg": 0.80},
            "mixed": {"n": 50, "ndcg": 0.60},
        }}
        drop = EV.mixed_drop(subgroups, "ndcg")
        self.assertAlmostEqual(drop["code_switch@20"]["drop"], 0.25)

    def test_missing_mixed_group_produces_nothing(self):
        subgroups = {"code_switch@20": {"ko": {"n": 10, "ndcg": 0.5}}}
        self.assertEqual(EV.mixed_drop(subgroups, "ndcg"), {})

    def test_every_threshold_is_reported(self):
        subgroups = {
            "code_switch@5": {"ko": {"n": 10, "ndcg": 0.8}, "mixed": {"n": 10, "ndcg": 0.7}},
            "code_switch@20": {"ko": {"n": 10, "ndcg": 0.8}, "mixed": {"n": 5, "ndcg": 0.6}},
        }
        self.assertEqual(set(EV.mixed_drop(subgroups, "ndcg")), {"code_switch@5", "code_switch@20"})


# --------------------------------------------------------------------------
# TASK-05: pluggable dense retriever, offline/mock-only
# --------------------------------------------------------------------------

class StubEncoder:
    """Minimal encoder double: exact text -> vector lookup, batches recorded.

    Deliberately not hash-based like the built-in MockEncoder, so ranking
    expectations in these tests can be computed by hand instead of by
    reproducing a hash function.
    """

    provider_name = "stub"
    model_id = "stub-model"
    revision_or_path = "stub-rev"
    library_version = "stub-0"
    dtype_applied = "stub-dtype"
    pooling_applied = "stub-pooling"
    model_revision_applied = "stub-model-revision"
    trust_remote_code_applied = "stub-trust-remote-code"
    code_revision_applied = "stub-code-revision"
    query_adapter_applied = "stub-query-adapter"
    passage_adapter_applied = "stub-passage-adapter"

    def __init__(self, vectors, native_max_seq_length=None):
        self.vectors = vectors
        self.batches: list[list[str]] = []
        self.roles: list[str | None] = []
        self._native_max_seq_length = native_max_seq_length

    def applied_max_seq_length(self, requested):
        if self._native_max_seq_length is None:
            return requested
        return min(requested, self._native_max_seq_length)

    def encode(self, texts, max_length, role=None):
        self.batches.append(list(texts))
        self.roles.append(role)
        return [list(self.vectors[text]) for text in texts]


class DenseHandComputedRankingTest(unittest.TestCase):
    def test_cosine_ranking_matches_hand_computation(self):
        # cosine(query, d1) = 1.0, cosine(query, d3) = 0.6, cosine(query, d2) = 0.0
        vectors = {
            "doc one": [1.0, 0.0],
            "doc two": [0.0, 1.0],
            "doc three": [0.6, 0.8],
            "query": [1.0, 0.0],
        }
        retriever = EV.DenseRetriever(
            {"normalize_embeddings": False, "similarity": "cosine"},
            encoder=StubEncoder(vectors),
        )
        retriever.index([("d1", "doc one"), ("d2", "doc two"), ("d3", "doc three")])
        query_vector = retriever.encode_queries(["query"])[0]
        ranked = retriever.search(query_vector, 3)
        self.assertEqual([doc_id for doc_id, _ in ranked], ["d1", "d3", "d2"])
        self.assertAlmostEqual(ranked[0][1], 1.0)
        self.assertAlmostEqual(ranked[1][1], 0.6)
        self.assertAlmostEqual(ranked[2][1], 0.0)

    def test_top_k_truncates_the_ranking(self):
        vectors = {"a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.0, 1.0], "query": [1.0, 0.0]}
        retriever = EV.DenseRetriever({"normalize_embeddings": False}, encoder=StubEncoder(vectors))
        retriever.index([("d1", "a"), ("d2", "b"), ("d3", "c")])
        query_vector = retriever.encode_queries(["query"])[0]
        ranked = retriever.search(query_vector, 2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual([doc_id for doc_id, _ in ranked], ["d1", "d2"])

    def test_ties_break_by_doc_id_ascending(self):
        vectors = {"same": [0.6, 0.8], "other": [0.6, 0.8], "query": [1.0, 0.0]}
        retriever = EV.DenseRetriever({"normalize_embeddings": False}, encoder=StubEncoder(vectors))
        retriever.index([("dB", "same"), ("dA", "other")])
        query_vector = retriever.encode_queries(["query"])[0]
        ranked = retriever.search(query_vector, 2)
        self.assertEqual([doc_id for doc_id, _ in ranked], ["dA", "dB"])
        self.assertAlmostEqual(ranked[0][1], ranked[1][1])

    def test_ranking_is_deterministic_across_repeated_search(self):
        vectors = {"a": [1.0, 0.0], "b": [0.6, 0.8], "query": [1.0, 0.0]}
        retriever = EV.DenseRetriever({"normalize_embeddings": False}, encoder=StubEncoder(vectors))
        retriever.index([("d1", "a"), ("d2", "b")])
        query_vector = retriever.encode_queries(["query"])[0]
        self.assertEqual(retriever.search(query_vector, 2), retriever.search(query_vector, 2))


class DenseBatchingTest(unittest.TestCase):
    def test_document_encoding_is_batched(self):
        texts = [f"d{i}" for i in range(5)]
        encoder = StubEncoder({t: [1.0, 0.0] for t in texts})
        retriever = EV.DenseRetriever({"document_batch_size": 2}, encoder=encoder)
        retriever.index([(f"id{i}", t) for i, t in enumerate(texts)])
        self.assertEqual([len(batch) for batch in encoder.batches], [2, 2, 1])

    def test_query_encoding_is_batched_separately_from_documents(self):
        texts = [f"q{i}" for i in range(3)]
        encoder = StubEncoder({t: [1.0, 0.0] for t in texts})
        retriever = EV.DenseRetriever({"query_batch_size": 2}, encoder=encoder)
        retriever.encode_queries(texts)
        self.assertEqual([len(batch) for batch in encoder.batches], [2, 1])

    def test_batch_sizes_default_when_unset(self):
        retriever = EV.DenseRetriever({}, encoder=StubEncoder({"solo": [1.0, 0.0]}))
        self.assertEqual(retriever.query_batch_size, 32)
        self.assertEqual(retriever.document_batch_size, 32)


class DenseNormalizationTest(unittest.TestCase):
    def test_normalize_true_yields_unit_vectors(self):
        retriever = EV.DenseRetriever({"normalize_embeddings": True}, encoder=StubEncoder({"d": [3.0, 4.0]}))
        retriever.index([("d1", "d")])
        norm = math.sqrt(sum(v * v for v in retriever.doc_vectors[0]))
        self.assertAlmostEqual(norm, 1.0)

    def test_normalize_false_keeps_raw_magnitude(self):
        retriever = EV.DenseRetriever({"normalize_embeddings": False}, encoder=StubEncoder({"d": [3.0, 4.0]}))
        retriever.index([("d1", "d")])
        norm = math.sqrt(sum(v * v for v in retriever.doc_vectors[0]))
        self.assertAlmostEqual(norm, 5.0)


class SimilarityFunctionTest(unittest.TestCase):
    def test_unknown_similarity_raises_at_construction(self):
        with self.assertRaises(ValueError):
            EV.DenseRetriever({"similarity": "euclidean"}, encoder=StubEncoder({}))

    def test_dot_product_similarity_is_selectable(self):
        encoder = StubEncoder({"d": [2.0, 0.0], "q": [3.0, 0.0]})
        retriever = EV.DenseRetriever(
            {"similarity": "dot_product", "normalize_embeddings": False}, encoder=encoder
        )
        retriever.index([("d1", "d")])
        query_vector = retriever.encode_queries(["q"])[0]
        ranked = retriever.search(query_vector, 1)
        self.assertAlmostEqual(ranked[0][1], 6.0)


class DenseLengthMismatchTest(unittest.TestCase):
    """The upskyy/bge-m3-korean risk: requested length must not be silently truncated."""

    def test_document_length_beyond_native_cap_raises(self):
        encoder = StubEncoder({"doc": [1.0, 0.0]}, native_max_seq_length=512)
        retriever = EV.DenseRetriever({"document_max_seq_length": 8194}, encoder=encoder)
        with self.assertRaises(ValueError) as ctx:
            retriever.index([("d1", "doc")])
        self.assertIn("512", str(ctx.exception))
        self.assertIn("8194", str(ctx.exception))

    def test_query_length_beyond_native_cap_raises(self):
        encoder = StubEncoder({"q": [1.0, 0.0]}, native_max_seq_length=512)
        retriever = EV.DenseRetriever({"query_max_seq_length": 8194}, encoder=encoder)
        with self.assertRaises(ValueError):
            retriever.encode_queries(["q"])

    def test_matching_length_does_not_raise(self):
        encoder = StubEncoder({"doc": [1.0, 0.0]}, native_max_seq_length=512)
        retriever = EV.DenseRetriever({"document_max_seq_length": 512}, encoder=encoder)
        retriever.index([("d1", "doc")])
        self.assertEqual(len(retriever.doc_vectors), 1)

    def test_metadata_exposes_the_mismatch_without_encoding_anything(self):
        encoder = StubEncoder({}, native_max_seq_length=512)
        retriever = EV.DenseRetriever({"document_max_seq_length": 8194}, encoder=encoder)
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertEqual(metadata["document_max_seq_length_requested"], 8194)
        self.assertEqual(metadata["document_max_seq_length_applied"], 512)


class DenseMetadataTest(unittest.TestCase):
    def test_query_and_document_lengths_are_recorded_separately(self):
        retriever = EV.DenseRetriever(
            {"query_max_seq_length": 64, "document_max_seq_length": 256}, encoder=StubEncoder({})
        )
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertEqual(metadata["query_max_seq_length_requested"], 64)
        self.assertEqual(metadata["document_max_seq_length_requested"], 256)

    def test_mock_run_is_explicitly_labeled(self):
        retriever = EV.DenseRetriever({"provider": "mock", "model_name_or_path": "demo"})
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertEqual(metadata["provider"], "mock")
        self.assertEqual(metadata["provider_library_version"], "mock")

    def test_mock_dtype_and_pooling_are_marked_not_applicable(self):
        retriever = EV.DenseRetriever({"provider": "mock", "dtype": "float16", "pooling": "cls"})
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertEqual(metadata["dtype_requested"], "float16")
        self.assertEqual(metadata["dtype_applied"], "not_applicable (mock backend)")
        self.assertEqual(metadata["pooling_requested"], "cls")
        self.assertEqual(metadata["pooling_applied"], "not_applicable (mock backend)")

    def test_config_digest_is_stable_and_distinguishing(self):
        digest_a = EV.DenseRetriever({"provider": "mock", "top_k": 5}).reproducibility_metadata(top_k_applied=5)["config_digest"]
        digest_b = EV.DenseRetriever({"provider": "mock", "top_k": 5}).reproducibility_metadata(top_k_applied=5)["config_digest"]
        digest_c = EV.DenseRetriever({"provider": "mock", "top_k": 10}).reproducibility_metadata(top_k_applied=10)["config_digest"]
        self.assertEqual(digest_a, digest_b)
        self.assertNotEqual(digest_a, digest_c)

    def test_top_k_requested_and_applied_are_distinct_fields(self):
        retriever = EV.DenseRetriever({"provider": "mock", "top_k": 20})
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertEqual(metadata["top_k_requested"], 20)
        self.assertEqual(metadata["top_k_applied"], 10)

    def test_top_k_requested_is_none_rather_than_a_guessed_default(self):
        retriever = EV.DenseRetriever({"provider": "mock"})
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertIsNone(metadata["top_k_requested"])

    def test_reported_fields_cover_the_required_reproducibility_set(self):
        metadata = EV.DenseRetriever({"provider": "mock"}).reproducibility_metadata(top_k_applied=10)
        required = {
            "provider", "model_id", "revision_or_local_path", "device",
            "model_revision_requested", "model_revision_applied",
            "trust_remote_code_requested", "trust_remote_code_applied",
            "code_revision_requested", "code_revision_applied",
            "query_adapter_requested", "query_adapter_applied",
            "passage_adapter_requested", "passage_adapter_applied",
            "query_lora_task_requested", "query_lora_task_applied",
            "passage_lora_task_requested", "passage_lora_task_applied",
            "query_prompt_text_applied", "passage_prompt_text_applied",
            "lora_task_routing_behaviorally_verified",
            "dtype_requested", "dtype_applied",
            "query_batch_size", "document_batch_size",
            "query_max_seq_length_requested", "document_max_seq_length_requested",
            "query_max_seq_length_applied", "document_max_seq_length_applied",
            "query_prefix", "document_prefix",
            "pooling_requested", "pooling_applied", "normalize_embeddings",
            "similarity", "top_k_requested", "top_k_applied",
            "provider_library_version", "config_digest", "device_applied",
            "snapshot_native_max_seq_length", "architecture_max_position_embeddings",
            "tokenizer_class", "adapter_route_counts", "peak_vram_bytes", "license_label",
        }
        self.assertTrue(required.issubset(metadata.keys()))

    def test_mock_remote_code_controls_are_marked_not_applicable(self):
        retriever = EV.DenseRetriever({"provider": "mock"})
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertEqual(metadata["trust_remote_code_applied"], "not_applicable (mock backend)")
        self.assertEqual(metadata["code_revision_applied"], "not_applicable (mock backend)")
        self.assertEqual(metadata["query_adapter_applied"], "not_applicable (mock backend)")
        self.assertEqual(metadata["passage_adapter_applied"], "not_applicable (mock backend)")

    def test_stub_backed_metadata_exposes_requested_and_applied_remote_code_controls(self):
        retriever = EV.DenseRetriever(
            {
                "trust_remote_code": True,
                "code_revision": "codeXYZ",
                "query_adapter": "retrieval.query",
                "passage_adapter": "retrieval.passage",
            },
            encoder=StubEncoder({}),
        )
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertTrue(metadata["trust_remote_code_requested"])
        self.assertEqual(metadata["trust_remote_code_applied"], "stub-trust-remote-code")
        self.assertEqual(metadata["code_revision_requested"], "codeXYZ")
        self.assertEqual(metadata["code_revision_applied"], "stub-code-revision")
        self.assertEqual(metadata["query_adapter_requested"], "retrieval.query")
        self.assertEqual(metadata["query_adapter_applied"], "stub-query-adapter")
        self.assertEqual(metadata["passage_adapter_requested"], "retrieval.passage")
        self.assertEqual(metadata["passage_adapter_applied"], "stub-passage-adapter")


class DenseRoleRoutingTest(unittest.TestCase):
    """The retriever must tell the encoder which side of the pair it is encoding.

    Jina's retrieval.query/retrieval.passage adapters can only be selected if the
    encoder knows whether a given encode() call is for a query or for a
    document/MCQ option - this is the plumbing that carries that role down from
    DenseRetriever, independent of any specific encoder backend.
    """

    def test_index_encodes_with_the_passage_role(self):
        encoder = StubEncoder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        retriever = EV.DenseRetriever({"normalize_embeddings": False}, encoder=encoder)
        retriever.index([("d1", "a"), ("d2", "b")])
        self.assertTrue(encoder.roles)
        self.assertTrue(all(role == "passage" for role in encoder.roles))

    def test_encode_queries_encodes_with_the_query_role(self):
        encoder = StubEncoder({"q1": [1.0, 0.0], "q2": [0.0, 1.0]})
        retriever = EV.DenseRetriever({"normalize_embeddings": False}, encoder=encoder)
        retriever.encode_queries(["q1", "q2"])
        self.assertTrue(encoder.roles)
        self.assertTrue(all(role == "query" for role in encoder.roles))

    def test_mcq_options_are_encoded_with_the_passage_role_not_query(self):
        # evaluate_mcq() calls retriever.index() over the option set, so options
        # must receive the same passage role as a retrieval corpus document.
        vectors = {"stem": [1.0, 0.0], "opt-a": [1.0, 0.0], "opt-b": [0.0, 1.0]}
        encoder = StubEncoder(vectors)
        item = {
            "question_id": "e1", "split": "dev", "query": "stem",
            "options": ["opt-a", "opt-b"], "answer_index": 1, "metadata": {},
        }
        with mock.patch.object(EV, "build_encoder", return_value=encoder):
            EV.evaluate_mcq([item], {"name": "dense", "normalize_embeddings": False})
        # First batch (index()) must be passage; the query batch must be query.
        self.assertIn("passage", encoder.roles)
        self.assertIn("query", encoder.roles)
        index_call_position = encoder.batches.index(["opt-a", "opt-b"])
        self.assertEqual(encoder.roles[index_call_position], "passage")
        query_call_position = encoder.batches.index(["stem"])
        self.assertEqual(encoder.roles[query_call_position], "query")


class DenseEvaluateRetrievalIntegrationTest(unittest.TestCase):
    def test_evaluate_retrieval_batches_queries_through_encode_queries(self):
        task = {"split": "dev"}
        data = {
            "queries": [
                {"query_id": "q1", "text": "alpha", "record": {}},
                {"query_id": "q2", "text": "beta", "record": {}},
            ],
            "documents": [("d1", "alpha doc"), ("d2", "beta doc")],
            "relevance": {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}},
            "judged": {"d1", "d2"},
        }
        vectors = {
            "alpha": [1.0, 0.0], "beta": [0.0, 1.0],
            "alpha doc": [1.0, 0.0], "beta doc": [0.0, 1.0],
        }
        retriever = EV.DenseRetriever({"normalize_embeddings": False}, encoder=StubEncoder(vectors))
        result = EV.evaluate_retrieval(task, data, retriever, {"ndcg": 10, "mrr": 10, "recall": 20})
        self.assertAlmostEqual(result["summary"]["nDCG@10"], 1.0)
        self.assertIn("retriever_metadata", result["summary"])
        self.assertEqual(result["summary"]["retriever_metadata"]["provider"], "stub")
        # top_k_applied must be the real cutoff-derived value every search() call
        # actually used (max of ndcg=10/mrr=10/recall=20 here), not a guessed
        # or configured number.
        self.assertEqual(result["summary"]["retriever_metadata"]["top_k_applied"], 20)


class DenseEvaluateMcqIntegrationTest(unittest.TestCase):
    def test_evaluate_mcq_uses_the_configured_dense_retriever(self):
        vectors = {
            "stem": [1.0, 0.0],
            "opt correct": [1.0, 0.0],
            "opt wrong a": [0.0, 1.0],
            "opt wrong b": [0.0, 1.0],
        }
        item = {
            "question_id": "e1",
            "split": "dev",
            "query": "stem",
            "options": ["opt correct", "opt wrong a", "opt wrong b"],
            "answer_index": 1,
            "metadata": {},
        }
        with mock.patch.object(EV, "build_encoder", return_value=StubEncoder(vectors)):
            result = EV.evaluate_mcq([item], {"name": "dense", "normalize_embeddings": False})
        self.assertEqual(result["summary"]["accuracy@1"], 1.0)
        self.assertIn("retriever_metadata", result["summary"])
        # MCQ candidate count varies per question, so the metadata must not
        # claim a single applied top_k number - the real count lives per-row.
        self.assertIsInstance(result["summary"]["retriever_metadata"]["top_k_applied"], str)
        self.assertEqual(result["rows"][0]["candidate_count"], 3)


class SingleLoadMcqLifecycleTest(unittest.TestCase):
    """TASK-05A: dense MCQ scoring must load its encoder/model exactly once."""

    def _items(self):
        return [
            {
                "question_id": "q1", "split": "dev", "query": "stem-a",
                "options": ["a-correct", "a-wrong"], "answer_index": 1, "metadata": {},
            },
            {
                "question_id": "q2", "split": "dev", "query": "stem-b",
                "options": ["b-wrong", "b-correct"], "answer_index": 2, "metadata": {},
            },
            {
                "question_id": "q3", "split": "dev", "query": "stem-c",
                "options": ["c-wrong-1", "c-correct", "c-wrong-2"], "answer_index": 2, "metadata": {},
            },
        ]

    def _vectors(self):
        return {
            "stem-a": [1.0, 0.0], "a-correct": [1.0, 0.0], "a-wrong": [0.0, 1.0],
            "stem-b": [0.0, 1.0], "b-correct": [0.0, 1.0], "b-wrong": [1.0, 0.0],
            "stem-c": [1.0, 0.0], "c-correct": [1.0, 0.0], "c-wrong-1": [0.0, 1.0], "c-wrong-2": [0.0, 1.0],
        }

    def test_dense_encoder_is_constructed_exactly_once_across_three_questions(self):
        build_calls = []

        def counting_build_encoder(settings):
            build_calls.append(settings)
            return StubEncoder(self._vectors())

        with mock.patch.object(EV, "build_encoder", side_effect=counting_build_encoder):
            result = EV.evaluate_mcq(self._items(), {"name": "dense", "normalize_embeddings": False})

        self.assertEqual(len(build_calls), 1)
        self.assertEqual([row["correct"] for row in result["rows"]], [True, True, True])
        self.assertEqual(result["summary"]["accuracy@1"], 1.0)

    def test_option_indexes_are_isolated_per_question_despite_the_shared_encoder(self):
        # Question 2's correct option text is identical in shape to question 1's
        # wrong option's vector-space direction; if indexes leaked between
        # questions this would rank the wrong option for at least one of them.
        with mock.patch.object(EV, "build_encoder", return_value=StubEncoder(self._vectors())):
            result = EV.evaluate_mcq(self._items(), {"name": "dense", "normalize_embeddings": False})

        self.assertEqual(result["rows"][0]["rank"], 1)
        self.assertEqual(result["rows"][1]["rank"], 1)
        self.assertEqual(result["rows"][2]["rank"], 1)
        self.assertEqual(result["rows"][0]["candidate_count"], 2)
        self.assertEqual(result["rows"][1]["candidate_count"], 2)
        self.assertEqual(result["rows"][2]["candidate_count"], 3)

    def test_bm25_mcq_still_builds_a_fresh_retriever_per_question(self):
        # BM25 has no build_shared_encoder hook, so its existing per-question
        # construction and results must be completely unchanged by this fix.
        # Stems deliberately repeat the correct option's wording, same as
        # MCQTest.items() above, so the lexical ranking mechanism is what is
        # under test here rather than realistic medical phrasing.
        items = [
            {
                "question_id": "e1", "split": "dev",
                "query": "세균성 폐렴 환자에게 항생제를 투여하려 한다. 적절한 처치는?",
                "options": ["항생제 투여", "인슐린 투여", "혈액 투석", "방사선 치료", "수술"],
                "answer_index": 1, "metadata": {},
            },
            {
                "question_id": "e2", "split": "dev",
                "query": "당뇨병 환자에게 인슐린을 투여하려 한다. 적절한 처치는?",
                "options": ["항생제 투여", "인슐린 투여", "혈액 투석", "방사선 치료", "수술"],
                "answer_index": 2, "metadata": {},
            },
        ]
        result = EV.evaluate_mcq(items, {})
        self.assertEqual(result["summary"]["accuracy@1"], 1.0)
        self.assertNotIn("retriever_metadata", result["summary"])


class McqRetrieverNameValidationTest(unittest.TestCase):
    """TASK-05B: an unknown retriever name must never silently fall back to BM25."""

    def test_invalid_retriever_name_raises_with_non_empty_items(self):
        items = [{
            "question_id": "e1", "split": "dev", "query": "q",
            "options": ["a", "b"], "answer_index": 1, "metadata": {},
        }]
        with self.assertRaises(ValueError) as ctx:
            EV.evaluate_mcq(items, {"name": "not_a_real_retriever"})
        self.assertIn("not_a_real_retriever", str(ctx.exception))

    def test_invalid_retriever_name_raises_with_empty_items(self):
        # The name must be validated before the item loop, not discovered by
        # iterating - an empty MCQ task must fail exactly the same way.
        with self.assertRaises(ValueError):
            EV.evaluate_mcq([], {"name": "not_a_real_retriever"})

    def test_default_bm25_name_still_works(self):
        items = [{
            "question_id": "e1", "split": "dev",
            "query": "세균성 폐렴 환자에게 항생제를 투여하려 한다. 적절한 처치는?",
            "options": ["항생제 투여", "인슐린 투여"], "answer_index": 1, "metadata": {},
        }]
        result = EV.evaluate_mcq(items, {})
        self.assertEqual(result["summary"]["accuracy@1"], 1.0)

    def test_explicit_dense_name_still_works(self):
        vectors = {"stem": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0]}
        items = [{
            "question_id": "e1", "split": "dev", "query": "stem",
            "options": ["a", "b"], "answer_index": 1, "metadata": {},
        }]
        with mock.patch.object(EV, "build_encoder", return_value=StubEncoder(vectors)):
            result = EV.evaluate_mcq(items, {"name": "dense", "normalize_embeddings": False})
        self.assertEqual(result["summary"]["accuracy@1"], 1.0)

    def test_empty_items_with_a_valid_name_returns_empty_result_not_an_error(self):
        result = EV.evaluate_mcq([], {"name": "bm25"})
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["summary"]["questions"], 0)


def _fake_sentence_transformers_module(constructor):
    return types.SimpleNamespace(SentenceTransformer=constructor, __version__="9.9.9-fake")


class _FakeTorchDtype:
    """Mimics real torch.dtype's str() - e.g. str(torch.float16) == 'torch.float16'.

    _apply_dtype()'s canonicalization (`str(applied).replace("torch.", "")`)
    depends on exactly this shape, so the fake must produce it too, or a
    passing test would only prove the fake matches itself rather than
    exercising the production canonicalization/comparison logic.
    """

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return f"torch.{self._name}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, _FakeTorchDtype) and self._name == other._name

    def __hash__(self):
        return hash(("_FakeTorchDtype", self._name))


def _fake_torch_module():
    return types.SimpleNamespace(
        float32=_FakeTorchDtype("float32"),
        float16=_FakeTorchDtype("float16"),
        bfloat16=_FakeTorchDtype("bfloat16"),
    )


class _FakeParameter:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakePoolingModule:
    def __init__(self, pooling_mode):
        self.pooling_mode = pooling_mode


class _FakeJinaTransformerModule:
    """Stand-in for custom_st.Transformer at a loaded model's ``[0]`` index.

    Exposes ``_lora_adaptations`` (mirroring the real snapshot's config.json)
    and a real ``forward(task=...)`` whose received ``task`` value a probe can
    observe by temporarily wrapping it - exactly the shape
    ``SentenceTransformerEncoder._verify_lora_task_routing`` depends on to
    behaviorally prove a requested LoRA task actually reaches the module,
    rather than trusting prompt-name-only evidence.
    """

    def __init__(self, lora_adaptations, use_flash_attn=None):
        self._lora_adaptations = list(lora_adaptations)
        if use_flash_attn is not None:
            self.auto_model = types.SimpleNamespace(config=types.SimpleNamespace(use_flash_attn=use_flash_attn))

    def forward(self, features, task=None, **kwargs):
        return {"token_embeddings": features, "task_applied": task}


class _FakeSentenceTransformerModel:
    """Pure-Python stand-in for a loaded SentenceTransformer.

    No torch and no sentence-transformers needed: `.to(dtype=...)` records
    whatever token it was given and `.parameters()` hands it back, mirroring
    just enough of the real API for _apply_dtype/_verify_pooling to exercise
    their actual logic against something other than themselves. Defaults to
    float32, like most real checkpoints before an explicit cast.
    """

    def __init__(
        self, pooling_mode="mean", max_seq_length=512, initial_dtype=None, ignore_to_dtype=False, prompts=None,
        lora_adaptations=None, use_flash_attn=None,
    ):
        self.max_seq_length = max_seq_length
        self._dtype = initial_dtype if initial_dtype is not None else _FakeTorchDtype("float32")
        self._lora_adaptations = lora_adaptations
        first_module = (
            _FakeJinaTransformerModule(lora_adaptations, use_flash_attn=use_flash_attn)
            if lora_adaptations is not None else types.SimpleNamespace()
        )
        self._modules = {"0": first_module, "1": _FakePoolingModule(pooling_mode)}
        # Simulates a backend/device silently ignoring the requested cast -
        # the defect TASK-05B exists to catch.
        self._ignore_to_dtype = ignore_to_dtype
        # None by default, like a standard-module KURE/BAAI checkpoint that
        # exposes no named prompts at all.
        self.prompts = prompts

    def __getitem__(self, index):
        return self._modules[str(index)]

    def to(self, dtype=None, device=None):
        if dtype is not None and not self._ignore_to_dtype:
            self._dtype = dtype
        return self

    def parameters(self):
        yield _FakeParameter(self._dtype)

    def encode(self, texts, prompt_name=None, task=None, **kwargs):
        if self._lora_adaptations is None:
            raise AssertionError("encode() must not run during construction-only tests")
        # Route through the real module object, mirroring how the real sentence_transformers base
        # model threads an encode()-level `task` kwarg into module_kwargs for any module that
        # declares it - so a wrapped/spied first_module.forward() (as
        # SentenceTransformerEncoder._verify_lora_task_routing installs) actually observes it.
        self[0].forward({"texts": list(texts)}, task=task)
        return [[0.0, 0.0] for _ in texts]


class _FakeJinaSentenceTransformerModel(_FakeSentenceTransformerModel):
    """Adds a working encode() that records which prompt_name/task it was given.

    Used by the adapter-routing tests below, which prove that DenseRetriever's
    query/passage roles reach the real SentenceTransformer
    `.encode(prompt_name=..., task=...)` call rather than merely being stored
    on the Python encoder object, and that `task` actually reaches the loaded
    Transformer module's `forward()`.
    """

    def __init__(self, vectors=None, prompts=None, lora_adaptations=None, **kwargs):
        super().__init__(
            prompts=prompts if prompts is not None else {
                "retrieval.query": "Represent the query: ", "retrieval.passage": "Represent the passage: ",
            },
            lora_adaptations=lora_adaptations if lora_adaptations is not None else [
                "retrieval.query", "retrieval.passage",
            ],
            **kwargs,
        )
        self.vectors = vectors or {}
        self.encode_calls = []

    def encode(self, texts, prompt_name=None, task=None, **kwargs):
        self.encode_calls.append({"texts": list(texts), "prompt_name": prompt_name, "task": task})
        self[0].forward({"texts": list(texts)}, task=task)
        return [list(self.vectors.get(text, [0.0, 0.0])) for text in texts]


class OfflineConstructionTest(unittest.TestCase):
    """TASK-05A: local_files_only, and requested-vs-applied dtype/pooling."""

    def test_local_files_only_is_always_passed(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model"})
        constructor.assert_called_once()
        _, kwargs = constructor.call_args
        self.assertIs(kwargs.get("local_files_only"), True)

    def test_local_model_path_expands_user_before_construction(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }), mock.patch.object(Path, "expanduser", return_value=Path("/tmp/expanded-model")):
            EV.SentenceTransformerEncoder({"local_model_path": "~/.cache/model"})
        args, _ = constructor.call_args
        self.assertEqual(args[0], "/tmp/expanded-model")

    def test_existing_snapshot_revision_mismatch_fails_before_construction(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "wrong-revision"
            snapshot.mkdir()
            with mock.patch.dict(sys.modules, {
                "sentence_transformers": _fake_sentence_transformers_module(constructor),
            }):
                with self.assertRaises(ValueError) as ctx:
                    EV.SentenceTransformerEncoder({
                        "model_name_or_path": "example/model",
                        "local_model_path": str(snapshot),
                        "revision": "expected-revision",
                    })
        self.assertIn("expected-revision", str(ctx.exception))
        self.assertIn("wrong-revision", str(ctx.exception))
        constructor.assert_not_called()

    def test_checkpoint_export_directory_is_matched_against_its_identity_not_its_name(self):
        # A contrastive-training final-step-<N> export is named after its optimizer step (see
        # train_contrastive.FINAL_EXPORT_DIR_PREFIX), not the base model revision - the directory
        # name would never equal a real revision hash. Its kbmem_checkpoint_identity.json is the
        # authoritative provenance record instead.
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "final-step-694"
            export_dir.mkdir()
            (export_dir / "kbmem_checkpoint_identity.json").write_text(
                json.dumps({"model_revision": "expected-revision", "global_step": 694}), encoding="utf-8",
            )
            with mock.patch.dict(sys.modules, {
                "sentence_transformers": _fake_sentence_transformers_module(constructor),
                "torch": _fake_torch_module(),
            }):
                EV.SentenceTransformerEncoder({
                    "model_name_or_path": "example/model",
                    "local_model_path": str(export_dir),
                    "revision": "expected-revision",
                })
        constructor.assert_called_once()

    def test_checkpoint_export_identity_revision_mismatch_fails_before_construction(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "final-step-694"
            export_dir.mkdir()
            (export_dir / "kbmem_checkpoint_identity.json").write_text(
                json.dumps({"model_revision": "some-other-revision", "global_step": 694}), encoding="utf-8",
            )
            with mock.patch.dict(sys.modules, {
                "sentence_transformers": _fake_sentence_transformers_module(constructor),
            }):
                with self.assertRaises(ValueError) as ctx:
                    EV.SentenceTransformerEncoder({
                        "model_name_or_path": "example/model",
                        "local_model_path": str(export_dir),
                        "revision": "expected-revision",
                    })
        self.assertIn("expected-revision", str(ctx.exception))
        self.assertIn("some-other-revision", str(ctx.exception))
        constructor.assert_not_called()

    def test_bare_hub_id_without_local_model_path_is_rejected(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {"sentence_transformers": _fake_sentence_transformers_module(constructor)}):
            with self.assertRaises(ValueError):
                EV.SentenceTransformerEncoder({"model_name_or_path": "upskyy/bge-m3-korean"})
        constructor.assert_not_called()

    def test_dtype_is_applied_and_confirmed_from_the_loaded_model(self):
        for requested in ("float32", "float16", "bfloat16"):
            with self.subTest(requested=requested):
                constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
                with mock.patch.dict(sys.modules, {
                    "sentence_transformers": _fake_sentence_transformers_module(constructor),
                    "torch": _fake_torch_module(),
                }):
                    encoder = EV.SentenceTransformerEncoder(
                        {"local_model_path": "/tmp/fake-model", "dtype": requested}
                    )
                self.assertEqual(encoder.dtype_applied, requested)

    def test_reported_revision_never_expands_the_host_home_path(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder({
                "model_name_or_path": "example/model",
                "local_model_path": "~/.cache/example/snapshot/revision123",
                "revision": "revision123",
            })
        self.assertEqual(encoder.revision_or_path, "revision123")
        self.assertNotIn(str(Path.home()), encoder.revision_or_path)

    def test_unsupported_dtype_fails_closed_before_scoring(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model", "dtype": "int8"})
        self.assertIn("int8", str(ctx.exception))

    def test_ignored_dtype_cast_is_rejected_before_scoring(self):
        # The backend claims to honor .to(dtype=float16) but the model stays
        # float32 - TASK-05B's exact defect: this must raise before any
        # encode()/search() call, not be reported as if it had succeeded.
        model = _FakeSentenceTransformerModel(ignore_to_dtype=True)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model", "dtype": "float16"})
        message = str(ctx.exception)
        self.assertIn("float16", message)
        self.assertIn("float32", message)
        # model.encode() raises AssertionError if ever called; reaching this
        # line without that firing already proves encode() was never reached.

    def test_matching_float32_dtype_path_is_accepted(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model", "dtype": "float32"})
        self.assertEqual(encoder.dtype_applied, "float32")

    def test_matching_bfloat16_dtype_path_is_accepted(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model", "dtype": "bfloat16"})
        self.assertEqual(encoder.dtype_applied, "bfloat16")

    def test_pooling_is_verified_against_the_loaded_model(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel(pooling_mode="mean"))
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model", "pooling": "mean"})
        self.assertEqual(encoder.pooling_applied, "mean")

    def test_pooling_mismatch_fails_closed_before_scoring(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel(pooling_mode="cls"))
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model", "pooling": "mean"})
        message = str(ctx.exception)
        self.assertIn("cls", message)
        self.assertIn("mean", message)

    def test_undetectable_pooling_fails_closed_rather_than_guessing(self):
        model = _FakeSentenceTransformerModel(pooling_mode="mean")
        model._modules = {"0": types.SimpleNamespace()}  # no module reports a pooling_mode
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError):
                EV.SentenceTransformerEncoder({"local_model_path": "/tmp/fake-model", "pooling": "mean"})

    def test_mock_provider_never_fails_closed_on_dtype_or_pooling(self):
        encoder = EV.build_encoder({"provider": "mock", "dtype": "int8", "pooling": "nonexistent-mode"})
        self.assertEqual(encoder.dtype_applied, "not_applicable (mock backend)")
        self.assertEqual(encoder.pooling_applied, "not_applicable (mock backend)")


# --------------------------------------------------------------------------
# TASK-06C: fail-closed Jina trust_remote_code/code_revision/adapter controls
# --------------------------------------------------------------------------

JINA_MODEL_ID = EV.JINA_REMOTE_CODE_PIN["model_id"]
JINA_MODEL_REVISION = EV.JINA_REMOTE_CODE_PIN["model_revision"]
JINA_CODE_REVISION = EV.JINA_REMOTE_CODE_PIN["code_revision"]
JINA_QUERY_ADAPTER = EV.JINA_REMOTE_CODE_PIN["query_adapter"]
JINA_PASSAGE_ADAPTER = EV.JINA_REMOTE_CODE_PIN["passage_adapter"]


class RemoteCodeControlValidationTest(unittest.TestCase):
    """Pure-function coverage of _validate_remote_code_controls, no model needed."""

    def _valid_jina_kwargs(self):
        return {
            "model_id": JINA_MODEL_ID,
            "requested_trust_remote_code": True,
            "requested_model_revision": JINA_MODEL_REVISION,
            "requested_code_revision": JINA_CODE_REVISION,
            "requested_query_adapter": JINA_QUERY_ADAPTER,
            "requested_passage_adapter": JINA_PASSAGE_ADAPTER,
        }

    def test_exact_jina_pin_is_accepted(self):
        applied = EV._validate_remote_code_controls(**self._valid_jina_kwargs())
        self.assertEqual(applied, {
            "trust_remote_code": True,
            "model_revision": JINA_MODEL_REVISION,
            "code_revision": JINA_CODE_REVISION,
            "query_adapter": JINA_QUERY_ADAPTER,
            "passage_adapter": JINA_PASSAGE_ADAPTER,
        })

    def test_kure_and_baai_default_path_is_accepted(self):
        for model_id in ("nlpai-lab/KURE-v1", "BAAI/bge-m3"):
            with self.subTest(model_id=model_id):
                applied = EV._validate_remote_code_controls(
                    model_id=model_id,
                    requested_trust_remote_code=False,
                    requested_model_revision="somerevision",
                    requested_code_revision=None,
                    requested_query_adapter=None,
                    requested_passage_adapter=None,
                )
                self.assertFalse(applied["trust_remote_code"])
                self.assertIsNone(applied["code_revision"])
                self.assertIsNone(applied["query_adapter"])
                self.assertIsNone(applied["passage_adapter"])

    def test_kure_or_baai_requesting_trust_remote_code_is_rejected(self):
        for model_id in ("nlpai-lab/KURE-v1", "BAAI/bge-m3"):
            with self.subTest(model_id=model_id):
                kwargs = self._valid_jina_kwargs()
                kwargs["model_id"] = model_id
                with self.assertRaises(ValueError) as ctx:
                    EV._validate_remote_code_controls(**kwargs)
                self.assertIn(model_id, str(ctx.exception))

    def test_arbitrary_non_jina_model_requesting_trust_remote_code_is_rejected(self):
        kwargs = self._valid_jina_kwargs()
        kwargs["model_id"] = "some-org/unrelated-model"
        with self.assertRaises(ValueError):
            EV._validate_remote_code_controls(**kwargs)

    def test_jina_with_mismatched_model_revision_is_rejected(self):
        kwargs = self._valid_jina_kwargs()
        kwargs["requested_model_revision"] = "0000000000000000000000000000000000000"
        with self.assertRaises(ValueError) as ctx:
            EV._validate_remote_code_controls(**kwargs)
        self.assertIn(JINA_MODEL_REVISION, str(ctx.exception))

    def test_jina_with_mismatched_code_revision_is_rejected(self):
        kwargs = self._valid_jina_kwargs()
        kwargs["requested_code_revision"] = "0000000000000000000000000000000000000"
        with self.assertRaises(ValueError) as ctx:
            EV._validate_remote_code_controls(**kwargs)
        self.assertIn(JINA_CODE_REVISION, str(ctx.exception))

    def test_jina_missing_query_adapter_is_rejected(self):
        kwargs = self._valid_jina_kwargs()
        kwargs["requested_query_adapter"] = None
        with self.assertRaises(ValueError) as ctx:
            EV._validate_remote_code_controls(**kwargs)
        self.assertIn(JINA_QUERY_ADAPTER, str(ctx.exception))

    def test_jina_missing_passage_adapter_is_rejected(self):
        kwargs = self._valid_jina_kwargs()
        kwargs["requested_passage_adapter"] = None
        with self.assertRaises(ValueError) as ctx:
            EV._validate_remote_code_controls(**kwargs)
        self.assertIn(JINA_PASSAGE_ADAPTER, str(ctx.exception))

    def test_jina_with_wrong_adapter_string_is_rejected(self):
        kwargs = self._valid_jina_kwargs()
        kwargs["requested_query_adapter"] = "retrieval.wrong"
        with self.assertRaises(ValueError):
            EV._validate_remote_code_controls(**kwargs)

    def test_jina_requesting_adapters_without_trust_remote_code_is_rejected(self):
        with self.assertRaises(ValueError):
            EV._validate_remote_code_controls(
                model_id=JINA_MODEL_ID,
                requested_trust_remote_code=False,
                requested_model_revision=JINA_MODEL_REVISION,
                requested_code_revision=None,
                requested_query_adapter=JINA_QUERY_ADAPTER,
                requested_passage_adapter=JINA_PASSAGE_ADAPTER,
            )

    def test_non_jina_requesting_adapters_without_trust_remote_code_is_rejected(self):
        with self.assertRaises(ValueError):
            EV._validate_remote_code_controls(
                model_id="nlpai-lab/KURE-v1",
                requested_trust_remote_code=False,
                requested_model_revision="somerevision",
                requested_code_revision=None,
                requested_query_adapter=JINA_QUERY_ADAPTER,
                requested_passage_adapter=None,
            )

    def test_jina_requesting_code_revision_without_trust_remote_code_is_rejected(self):
        with self.assertRaises(ValueError):
            EV._validate_remote_code_controls(
                model_id=JINA_MODEL_ID,
                requested_trust_remote_code=False,
                requested_model_revision=JINA_MODEL_REVISION,
                requested_code_revision=JINA_CODE_REVISION,
                requested_query_adapter=None,
                requested_passage_adapter=None,
            )


class JinaConstructionTest(unittest.TestCase):
    """SentenceTransformerEncoder must apply _validate_remote_code_controls before
    ever calling the SentenceTransformer constructor with trust_remote_code=True."""

    def _jina_settings(self, **overrides):
        settings = {
            "local_model_path": "/tmp/fake-jina",
            "model_name_or_path": JINA_MODEL_ID,
            "revision": JINA_MODEL_REVISION,
            "trust_remote_code": True,
            "code_revision": JINA_CODE_REVISION,
            "query_adapter": JINA_QUERY_ADAPTER,
            "passage_adapter": JINA_PASSAGE_ADAPTER,
        }
        settings.update(overrides)
        return settings

    def test_valid_jina_pin_passes_exact_constructor_arguments(self):
        model = _FakeSentenceTransformerModel(
            prompts={JINA_QUERY_ADAPTER: "q", JINA_PASSAGE_ADAPTER: "p"},
            lora_adaptations=[JINA_QUERY_ADAPTER, JINA_PASSAGE_ADAPTER],
        )
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder(self._jina_settings())
        constructor.assert_called_once()
        _, kwargs = constructor.call_args
        self.assertIs(kwargs.get("trust_remote_code"), True)
        self.assertEqual(kwargs.get("revision"), JINA_MODEL_REVISION)
        self.assertEqual(kwargs.get("model_kwargs"), {"code_revision": JINA_CODE_REVISION})
        self.assertEqual(kwargs.get("config_kwargs"), {"code_revision": JINA_CODE_REVISION})
        self.assertIsNot(kwargs.get("model_kwargs"), kwargs.get("config_kwargs"))
        self.assertEqual(kwargs["model_kwargs"].pop("code_revision"), JINA_CODE_REVISION)
        self.assertEqual(kwargs["model_kwargs"]["code_revision"], JINA_CODE_REVISION)
        self.assertIs(kwargs.get("local_files_only"), True)
        self.assertTrue(encoder.trust_remote_code_applied)
        self.assertEqual(encoder.code_revision_applied, JINA_CODE_REVISION)
        # query_adapter_applied/passage_adapter_applied: prompt-NAME selection evidence only.
        self.assertEqual(encoder.query_adapter_applied, JINA_QUERY_ADAPTER)
        self.assertEqual(encoder.passage_adapter_applied, JINA_PASSAGE_ADAPTER)
        self.assertEqual(encoder.query_prompt_text_applied, "q")
        self.assertEqual(encoder.passage_prompt_text_applied, "p")
        # query_lora_task_applied/passage_lora_task_applied: separate, behaviorally-verified
        # evidence that the real LoRA adapter (not just its text prompt) was reachable.
        self.assertEqual(encoder.query_lora_task_applied, JINA_QUERY_ADAPTER)
        self.assertEqual(encoder.passage_lora_task_applied, JINA_PASSAGE_ADAPTER)
        self.assertTrue(encoder.lora_task_routing_behaviorally_verified)

    def test_non_jina_model_requesting_trust_remote_code_never_reaches_the_constructor(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError):
                EV.SentenceTransformerEncoder(self._jina_settings(model_name_or_path="nlpai-lab/KURE-v1"))
        constructor.assert_not_called()

    def test_mismatched_model_revision_never_reaches_the_constructor(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError):
                EV.SentenceTransformerEncoder(self._jina_settings(revision="deadbeef"))
        constructor.assert_not_called()

    def test_mismatched_code_revision_never_reaches_the_constructor(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError):
                EV.SentenceTransformerEncoder(self._jina_settings(code_revision="deadbeef"))
        constructor.assert_not_called()

    def test_missing_adapter_never_reaches_the_constructor(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError):
                EV.SentenceTransformerEncoder(self._jina_settings(passage_adapter=None))
        constructor.assert_not_called()

    def test_kure_style_settings_pass_trust_remote_code_false_explicitly(self):
        model = _FakeSentenceTransformerModel()
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder({
                "local_model_path": "/tmp/fake-kure",
                "model_name_or_path": "nlpai-lab/KURE-v1",
                "revision": "d14c8a9423946e268a0c9952fecf3a7aabd73bd9",
            })
        _, kwargs = constructor.call_args
        self.assertIs(kwargs.get("trust_remote_code"), False)
        self.assertIsNone(kwargs.get("model_kwargs"))
        self.assertIsNone(kwargs.get("config_kwargs"))
        self.assertFalse(encoder.trust_remote_code_applied)
        self.assertIsNone(encoder.code_revision_applied)
        self.assertIsNone(encoder.query_adapter_applied)
        self.assertIsNone(encoder.passage_adapter_applied)

    def test_adapter_not_in_loaded_models_prompts_fails_closed(self):
        # The config-level pin matches exactly, but the loaded model object does
        # not actually expose retrieval.passage in its prompts - this must still
        # fail before any encode() call, per docs/task06_phase_b_gate.md Section 3's
        # "fail closed rather than silently run Jina without the task adapter."
        model = _FakeSentenceTransformerModel(prompts={JINA_QUERY_ADAPTER: "q"})  # passage adapter missing
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder(self._jina_settings())
        self.assertIn(JINA_PASSAGE_ADAPTER, str(ctx.exception))

    def test_model_without_prompts_attribute_fails_closed(self):
        model = _FakeSentenceTransformerModel(prompts=None)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError):
                EV.SentenceTransformerEncoder(self._jina_settings())


class JinaAdapterRoutingTest(unittest.TestCase):
    """Proves query/passage adapter separation reaches the real .encode() call."""

    def _jina_settings(self):
        return {
            "local_model_path": "/tmp/fake-jina",
            "model_name_or_path": JINA_MODEL_ID,
            "revision": JINA_MODEL_REVISION,
            "trust_remote_code": True,
            "code_revision": JINA_CODE_REVISION,
            "query_adapter": JINA_QUERY_ADAPTER,
            "passage_adapter": JINA_PASSAGE_ADAPTER,
            "normalize_embeddings": False,
        }

    def test_query_and_passage_encode_calls_use_distinct_prompt_names(self):
        vectors = {"질문": [1.0, 0.0], "문서": [0.0, 1.0]}
        model = _FakeJinaSentenceTransformerModel(vectors=vectors)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder(self._jina_settings())
            retriever = EV.DenseRetriever(self._jina_settings(), encoder=encoder)
            retriever.index([("d1", "문서")])
            retriever.encode_queries(["질문"])
        prompt_names_by_text = {
            call["texts"][0]: call["prompt_name"] for call in model.encode_calls
        }
        task_by_text = {call["texts"][0]: call["task"] for call in model.encode_calls}
        self.assertEqual(prompt_names_by_text["문서"], JINA_PASSAGE_ADAPTER)
        self.assertEqual(prompt_names_by_text["질문"], JINA_QUERY_ADAPTER)
        self.assertNotEqual(prompt_names_by_text["문서"], prompt_names_by_text["질문"])
        # TASK-08B-R1: every real encode() call must carry both prompt_name and the matching
        # LoRA task - prompt_name alone is not sufficient evidence the adapter was applied.
        self.assertEqual(task_by_text["문서"], JINA_PASSAGE_ADAPTER)
        self.assertEqual(task_by_text["질문"], JINA_QUERY_ADAPTER)
        self.assertEqual(encoder.adapter_route_counts["passage:retrieval.passage:retrieval.passage"], 1)
        self.assertEqual(encoder.adapter_route_counts["query:retrieval.query:retrieval.query"], 1)

    def test_mcq_options_use_the_passage_adapter_not_the_query_adapter(self):
        vectors = {"stem-ko": [1.0, 0.0], "correct-opt": [1.0, 0.0], "wrong-opt": [0.0, 1.0]}
        model = _FakeJinaSentenceTransformerModel(vectors=vectors)
        constructor = mock.Mock(return_value=model)
        item = {
            "question_id": "e1", "split": "dev", "query": "stem-ko",
            "options": ["correct-opt", "wrong-opt"], "answer_index": 1, "metadata": {},
        }
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with mock.patch.object(
                EV, "build_encoder", return_value=EV.SentenceTransformerEncoder(self._jina_settings())
            ):
                result = EV.evaluate_mcq([item], {"name": "dense", **self._jina_settings()})
        self.assertEqual(result["summary"]["accuracy@1"], 1.0)
        prompt_names_by_text = {
            text: call["prompt_name"] for call in model.encode_calls for text in call["texts"]
        }
        task_by_text = {text: call["task"] for call in model.encode_calls for text in call["texts"]}
        self.assertEqual(prompt_names_by_text["stem-ko"], JINA_QUERY_ADAPTER)
        self.assertEqual(prompt_names_by_text["correct-opt"], JINA_PASSAGE_ADAPTER)
        self.assertEqual(prompt_names_by_text["wrong-opt"], JINA_PASSAGE_ADAPTER)
        # MCQ options (and the stem) must carry the real LoRA task alongside the prompt name.
        self.assertEqual(task_by_text["stem-ko"], JINA_QUERY_ADAPTER)
        self.assertEqual(task_by_text["correct-opt"], JINA_PASSAGE_ADAPTER)
        self.assertEqual(task_by_text["wrong-opt"], JINA_PASSAGE_ADAPTER)


class LoraTaskRoutingVerificationTest(unittest.TestCase):
    """TASK-08B-R1: prompt_name alone is not proof a task-specific LoRA adapter ran.

    ``_verify_lora_task_routing`` must fail closed both structurally (the
    requested task names are not really in the loaded model's
    ``_lora_adaptations``) and behaviorally (a real forward call, observed by
    wrapping the loaded Transformer module, does not receive the requested
    ``task`` value) - and it must never run at all for a non-Jina model.
    """

    def _jina_settings(self, **overrides):
        settings = {
            "local_model_path": "/tmp/fake-jina",
            "model_name_or_path": JINA_MODEL_ID,
            "revision": JINA_MODEL_REVISION,
            "trust_remote_code": True,
            "code_revision": JINA_CODE_REVISION,
            "query_adapter": JINA_QUERY_ADAPTER,
            "passage_adapter": JINA_PASSAGE_ADAPTER,
        }
        settings.update(overrides)
        return settings

    def test_task_missing_from_real_lora_adaptations_fails_before_any_scoring(self):
        # The config-level pin matches exactly and .prompts has both names, but the loaded
        # model's real lora_adaptations list is missing retrieval.passage entirely.
        model = _FakeJinaSentenceTransformerModel(lora_adaptations=[JINA_QUERY_ADAPTER])
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder(self._jina_settings())
        self.assertIn(JINA_PASSAGE_ADAPTER, str(ctx.exception))
        self.assertEqual(model.encode_calls, [])  # no probe encode() call was ever reached

    def test_empty_lora_adaptations_fails_closed(self):
        model = _FakeJinaSentenceTransformerModel(lora_adaptations=[])
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder(self._jina_settings())
        self.assertIn("_lora_adaptations", str(ctx.exception))

    def test_task_that_does_not_reach_forward_fails_the_behavioral_probe(self):
        # Structurally everything checks out (prompts + lora_adaptations both list the reviewed
        # names), but the loaded module's forward() silently drops `task` before it can be
        # observed - simulating a model/version whose remote code no longer threads `task`
        # through, or a preprocessing step that strips it before forward() sees it.
        model = _FakeJinaSentenceTransformerModel()

        def _drop_task_before_forward(texts, prompt_name=None, task=None, **kwargs):
            model.encode_calls.append({"texts": list(texts), "prompt_name": prompt_name, "task": task})
            model[0].forward({"texts": list(texts)})  # task deliberately never forwarded
            return [[0.0, 0.0] for _ in texts]

        model.encode = _drop_task_before_forward
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder(self._jina_settings())
        self.assertIn("Behavioral probe failed", str(ctx.exception))

    def test_forward_receiving_the_wrong_task_fails_the_behavioral_probe(self):
        # Simulates a routing bug where the query probe's task leaks into what forward() actually
        # receives for the passage probe (or vice versa) - the probe must catch a wrong value,
        # not just a missing one.
        model = _FakeJinaSentenceTransformerModel()

        def _swap_task_before_forward(texts, prompt_name=None, task=None, **kwargs):
            model.encode_calls.append({"texts": list(texts), "prompt_name": prompt_name, "task": task})
            swapped = JINA_PASSAGE_ADAPTER if task == JINA_QUERY_ADAPTER else JINA_QUERY_ADAPTER
            model[0].forward({"texts": list(texts)}, task=swapped)
            return [[0.0, 0.0] for _ in texts]

        model.encode = _swap_task_before_forward
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder(self._jina_settings())
        self.assertIn("Behavioral probe failed", str(ctx.exception))

    def test_successful_construction_provides_prompt_and_lora_task_evidence_as_distinct_fields(self):
        # Neither field alone stands in for the other: prompt text selection and behaviorally
        # verified LoRA task application are tracked, and asserted, separately.
        model = _FakeJinaSentenceTransformerModel()
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder(self._jina_settings())
        self.assertEqual(encoder.query_adapter_applied, JINA_QUERY_ADAPTER)
        self.assertEqual(encoder.passage_adapter_applied, JINA_PASSAGE_ADAPTER)
        self.assertEqual(encoder.query_prompt_text_applied, "Represent the query: ")
        self.assertEqual(encoder.passage_prompt_text_applied, "Represent the passage: ")
        self.assertEqual(encoder.query_lora_task_applied, JINA_QUERY_ADAPTER)
        self.assertEqual(encoder.passage_lora_task_applied, JINA_PASSAGE_ADAPTER)
        self.assertIs(encoder.lora_task_routing_behaviorally_verified, True)

    def test_non_jina_model_never_triggers_lora_task_verification_or_receives_a_task_kwarg(self):
        model = _FakeSentenceTransformerModel(pooling_mode="cls")
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder({
                "local_model_path": "/tmp/fake-kure", "model_name_or_path": "nlpai-lab/KURE-v1",
                "pooling": "cls",
            })
        self.assertIsNone(encoder.query_lora_task_applied)
        self.assertIsNone(encoder.passage_lora_task_applied)
        self.assertFalse(encoder.lora_task_routing_behaviorally_verified)
        # encode() must never crash (the base fake's encode() raises if reached with
        # lora_adaptations unset) - construction never called it, and neither does a real score.
        encoder._model.encode = mock.Mock(return_value=[[0.1, 0.2]])
        encoder.encode(["문서"], max_length=512, role="passage")
        _, kwargs = encoder._model.encode.call_args
        self.assertNotIn("task", kwargs)
        self.assertNotIn("prompt_name", kwargs)


class UseFlashAttnOverrideTest(unittest.TestCase):
    """TASK-08B-R2: once flash_attn is installed, Jina's mha.py hard-asserts qkv.dtype is
    fp16/bf16, so a fixed fp32 evaluation contract needs a way to force the PyTorch-native
    attention path regardless of whether flash_attn is importable in the running interpreter.
    """

    def _jina_settings(self, **overrides):
        settings = {
            "local_model_path": "/tmp/fake-jina",
            "model_name_or_path": JINA_MODEL_ID,
            "revision": JINA_MODEL_REVISION,
            "trust_remote_code": True,
            "code_revision": JINA_CODE_REVISION,
            "query_adapter": JINA_QUERY_ADAPTER,
            "passage_adapter": JINA_PASSAGE_ADAPTER,
        }
        settings.update(overrides)
        return settings

    def test_unset_by_default_and_never_verified(self):
        model = _FakeJinaSentenceTransformerModel(use_flash_attn=True)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder(self._jina_settings())
        self.assertIsNone(encoder.use_flash_attn_requested)
        self.assertIsNone(encoder.use_flash_attn_applied)

    def test_requested_false_is_passed_as_a_config_override(self):
        model = _FakeJinaSentenceTransformerModel(use_flash_attn=False)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            EV.SentenceTransformerEncoder(self._jina_settings(use_flash_attn=False))
        _, kwargs = constructor.call_args
        self.assertEqual(kwargs.get("config_kwargs"), {"code_revision": JINA_CODE_REVISION, "use_flash_attn": False})

    def test_requested_and_applied_match_is_recorded(self):
        model = _FakeJinaSentenceTransformerModel(use_flash_attn=False)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder(self._jina_settings(use_flash_attn=False))
        self.assertIs(encoder.use_flash_attn_requested, False)
        self.assertIs(encoder.use_flash_attn_applied, False)

    def test_silent_failure_to_apply_the_override_fails_closed(self):
        # The loaded model's real config still reports use_flash_attn=True even though False was
        # requested - simulating the override silently not taking effect.
        model = _FakeJinaSentenceTransformerModel(use_flash_attn=True)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                EV.SentenceTransformerEncoder(self._jina_settings(use_flash_attn=False))
        self.assertIn("use_flash_attn", str(ctx.exception))

    def test_reproducibility_metadata_exposes_requested_and_applied(self):
        model = _FakeJinaSentenceTransformerModel(use_flash_attn=False)
        constructor = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            encoder = EV.SentenceTransformerEncoder(self._jina_settings(use_flash_attn=False))
        retriever = EV.DenseRetriever(self._jina_settings(use_flash_attn=False), encoder=encoder)
        metadata = retriever.reproducibility_metadata(top_k_applied=10)
        self.assertIs(metadata["use_flash_attn_requested"], False)
        self.assertIs(metadata["use_flash_attn_applied"], False)


class DenseMockSmokeTest(unittest.TestCase):
    """Full offline pipeline: real evaluator code, mock encoder, no ML package."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        base = self.root / "data"
        base.mkdir()
        with gzip.open(base / "queries.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"query_id": "q1", "split": "dev", "text": "폐렴 치료", "metadata": {}},
                ensure_ascii=False) + "\n")
        with gzip.open(base / "corpus.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"doc_id": "d1", "text": "폐렴은 항생제로 치료한다."}, ensure_ascii=False) + "\n")
            handle.write(json.dumps({"doc_id": "d2", "text": "당뇨병은 혈당을 조절한다."}, ensure_ascii=False) + "\n")
        with (base / "qrels.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["query_id", "doc_id", "relevance", "split"])
            writer.writerow(["q1", "d1", 1, "dev"])

        self.config = {
            "output_dir": "out",
            "locked_splits": ["test"],
            "cutoffs": {"ndcg": 10, "mrr": 10, "recall": 20},
            "retriever": {"name": "dense", "provider": "mock", "mock_embedding_dim": 16},
            "tasks": [{
                "name": "smoke",
                "type": "retrieval",
                "split": "dev",
                "queries": "data/queries.jsonl.gz",
                "qrels": "data/qrels.tsv",
                "qrels_split_field": "split",
                "corpus": [{"path": "data/corpus.jsonl.gz"}],
            }],
        }

    def test_full_dense_pipeline_runs_offline_with_bm25_compatible_schema(self):
        summaries = EV.run_config(self.root, self.config, self.root / "out")
        summary = summaries["smoke"]
        self.assertIn("nDCG@10", summary)
        self.assertIn("MRR@10", summary)
        self.assertIn("Recall@20", summary)
        self.assertIn("retriever_metadata", summary)
        self.assertEqual(summary["retriever_metadata"]["provider"], "mock")
        output_dir = self.root / "out" / "smoke"
        self.assertTrue((output_dir / "summary.json").exists())
        self.assertTrue((output_dir / "per_query.tsv").exists())
        self.assertNotIn(b"\r\n", (output_dir / "per_query.tsv").read_bytes())


class NoNetworkImportTest(unittest.TestCase):
    def test_import_does_not_load_optional_ml_packages(self):
        code = (
            "import sys, importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('evaluate_retrieval', {str(MODULE_PATH)!r})\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "leaked = sorted(n for n in ('torch', 'transformers', 'sentence_transformers') if n in sys.modules)\n"
            "assert not leaked, leaked\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


class OptionalDependencyErrorTest(unittest.TestCase):
    def test_missing_sentence_transformers_raises_actionable_error(self):
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(RuntimeError) as ctx:
                EV.build_encoder({
                    "provider": "sentence_transformers",
                    "model_name_or_path": "upskyy/bge-m3-korean",
                })
        message = str(ctx.exception)
        self.assertIn("sentence-transformers", message)
        self.assertIn("mock", message)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            EV.build_encoder({"provider": "not_a_real_provider"})

    def test_mock_provider_needs_no_optional_package(self):
        encoder = EV.build_encoder({"provider": "mock"})
        self.assertEqual(encoder.provider_name, "mock")


class LockedSplitTest(unittest.TestCase):
    """Proves locked splits block execution outright, not just result-discarding."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.output_root = self.root / "out"

    def _unresolvable_dense_settings(self):
        # If a retriever were ever built from this, it raises immediately -
        # sentence_transformers is patched out of sys.modules by the caller.
        return {"name": "dense", "provider": "sentence_transformers", "model_name_or_path": "does-not-exist/model"}

    def test_locked_mcq_task_never_builds_a_retriever(self):
        config = {
            "locked_splits": ["test"],
            "tasks": [{
                "name": "locked_mcq",
                "type": "mcq",
                "split": "test",
                "questions": "does/not/exist.jsonl.gz",
                "retriever": self._unresolvable_dense_settings(),
            }],
        }
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            summaries = EV.run_config(self.root, config, self.output_root)
        self.assertEqual(summaries, {})

    def test_locked_retrieval_task_never_builds_a_retriever(self):
        config = {
            "locked_splits": ["test"],
            "cutoffs": {"ndcg": 10, "mrr": 10, "recall": 20},
            "tasks": [{
                "name": "locked_retrieval",
                "type": "retrieval",
                "split": "test",
                "queries": "does/not/exist.jsonl.gz",
                "qrels": "does/not/exist.tsv",
                "corpus": [{"path": "does/not/exist.jsonl.gz"}],
                "retriever": self._unresolvable_dense_settings(),
            }],
        }
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            summaries = EV.run_config(self.root, config, self.output_root)
        self.assertEqual(summaries, {})

    def test_bm25_locked_task_is_skipped_by_the_same_check(self):
        config = {
            "locked_splits": ["test"],
            "tasks": [{
                "name": "locked_bm25",
                "type": "mcq",
                "split": "test",
                "questions": "does/not/exist.jsonl.gz",
                "retriever": {"name": "bm25"},
            }],
        }
        summaries = EV.run_config(self.root, config, self.output_root)
        self.assertEqual(summaries, {})

    def test_unlocked_task_with_the_same_bad_retriever_does_raise(self):
        # Positive control: the same settings really do fail once a retriever is
        # built, so the locked-task tests above are proving prevention, not luck.
        base = self.root / "data"
        base.mkdir()
        with gzip.open(base / "questions.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "question_id": "e1", "split": "dev", "query": "q",
                "options": ["a", "b"], "answer_index": 1, "metadata": {},
            }, ensure_ascii=False) + "\n")
        config = {
            "locked_splits": ["test"],
            "tasks": [{
                "name": "dev_mcq",
                "type": "mcq",
                "split": "dev",
                "questions": "data/questions.jsonl.gz",
                "retriever": self._unresolvable_dense_settings(),
            }],
        }
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(RuntimeError):
                EV.run_config(self.root, config, self.output_root)


class ZeroShotDevConfigTest(unittest.TestCase):
    def _config(self):
        config_path = PROJECT_ROOT / "configs" / "eval_zero_shot_dev.json"
        return json.loads(config_path.read_text(encoding="utf-8"))

    def test_test_split_is_locked_and_never_enabled(self):
        config = self._config()
        self.assertIn("test", config.get("locked_splits", []))
        for task in config["tasks"]:
            if task.get("enabled", True):
                self.assertEqual(task["split"], "dev")

    def test_default_retriever_is_dense_mock_not_a_real_download(self):
        config = self._config()
        self.assertEqual(config["retriever"]["name"], "dense")
        self.assertEqual(config["retriever"]["provider"], "mock")

    def test_expanded_track_and_pseudo_qrels_are_absent(self):
        # Notes are free to explain the prohibition in prose; only the paths a
        # task actually reads matter for this check.
        config = self._config()
        for task in config["tasks"]:
            if not task.get("enabled", True):
                continue
            self.assertNotIn("expanded", task["name"])
            for field in ("queries", "qrels", "questions"):
                if field in task:
                    self.assertNotIn("processed_v2", task[field])
            for source in task.get("corpus", []):
                self.assertNotIn("processed_v2", source["path"])

    def test_query_and_document_max_lengths_are_both_present(self):
        config = self._config()
        retriever = config["retriever"]
        self.assertIn("query_max_seq_length", retriever)
        self.assertIn("document_max_seq_length", retriever)


class ZeroShotModelsConfigTest(unittest.TestCase):
    """TASK-06B-R3: the execution config encodes four exact local frozen snapshots."""

    def _config(self):
        config_path = PROJECT_ROOT / "configs" / "eval_zero_shot_models.json"
        return json.loads(config_path.read_text(encoding="utf-8"))

    def test_all_four_frozen_candidates_are_present(self):
        config = self._config()
        self.assertEqual(set(config["models"]), {
            "kure_v1", "baai_bge_m3", "jina_embeddings_v3", "upskyy_bge_m3_korean",
        })

    def test_every_model_uses_its_exact_dedicated_cache_snapshot(self):
        config = self._config()
        for name, model in config["models"].items():
            with self.subTest(model=name):
                path = Path(model["local_model_path"]).expanduser()
                self.assertEqual(path.name, model["revision"])
                self.assertIn(".cache/k-bmem/huggingface/hub", path.as_posix())
                self.assertEqual(model["device"], "cuda")

    def test_frozen_pooling_matches_snapshot_modules(self):
        config = self._config()["models"]
        self.assertEqual(config["kure_v1"]["pooling"], "cls")
        self.assertEqual(config["baai_bge_m3"]["pooling"], "cls")
        self.assertEqual(config["jina_embeddings_v3"]["pooling"], "mean")
        self.assertEqual(config["upskyy_bge_m3_korean"]["pooling"], "mean")

    def test_standard_module_models_require_trust_remote_code_false_with_no_adapters(self):
        config = self._config()
        for name in ("kure_v1", "baai_bge_m3", "upskyy_bge_m3_korean"):
            with self.subTest(model=name):
                model = config["models"][name]
                self.assertFalse(model["trust_remote_code"])
                self.assertIsNone(model["code_revision"])
                self.assertIsNone(model["query_adapter"])
                self.assertIsNone(model["passage_adapter"])

    def test_kure_revision_matches_the_gate(self):
        model = self._config()["models"]["kure_v1"]
        self.assertEqual(model["model_name_or_path"], "nlpai-lab/KURE-v1")
        self.assertEqual(model["revision"], "d14c8a9423946e268a0c9952fecf3a7aabd73bd9")

    def test_baai_revision_matches_the_gate(self):
        model = self._config()["models"]["baai_bge_m3"]
        self.assertEqual(model["model_name_or_path"], "BAAI/bge-m3")
        self.assertEqual(model["revision"], "5617a9f61b028005a4858fdac845db406aefb181")

    def test_upskyy_exact_internal_only_contract(self):
        model = self._config()["models"]["upskyy_bge_m3_korean"]
        self.assertEqual(model["model_name_or_path"], "upskyy/bge-m3-korean")
        self.assertEqual(model["revision"], "069ae0627320935e4b2879522edbb54650b59bf5")
        self.assertEqual(model["dtype"], "float32")
        self.assertEqual(model["query_max_seq_length"], 512)
        self.assertEqual(model["document_max_seq_length"], 512)
        self.assertEqual(model["license_label"], "LICENSE UNKNOWN — INTERNAL EXPERIMENT ONLY")

    def test_jina_controls_match_the_evaluator_pin_exactly(self):
        model = self._config()["models"]["jina_embeddings_v3"]
        pin = EV.JINA_REMOTE_CODE_PIN
        self.assertEqual(model["model_name_or_path"], pin["model_id"])
        self.assertEqual(model["revision"], pin["model_revision"])
        self.assertTrue(model["trust_remote_code"])
        self.assertEqual(model["code_revision"], pin["code_revision"])
        self.assertEqual(model["query_adapter"], pin["query_adapter"])
        self.assertEqual(model["passage_adapter"], pin["passage_adapter"])

    def test_every_model_entry_passes_remote_code_validation_as_configured(self):
        # Each preset must be internally consistent with the fail-closed contract
        # it will be validated against at construction time - this is a static
        # regression guard against config drift, independent of any real model.
        config = self._config()
        for name, model in config["models"].items():
            with self.subTest(model=name):
                applied = EV._validate_remote_code_controls(
                    model_id=model["model_name_or_path"],
                    requested_trust_remote_code=bool(model["trust_remote_code"]),
                    requested_model_revision=model["revision"],
                    requested_code_revision=model["code_revision"],
                    requested_query_adapter=model["query_adapter"],
                    requested_passage_adapter=model["passage_adapter"],
                )
                self.assertEqual(applied["trust_remote_code"], bool(model["trust_remote_code"]))

    def test_model_selection_merges_without_mutating_dev_lock(self):
        base = json.loads((PROJECT_ROOT / "configs" / "eval_zero_shot_dev.json").read_text(encoding="utf-8"))
        matrix = self._config()
        merged = EV.select_model_config(base, matrix, "upskyy_bge_m3_korean")
        self.assertEqual(base["retriever"]["provider"], "mock")
        self.assertEqual(merged["retriever"]["provider"], "sentence_transformers")
        self.assertEqual(merged["_model_key"], "upskyy_bge_m3_korean")
        self.assertEqual(merged["_license_label"], "LICENSE UNKNOWN — INTERNAL EXPERIMENT ONLY")
        self.assertIn("test", merged["locked_splits"])
        self.assertTrue(all(task["split"] == "dev" for task in merged["tasks"] if task.get("enabled", True)))

    def test_unknown_model_selection_fails_closed(self):
        with self.assertRaises(ValueError):
            EV.select_model_config({}, self._config(), "not-a-model")
