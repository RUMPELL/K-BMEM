import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCHES_MODULE_PATH = PROJECT_ROOT / "scripts" / "build_contrastive_batches.py"
BATCHES_SPEC = importlib.util.spec_from_file_location("build_contrastive_batches_for_tc_test", BATCHES_MODULE_PATH)
BATCHES = importlib.util.module_from_spec(BATCHES_SPEC)
assert BATCHES_SPEC.loader is not None
BATCHES_SPEC.loader.exec_module(BATCHES)

TC_MODULE_PATH = PROJECT_ROOT / "scripts" / "train_contrastive.py"
TC_SPEC = importlib.util.spec_from_file_location("train_contrastive", TC_MODULE_PATH)
TC = importlib.util.module_from_spec(TC_SPEC)
assert TC_SPEC.loader is not None
sys.modules[TC_SPEC.name] = TC
TC_SPEC.loader.exec_module(TC)

CONFIG_PATH = PROJECT_ROOT / "configs" / "train_contrastive_base.json"
REQUIRED_MODEL_KEYS = {"kure_v1", "baai_bge_m3", "jina_embeddings_v3", "upskyy_bge_m3_korean"}


def _build_fixture_plan(tmp_root: Path, rows_by_batch: list[list[dict]]) -> tuple[Path, dict]:
    """Build a real, digest-consistent tiny batch plan using the TASK-04 builder itself.

    Reusing BATCHES.build() - rather than hand-writing a batches.jsonl.gz/summary.json
    pair - guarantees the fixture's deterministic_digest is computed by the exact
    same formula scripts/train_contrastive.py._recompute_batch_plan_digest reproduces,
    so a passing digest test exercises real cross-script agreement, not a tautology.
    """
    fixture_input = tmp_root / "fixture_source.jsonl.gz"
    flat_rows = []
    for batch in rows_by_batch:
        flat_rows.extend(batch)
    with gzip.open(fixture_input, "wt", encoding="utf-8") as handle:
        for index, row in enumerate(flat_rows):
            handle.write(json.dumps({
                "triplet_id": f"t{index}", "split": "train", "query": f"질문 {index}",
                "positive": row["positive"], "hard_negatives": [row["negative"]], "option_len_rank": 1,
            }, ensure_ascii=False) + "\n")
    config = {
        "inputs": [{"path": "fixture_source.jsonl.gz"}], "enforce_registered_inputs": False,
        "seed": "tc-fixture", "batch_size": max(len(b) for b in rows_by_batch),
        "minimum_batch_size": 1, "max_mean_length_gap": 999, "max_quantile_length_gap": 999,
        "gzip_compresslevel": 6,
    }
    plan_dir = tmp_root / "plan"
    plan_dir.mkdir()
    summary = BATCHES.build(config, tmp_root, plan_dir)
    return plan_dir, summary


class LoadBatchPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _plan_config(self, plan_dir: Path, expected_digest: str) -> dict:
        return {
            "dir": str(plan_dir.relative_to(self.root)), "batches_file": "batches.jsonl.gz",
            "summary_file": "summary.json", "expected_digest": expected_digest, "source_run_id": "fixture",
        }

    def test_matching_digest_loads_rows_and_group_sizes(self):
        plan_dir, summary = _build_fixture_plan(self.root, [
            [{"positive": "가나다", "negative": "라마바사"}, {"positive": "사아자", "negative": "차카타"}],
            [{"positive": "파하", "negative": "가나"}],
        ])
        rows, group_sizes, loaded_summary = TC.load_batch_plan(
            self.root, self._plan_config(plan_dir, summary["deterministic_digest"])
        )
        self.assertEqual(sum(group_sizes), len(rows))
        self.assertEqual(loaded_summary["deterministic_digest"], summary["deterministic_digest"])
        self.assertEqual(len(set(row["batch_id"] for row in rows)), len(group_sizes))

    def test_wrong_expected_digest_is_rejected_before_any_row_is_used(self):
        plan_dir, summary = _build_fixture_plan(self.root, [
            [{"positive": "가나다", "negative": "라마바사"}, {"positive": "사아자", "negative": "차카타"}],
        ])
        with self.assertRaises(ValueError) as ctx:
            TC.load_batch_plan(self.root, self._plan_config(plan_dir, "0" * 64))
        self.assertIn("digest preflight failed", str(ctx.exception))

    def test_tampered_batches_file_is_caught_by_independent_recompute(self):
        """Mutating batches.jsonl.gz after the fact must fail even though summary.json still says the old digest."""
        plan_dir, summary = _build_fixture_plan(self.root, [
            [{"positive": "가나다", "negative": "라마바사"}, {"positive": "사아자", "negative": "차카타"}],
        ])
        batches_path = plan_dir / "batches.jsonl.gz"
        with gzip.open(batches_path, "at", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "batch_id": "batch_999999", "record_id": "injected", "source": "attacker",
                "query": "q", "positive": "p", "negative": "n", "option_len_rank": 1,
            }, ensure_ascii=False) + "\n")
        with self.assertRaises(ValueError) as ctx:
            TC.load_batch_plan(self.root, self._plan_config(plan_dir, summary["deterministic_digest"]))
        message = str(ctx.exception)
        self.assertIn("independent recompute", message)

    def test_missing_plan_directory_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            TC.load_batch_plan(self.root, self._plan_config(self.root / "missing", "0" * 64))

    def test_declared_population_counts_are_enforced_when_present(self):
        plan_dir, summary = _build_fixture_plan(self.root, [
            [{"positive": "가나다", "negative": "라마바사"}, {"positive": "사아자", "negative": "차카타"}],
            [{"positive": "파하", "negative": "가나"}],
        ])
        base = self._plan_config(plan_dir, summary["deterministic_digest"])
        rows, group_sizes, _ = TC.load_batch_plan(
            self.root, {**base, "expected_rows": 3, "expected_batch_groups": 2}
        )
        self.assertEqual((len(rows), len(group_sizes)), (3, 2))
        for override in ({"expected_rows": 22204}, {"expected_batch_groups": 694}):
            with self.subTest(override=override):
                with self.assertRaises(ValueError) as ctx:
                    TC.load_batch_plan(self.root, {**base, **override})
                self.assertIn("population preflight failed", str(ctx.exception))


class GroupSizesContiguityTest(unittest.TestCase):
    def test_contiguous_batches_are_sized_correctly(self):
        rows = (
            [{"batch_id": "a"}] * 3 + [{"batch_id": "b"}] * 2 + [{"batch_id": "c"}] * 4
        )
        self.assertEqual(TC._group_sizes_by_contiguous_batch_id(rows), [3, 2, 4])

    def test_non_contiguous_batch_id_is_rejected(self):
        rows = [{"batch_id": "a"}, {"batch_id": "b"}, {"batch_id": "a"}]
        with self.assertRaises(ValueError):
            TC._group_sizes_by_contiguous_batch_id(rows)


class ConfigDigestTest(unittest.TestCase):
    def test_is_deterministic_and_order_independent(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        self.assertEqual(TC._config_digest(a), TC._config_digest(b))

    def test_differs_when_content_differs(self):
        self.assertNotEqual(TC._config_digest({"a": 1}), TC._config_digest({"a": 2}))


class RuntimeLockFingerprintTest(unittest.TestCase):
    def test_excludes_pip_itself_and_is_deterministic_for_the_same_interpreter(self):
        first = TC.runtime_lock_fingerprint(sys.executable)
        second = TC.runtime_lock_fingerprint(sys.executable)
        self.assertEqual(first, second)
        self.assertGreater(first["package_count"], 0)


class ResolveCheckpointIdentityTest(unittest.TestCase):
    def test_builds_exactly_the_mismatch_fields_plus_schema_version(self):
        config = {
            "objective": {"loss": "multiple_negatives_ranking_loss"}, "runtime": {"precision": "bfloat16"},
            "schedule": {"seed": 42},
        }
        metadata = {"revision_applied": "rev-1", "code_revision_applied": None, "max_seq_length_applied": 512}
        identity = TC.resolve_checkpoint_identity(
            config=config, config_digest="cfg-digest", batch_plan_digest="batch-digest", metadata=metadata,
            runtime_lock_sha256="runtime-digest",
        )
        for field in TC.IDENTITY_MISMATCH_FIELDS:
            self.assertIn(field, identity)
        self.assertEqual(identity["model_revision"], "rev-1")
        self.assertEqual(identity["seed"], 42)
        self.assertEqual(identity["schema_version"], TC.CHECKPOINT_IDENTITY_SCHEMA_VERSION)


class EstimateTrainingDiskBytesTest(unittest.TestCase):
    """Replaces a single fixed free-space threshold with a component-based estimate."""

    def test_hand_computed_breakdown(self):
        estimate = TC.estimate_training_disk_bytes(
            param_count=1000, bytes_per_param=2, retained_checkpoints=3,
            optimizer_state_multiplier=2.0, fixed_overhead_bytes=100,
            staging_overhead_checkpoints=1, run_overhead_bytes=1000, safety_margin=0.5,
        )
        self.assertEqual(estimate["model_weight_bytes"], 2000)
        self.assertEqual(estimate["optimizer_state_bytes"], 4000)
        self.assertEqual(estimate["gradient_bytes"], 0)
        self.assertEqual(estimate["one_checkpoint_bytes"], 2000 + 4000 + 100)
        self.assertEqual(estimate["checkpoint_pool_bytes"], 6100 * (3 + 1))
        self.assertEqual(estimate["final_export_bytes"], 2000)
        subtotal = 6100 * 4 + 2000 + 1000
        self.assertEqual(estimate["subtotal_bytes"], subtotal)
        self.assertEqual(estimate["required_bytes"], int(subtotal * 1.5))

    def test_more_retained_checkpoints_increases_requirement(self):
        low = TC.estimate_training_disk_bytes(param_count=1000, bytes_per_param=2, retained_checkpoints=1)
        high = TC.estimate_training_disk_bytes(param_count=1000, bytes_per_param=2, retained_checkpoints=5)
        self.assertGreater(high["required_bytes"], low["required_bytes"])

    def test_staging_overhead_increases_requirement(self):
        without = TC.estimate_training_disk_bytes(
            param_count=1000, bytes_per_param=2, retained_checkpoints=2, staging_overhead_checkpoints=0,
        )
        with_staging = TC.estimate_training_disk_bytes(
            param_count=1000, bytes_per_param=2, retained_checkpoints=2, staging_overhead_checkpoints=1,
        )
        self.assertGreater(with_staging["required_bytes"], without["required_bytes"])

    def test_larger_safety_margin_increases_requirement(self):
        low = TC.estimate_training_disk_bytes(
            param_count=1000, bytes_per_param=2, retained_checkpoints=2, safety_margin=0.0,
        )
        high = TC.estimate_training_disk_bytes(
            param_count=1000, bytes_per_param=2, retained_checkpoints=2, safety_margin=1.0,
        )
        self.assertGreater(high["required_bytes"], low["required_bytes"])
        self.assertEqual(low["required_bytes"], low["subtotal_bytes"])

    def test_invalid_configuration_is_rejected(self):
        base = dict(param_count=1000, bytes_per_param=2, retained_checkpoints=2)
        for override in (
            {"param_count": 0}, {"param_count": -1}, {"bytes_per_param": 0}, {"bytes_per_param": -1},
            {"retained_checkpoints": 0}, {"retained_checkpoints": -1}, {"staging_overhead_checkpoints": -1},
            {"safety_margin": -0.1},
        ):
            with self.assertRaises(ValueError):
                TC.estimate_training_disk_bytes(**{**base, **override})


class DiskPreflightTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _estimate(self, required_bytes):
        return {"required_bytes": required_bytes, "note": "synthetic test estimate"}

    def test_sufficient_space_passes_and_reports_usage(self):
        target = self.root / "checkpoints" / "nested"
        usage = TC.disk_preflight(target, self._estimate(1))
        self.assertTrue(target.is_dir())
        self.assertIn("free", usage)
        self.assertGreaterEqual(usage["free"], 1)
        self.assertIn("estimate", usage)

    def test_insufficient_space_raises_before_any_write(self):
        target = self.root / "checkpoints"
        with self.assertRaises(RuntimeError):
            TC.disk_preflight(target, self._estimate(10 ** 18))

    def test_exact_pass_boundary(self):
        target = self.root / "checkpoints"
        target.mkdir()
        free = shutil.disk_usage(target).free
        TC.disk_preflight(target, self._estimate(free))  # exactly free bytes must pass

    def test_exact_fail_boundary(self):
        target = self.root / "checkpoints"
        target.mkdir()
        free = shutil.disk_usage(target).free
        with self.assertRaises(RuntimeError):
            TC.disk_preflight(target, self._estimate(free + 1))


class TokenizerLengthStatsTest(unittest.TestCase):
    class _FakeTokenizer:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": list(range(len(text)))}

    def test_hand_computed_mean_median_and_gap(self):
        rows = [
            {"positive": "12345", "negative": "12"},   # positive len 5, negative len 2
            {"positive": "1234567", "negative": "1234"},  # positive len 7, negative len 4
        ]
        stats = TC.tokenizer_length_stats(self._FakeTokenizer(), rows, ["positive", "negative"], None, seed=0)
        self.assertEqual(stats["positive"]["mean"], 6.0)
        self.assertEqual(stats["negative"]["mean"], 3.0)
        self.assertEqual(stats["gap"]["mean_gap"], 3.0)
        self.assertEqual(stats["sample_size"], 2)
        self.assertEqual(stats["population_size"], 2)

    def test_sample_size_is_deterministic_for_a_fixed_seed(self):
        rows = [{"positive": "x" * i, "negative": "y" * i} for i in range(1, 21)]
        first = TC.tokenizer_length_stats(self._FakeTokenizer(), rows, ["positive"], 5, seed=7)
        second = TC.tokenizer_length_stats(self._FakeTokenizer(), rows, ["positive"], 5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_size"], 5)

    def test_empty_rows_reports_zero_counts_without_error(self):
        stats = TC.tokenizer_length_stats(self._FakeTokenizer(), [], ["positive", "negative"], None, seed=0)
        self.assertEqual(stats["positive"]["count"], 0)
        self.assertNotIn("gap", stats)

    def test_percentiles_and_gaps_are_hand_computed(self):
        # Ten positives of length 1..10, tokenizer id-count == text length.
        rows = [{"positive": "x" * i, "negative": "y" * i} for i in range(1, 11)]
        stats = TC.tokenizer_length_stats(self._FakeTokenizer(), rows, ["positive", "negative"], None, seed=0)
        self.assertEqual(stats["positive"]["p50"], 5)
        self.assertEqual(stats["positive"]["p95"], 10)
        self.assertEqual(stats["positive"]["p99"], 10)
        self.assertEqual(stats["gap"]["p50_gap"], 0)
        self.assertIn("p50_gap", stats["gap"])
        self.assertIn("p95_gap", stats["gap"])
        self.assertIn("p99_gap", stats["gap"])

    def test_max_seq_length_reports_count_and_percentage_exceeding(self):
        rows = [{"positive": "x" * i} for i in (3, 5, 8, 12)]  # lengths 3,5,8,12
        stats = TC.tokenizer_length_stats(self._FakeTokenizer(), rows, ["positive"], None, seed=0, max_seq_length=8)
        self.assertEqual(stats["positive"]["count_exceeding_max"], 1)  # only length 12
        self.assertEqual(stats["positive"]["percentage_exceeding_max"], 25.0)
        self.assertEqual(stats["positive"]["requested_max_seq_length"], 8)
        self.assertEqual(stats["positive"]["applied_max_seq_length"], 8)

    def test_max_seq_length_omitted_leaves_exceeding_fields_absent(self):
        stats = TC.tokenizer_length_stats(self._FakeTokenizer(), [{"positive": "abc"}], ["positive"], None, seed=0)
        self.assertNotIn("count_exceeding_max", stats["positive"])


class MakeTaskAwareMnrlLossTest(unittest.TestCase):
    """Pure unit coverage of make_task_aware_mnrl_loss(), independent of run_training() wiring.

    Proves the wrapper (a) routes sentence_features[0] through query_task and every later entry
    through passage_task, (b) delegates scoring to the installed loss's own
    compute_loss_from_embeddings() unmodified, and (c) never touches sentence_transformers or
    transformers - only real torch, matching the "do not patch installed libraries" constraint.
    """

    class _FakeTaskModel:
        def __init__(self):
            self.calls = []

        def __call__(self, features, task=None):
            self.calls.append((features, task))
            return {"sentence_embedding": f"embedding::{features}::{task}"}

    class _FakeBaseLoss:
        def __init__(self, model):
            self.model = model
            self.compute_calls = []

        def compute_loss_from_embeddings(self, embeddings, labels):
            self.compute_calls.append((list(embeddings), labels))
            return "fake-loss"

        def get_config_dict(self):
            return {"scale": 20.0}

    def test_first_entry_uses_query_task_and_rest_use_passage_task(self):
        model = self._FakeTaskModel()
        base_loss = self._FakeBaseLoss(model)
        wrapped = TC.make_task_aware_mnrl_loss(base_loss, query_task="retrieval.query", passage_task="retrieval.passage")
        sentence_features = ["query-features", "positive-features", "negative-features"]

        result = wrapped(sentence_features, labels="labels-sentinel")

        self.assertEqual(result, "fake-loss")
        self.assertEqual(
            [task for _features, task in model.calls],
            ["retrieval.query", "retrieval.passage", "retrieval.passage"],
        )
        self.assertEqual([features for features, _task in model.calls], sentence_features)
        embeddings_passed, labels_passed = base_loss.compute_calls[0]
        self.assertEqual(labels_passed, "labels-sentinel")
        self.assertEqual(embeddings_passed, [
            "embedding::query-features::retrieval.query",
            "embedding::positive-features::retrieval.passage",
            "embedding::negative-features::retrieval.passage",
        ])

    def test_handles_multiple_hard_negatives_beyond_the_frozen_triplet_shape(self):
        model = self._FakeTaskModel()
        base_loss = self._FakeBaseLoss(model)
        wrapped = TC.make_task_aware_mnrl_loss(base_loss, query_task="Q", passage_task="P")

        wrapped(["anchor", "pos", "neg1", "neg2", "neg3"], labels=None)

        self.assertEqual([task for _f, task in model.calls], ["Q", "P", "P", "P", "P"])

    def test_pair_only_input_still_routes_position_zero_as_query(self):
        model = self._FakeTaskModel()
        base_loss = self._FakeBaseLoss(model)
        wrapped = TC.make_task_aware_mnrl_loss(base_loss, query_task="Q", passage_task="P")

        wrapped(["anchor", "pos"], labels=None)

        self.assertEqual([task for _f, task in model.calls], ["Q", "P"])

    def test_get_config_dict_merges_base_config_and_task_fields(self):
        model = self._FakeTaskModel()
        base_loss = self._FakeBaseLoss(model)
        wrapped = TC.make_task_aware_mnrl_loss(base_loss, query_task="retrieval.query", passage_task="retrieval.passage")

        config = wrapped.get_config_dict()

        self.assertEqual(config["scale"], 20.0)
        self.assertEqual(config["query_lora_task"], "retrieval.query")
        self.assertEqual(config["passage_lora_task"], "retrieval.passage")

    def test_does_not_import_sentence_transformers_or_transformers(self):
        with mock.patch.dict(sys.modules, {"sentence_transformers": None, "transformers": None}):
            model = self._FakeTaskModel()
            base_loss = self._FakeBaseLoss(model)
            wrapped = TC.make_task_aware_mnrl_loss(base_loss, query_task="Q", passage_task="P")
            wrapped(["a", "b"], labels=None)  # must not raise ImportError


class FrozenGroupBatchSamplerTest(unittest.TestCase):
    def test_yields_exactly_the_configured_groups_with_no_generator(self):
        dataset = list(range(9))
        sampler = TC.FrozenGroupBatchSampler(dataset, group_sizes=[3, 2, 4])
        groups = list(sampler)
        self.assertEqual(groups, [[0, 1, 2], [3, 4], [5, 6, 7, 8]])
        self.assertEqual(len(sampler), 3)

    def test_shuffled_order_never_splits_or_merges_a_group(self):
        import torch

        dataset = list(range(10))
        generator = torch.Generator()
        generator.manual_seed(1234)
        sampler = TC.FrozenGroupBatchSampler(dataset, generator=generator, group_sizes=[4, 3, 3])
        groups = list(sampler)
        expected_groups = {(0, 1, 2, 3), (4, 5, 6), (7, 8, 9)}
        observed_groups = {tuple(group) for group in groups}
        self.assertEqual(observed_groups, expected_groups)

    def test_group_sizes_must_match_dataset_length(self):
        with self.assertRaises(ValueError):
            TC.FrozenGroupBatchSampler(list(range(9)), group_sizes=[3, 3])

    def test_missing_group_sizes_is_rejected(self):
        with self.assertRaises(ValueError):
            TC.FrozenGroupBatchSampler(list(range(9)))

    def test_factory_accepts_and_ignores_trainer_supplied_kwargs(self):
        factory = TC.make_frozen_batch_sampler_factory([2, 2])
        sampler = factory(list(range(4)), batch_size=32, drop_last=False, valid_label_columns=["label"],
                           generator=None, seed=0)
        self.assertEqual(list(sampler), [[0, 1], [2, 3]])


class ResolveCheckpointRootTest(unittest.TestCase):
    """A real GPU smoke run proved a run_id-scoped checkpoint path breaks resume across restarts."""

    def test_stable_across_repeated_calls_for_the_same_model_key(self):
        checkpoint_cfg = {"root_dir": "/tmp/example-checkpoints"}
        first = TC.resolve_checkpoint_root(checkpoint_cfg, "upskyy_bge_m3_korean")
        second = TC.resolve_checkpoint_root(checkpoint_cfg, "upskyy_bge_m3_korean")
        self.assertEqual(first, second)
        self.assertNotIn("run_", str(first))

    def test_different_model_keys_get_different_roots(self):
        checkpoint_cfg = {"root_dir": "/tmp/example-checkpoints"}
        kure_root = TC.resolve_checkpoint_root(checkpoint_cfg, "kure_v1")
        baai_root = TC.resolve_checkpoint_root(checkpoint_cfg, "baai_bge_m3")
        self.assertNotEqual(kure_root, baai_root)

    def test_expands_user_home_in_root_dir(self):
        result = TC.resolve_checkpoint_root({"root_dir": "~/checkpoints"}, "kure_v1")
        self.assertFalse(str(result).startswith("~"))


def _write_fake_checkpoint_state_files(staged_dir: Path) -> None:
    staged_dir.mkdir(parents=True, exist_ok=True)
    for name in TC.REQUIRED_CHECKPOINT_FILES:
        (staged_dir / name).write_bytes(f"fake {name}".encode("utf-8"))


class AtomicSaveCheckpointTest(unittest.TestCase):
    """TASK-07R1: trainer_state.json alone is not a completion marker - proves the real contract."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_missing_required_state_file_is_rejected_before_any_write(self):
        staged = self.root / ".staging" / "checkpoint-1"
        staged.mkdir(parents=True)
        (staged / "model.safetensors").write_bytes(b"x")  # optimizer.pt etc. missing
        with self.assertRaises(RuntimeError):
            TC.atomic_save_checkpoint(staged_dir=staged, final_dir=self.root / "checkpoint-1", identity={"a": 1})
        self.assertFalse((self.root / "checkpoint-1").exists())

    def test_successful_save_writes_identity_and_marker_then_atomically_renames(self):
        staged = self.root / ".staging" / "checkpoint-5"
        _write_fake_checkpoint_state_files(staged)
        final_dir = self.root / "checkpoint-5"
        identity = {"schema_version": 1, "global_step": 5}

        result = TC.atomic_save_checkpoint(staged_dir=staged, final_dir=final_dir, identity=identity)

        self.assertEqual(result, final_dir)
        self.assertFalse(staged.exists())  # renamed away, not copied
        self.assertTrue(final_dir.is_dir())
        saved_identity = json.loads((final_dir / TC.CHECKPOINT_IDENTITY_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(saved_identity, identity)
        marker = json.loads((final_dir / TC.CHECKPOINT_COMPLETE_MARKER).read_text(encoding="utf-8"))
        self.assertTrue(marker["complete"])
        expected_identity_text = json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        self.assertEqual(marker["identity_sha256"], hashlib.sha256(expected_identity_text.encode("utf-8")).hexdigest())

    def test_refuses_to_overwrite_an_already_completed_checkpoint(self):
        staged = self.root / ".staging" / "checkpoint-5"
        _write_fake_checkpoint_state_files(staged)
        final_dir = self.root / "checkpoint-5"
        final_dir.mkdir(parents=True)
        (final_dir / "existing").write_text("already here", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            TC.atomic_save_checkpoint(staged_dir=staged, final_dir=final_dir, identity={})
        self.assertTrue((final_dir / "existing").is_file())  # untouched


def _write_fake_final_export_files(directory: Path) -> None:
    """Stand in for SentenceTransformer.save(): every file the export contract requires."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "1_Pooling").mkdir(exist_ok=True)
    (directory / "1_Pooling" / "config.json").write_text('{"pooling_mode_cls_token": true}', encoding="utf-8")
    for name in TC.REQUIRED_FINAL_EXPORT_FILES:
        (directory / name).write_bytes(f"fake {name}".encode("utf-8"))


class _FakeSavableModel:
    """Records what it was asked to save, and writes a plausible export tree."""

    def __init__(self, writer=_write_fake_final_export_files) -> None:
        self.writer = writer
        self.save_calls: list[tuple[str, dict]] = []

    def save(self, path, **kwargs):
        self.save_calls.append((str(path), kwargs))
        self.writer(Path(path))


class AtomicExportFinalModelTest(unittest.TestCase):
    """TASK-08A: with save_steps=50 the newest checkpoint is 44 steps behind the trained model,
    so the exact final state needs its own durable, identity-bound, non-overwritable export.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_successful_export_writes_identity_and_marker_then_atomically_renames(self):
        model = _FakeSavableModel()
        identity = {"schema_version": 1, "global_step": 694, "export_kind": TC.FINAL_EXPORT_KIND}
        staged = self.root / ".staging" / "final-step-694"
        final_dir = self.root / "final-step-694"

        result = TC.atomic_export_final_model(
            save_model=lambda directory: model.save(directory), staged_dir=staged, final_dir=final_dir,
            identity=identity,
        )

        self.assertFalse(staged.exists())  # renamed away, not copied
        self.assertTrue(final_dir.is_dir())
        self.assertEqual(result["path"], str(final_dir))
        self.assertEqual(result["global_step"], 694)
        saved_identity = json.loads((final_dir / TC.CHECKPOINT_IDENTITY_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(saved_identity, identity)
        marker = json.loads((final_dir / TC.CHECKPOINT_COMPLETE_MARKER).read_text(encoding="utf-8"))
        self.assertTrue(marker["complete"])
        self.assertEqual(marker["global_step"], 694)
        self.assertEqual(marker["export_kind"], TC.FINAL_EXPORT_KIND)
        identity_text = json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        self.assertEqual(marker["identity_sha256"], hashlib.sha256(identity_text.encode("utf-8")).hexdigest())
        self.assertEqual(
            result["model_safetensors_sha256"],
            hashlib.sha256(b"fake model.safetensors").hexdigest(),
        )
        self.assertIn("1_Pooling/config.json", result["files"])
        self.assertGreater(result["total_bytes"], 0)

    def test_missing_required_file_leaves_no_final_directory(self):
        def incomplete_writer(directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "model.safetensors").write_bytes(b"weights only, no module configs")

        final_dir = self.root / "final-step-694"
        with self.assertRaises(RuntimeError) as ctx:
            TC.atomic_export_final_model(
                save_model=incomplete_writer, staged_dir=self.root / ".staging" / "final-step-694",
                final_dir=final_dir, identity={"global_step": 694},
            )
        self.assertIn("modules.json", str(ctx.exception))
        self.assertFalse(final_dir.exists())

    def test_refuses_to_overwrite_an_existing_export(self):
        final_dir = self.root / "final-step-694"
        final_dir.mkdir(parents=True)
        (final_dir / "existing").write_text("already here", encoding="utf-8")
        model = _FakeSavableModel()
        with self.assertRaises(RuntimeError):
            TC.atomic_export_final_model(
                save_model=lambda directory: model.save(directory),
                staged_dir=self.root / ".staging" / "final-step-694", final_dir=final_dir,
                identity={"global_step": 694},
            )
        self.assertEqual(model.save_calls, [])  # refused before writing anything
        self.assertTrue((final_dir / "existing").is_file())

    def test_refuses_to_reuse_leftover_staging(self):
        staged = self.root / ".staging" / "final-step-694"
        staged.mkdir(parents=True)
        (staged / "half-written.bin").write_bytes(b"residue")
        with self.assertRaises(RuntimeError):
            TC.atomic_export_final_model(
                save_model=_write_fake_final_export_files, staged_dir=staged,
                final_dir=self.root / "final-step-694", identity={"global_step": 694},
            )
        self.assertFalse((self.root / "final-step-694").exists())

    def test_export_final_model_records_step_kind_and_calls_save_weights_only(self):
        model = _FakeSavableModel()
        result = TC.export_final_model(
            model, checkpoint_root=self.root, identity_context={"model_revision": "rev-1", "seed": 20260810},
            global_step=694, expected_global_step=694,
        )
        self.assertEqual(Path(result["path"]), self.root / "final-step-694")
        self.assertEqual(result["identity"]["global_step"], 694)
        self.assertEqual(result["identity"]["export_kind"], TC.FINAL_EXPORT_KIND)
        self.assertEqual(result["identity"]["model_revision"], "rev-1")
        self.assertIn("created_at", result["identity"])
        self.assertEqual(len(model.save_calls), 1)
        _path, kwargs = model.save_calls[0]
        self.assertFalse(kwargs["create_model_card"])
        self.assertTrue(kwargs["safe_serialization"])

    def test_export_with_source_auto_map_restores_it_over_a_rewritten_local_form(self):
        # Mirrors what SentenceTransformer.save() does for a trust_remote_code model: it rewrites
        # config.json's auto_map to a bare local form (module.Class, no repo prefix), which is
        # exactly what triggers the transformers local-directory recursive-import bug this
        # restoration step exists to avoid.
        def writer(directory: Path) -> None:
            _write_fake_final_export_files(directory)
            (directory / "config.json").write_text(json.dumps({
                "auto_map": {"AutoConfig": "configuration_xlm_roberta.XLMRobertaFlashConfig"},
            }), encoding="utf-8")

        model = _FakeSavableModel(writer=writer)
        source_auto_map = {
            "AutoConfig": "jinaai/xlm-roberta-flash-implementation--configuration_xlm_roberta.XLMRobertaFlashConfig",
        }
        result = TC.export_final_model(
            model, checkpoint_root=self.root, identity_context={"model_revision": "rev-1"},
            global_step=694, expected_global_step=694, source_auto_map=source_auto_map,
        )
        final_config = json.loads((Path(result["path"]) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(final_config["auto_map"], source_auto_map)

    def test_export_without_source_auto_map_leaves_a_rewritten_config_untouched(self):
        def writer(directory: Path) -> None:
            _write_fake_final_export_files(directory)
            (directory / "config.json").write_text(
                json.dumps({"auto_map": {"AutoConfig": "configuration_xlm_roberta.XLMRobertaFlashConfig"}}),
                encoding="utf-8",
            )

        model = _FakeSavableModel(writer=writer)
        result = TC.export_final_model(
            model, checkpoint_root=self.root, identity_context={"model_revision": "rev-1"},
            global_step=694, expected_global_step=694, source_auto_map=None,
        )
        final_config = json.loads((Path(result["path"]) / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(final_config["auto_map"]["AutoConfig"], "configuration_xlm_roberta.XLMRobertaFlashConfig")

    def test_export_with_source_auto_map_is_a_noop_for_a_model_without_auto_map(self):
        # KURE/BAAI/Upskyy: real config.json, but no auto_map key at all - restoring must not add one.
        def writer(directory: Path) -> None:
            _write_fake_final_export_files(directory)
            (directory / "config.json").write_text(json.dumps({"hidden_size": 768}), encoding="utf-8")

        model = _FakeSavableModel(writer=writer)
        source_auto_map = {"AutoConfig": "should-never-appear"}
        result = TC.export_final_model(
            model, checkpoint_root=self.root, identity_context={"model_revision": "rev-1"},
            global_step=694, expected_global_step=694, source_auto_map=source_auto_map,
        )
        final_config = json.loads((Path(result["path"]) / "config.json").read_text(encoding="utf-8"))
        self.assertNotIn("auto_map", final_config)


class ReadBaseSnapshotAutoMapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_returns_none_without_a_local_model_path(self):
        self.assertIsNone(TC._read_base_snapshot_auto_map({}))

    def test_returns_none_when_config_json_is_missing(self):
        self.assertIsNone(TC._read_base_snapshot_auto_map({"local_model_path": str(self.root)}))

    def test_returns_none_when_auto_map_key_is_absent(self):
        (self.root / "config.json").write_text(json.dumps({"hidden_size": 768}), encoding="utf-8")
        self.assertIsNone(TC._read_base_snapshot_auto_map({"local_model_path": str(self.root)}))

    def test_returns_the_real_auto_map_dict_when_present(self):
        auto_map = {"AutoConfig": "jinaai/xlm-roberta-flash-implementation--configuration_xlm_roberta.XLMRobertaFlashConfig"}
        (self.root / "config.json").write_text(json.dumps({"auto_map": auto_map}), encoding="utf-8")
        self.assertEqual(TC._read_base_snapshot_auto_map({"local_model_path": str(self.root)}), auto_map)

    def test_expands_user_home_in_local_model_path(self):
        with mock.patch.object(Path, "expanduser", return_value=self.root):
            (self.root / "config.json").write_text(json.dumps({"auto_map": {"a": "b"}}), encoding="utf-8")
            result = TC._read_base_snapshot_auto_map({"local_model_path": "~/wherever"})
        self.assertEqual(result, {"a": "b"})

    def test_export_is_refused_when_training_stopped_short_of_the_planned_step(self):
        model = _FakeSavableModel()
        with self.assertRaises(RuntimeError) as ctx:
            TC.export_final_model(
                model, checkpoint_root=self.root, identity_context={}, global_step=650, expected_global_step=694,
            )
        self.assertIn("650", str(ctx.exception))
        self.assertIn("694", str(ctx.exception))
        self.assertEqual(model.save_calls, [])
        self.assertEqual(list(self.root.iterdir()), [])  # nothing written, checkpoints untouched

    def test_export_directory_is_never_treated_as_a_resume_checkpoint(self):
        model = _FakeSavableModel()
        TC.export_final_model(
            model, checkpoint_root=self.root, identity_context=dict(_SAMPLE_IDENTITY), global_step=694,
        )
        self.assertTrue((self.root / "final-step-694" / TC.CHECKPOINT_COMPLETE_MARKER).is_file())
        self.assertIsNone(TC._select_resume_checkpoint(self.root, _SAMPLE_IDENTITY))

    def test_export_survives_checkpoint_retention_pruning(self):
        model = _FakeSavableModel()
        TC.export_final_model(model, checkpoint_root=self.root, identity_context={}, global_step=694)
        for step in (550, 600, 650):
            staged = self.root / ".staging" / f"checkpoint-{step}"
            _write_fake_checkpoint_state_files(staged)
            TC.atomic_save_checkpoint(
                staged_dir=staged, final_dir=self.root / f"checkpoint-{step}", identity={"global_step": step},
            )
        callback = TC._AtomicCheckpointCallback(
            checkpoint_root=self.root, staging_root=self.root / ".staging", save_total_limit=1, identity_context={},
        )
        callback._enforce_retention()
        self.assertTrue((self.root / "final-step-694").is_dir())
        self.assertTrue((self.root / "checkpoint-650").is_dir())
        self.assertFalse((self.root / "checkpoint-550").exists())


class ResolvePlannedOptimizerStepsTest(unittest.TestCase):
    def test_one_epoch_over_the_frozen_groups_is_one_step_per_group(self):
        schedule = {"num_train_epochs": 1, "max_steps": -1}
        self.assertEqual(TC._resolve_planned_optimizer_steps(schedule, [32] * 694), 694)

    def test_positive_max_steps_override_wins(self):
        schedule = {"num_train_epochs": 1, "max_steps": -1}
        self.assertEqual(TC._resolve_planned_optimizer_steps(schedule, [32] * 694, max_steps=5), 5)

    def test_config_max_steps_is_used_when_no_cli_override(self):
        self.assertEqual(TC._resolve_planned_optimizer_steps({"num_train_epochs": 1, "max_steps": 12}, [32] * 694), 12)


_SAMPLE_IDENTITY = {
    "schema_version": TC.CHECKPOINT_IDENTITY_SCHEMA_VERSION,
    "training_config_sha256": "config-abc",
    "batch_plan_digest": "28809c25" + "0" * 56,
    "model_revision": "rev-1",
    "code_revision": None,
    "runtime_lock_sha256": "runtime-xyz",
    "objective": "multiple_negatives_ranking_loss",
    "precision": "bfloat16",
    "max_seq_length_applied": 512,
    "seed": 20260810,
}


class SelectResumeCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _complete_checkpoint(self, step: int, identity: dict) -> Path:
        staged = self.root / ".staging" / f"checkpoint-{step}"
        _write_fake_checkpoint_state_files(staged)
        final_dir = self.root / f"checkpoint-{step}"
        TC.atomic_save_checkpoint(staged_dir=staged, final_dir=final_dir, identity=identity)
        return final_dir

    def test_returns_none_for_missing_or_empty_directory(self):
        self.assertIsNone(TC._select_resume_checkpoint(self.root / "does-not-exist", _SAMPLE_IDENTITY))
        self.root.mkdir(exist_ok=True)
        self.assertIsNone(TC._select_resume_checkpoint(self.root, _SAMPLE_IDENTITY))

    def test_skips_a_staged_but_never_completed_checkpoint(self):
        never_completed = self.root / "checkpoint-10"
        _write_fake_checkpoint_state_files(never_completed)  # state files but no marker/identity
        self.assertIsNone(TC._select_resume_checkpoint(self.root, _SAMPLE_IDENTITY))

    def test_returns_newest_complete_checkpoint_matching_the_current_identity(self):
        self._complete_checkpoint(10, _SAMPLE_IDENTITY)
        newest = self._complete_checkpoint(20, _SAMPLE_IDENTITY)
        result = TC._select_resume_checkpoint(self.root, _SAMPLE_IDENTITY)
        self.assertEqual(Path(result), newest)

    def test_raises_on_identity_mismatch_instead_of_silently_ignoring_it(self):
        drifted_identity = {**_SAMPLE_IDENTITY, "model_revision": "a-different-revision", "seed": 1}
        self._complete_checkpoint(10, drifted_identity)
        with self.assertRaises(ValueError) as ctx:
            TC._select_resume_checkpoint(self.root, _SAMPLE_IDENTITY)
        message = str(ctx.exception)
        self.assertIn("model_revision", message)
        self.assertIn("seed", message)

    def test_each_identity_mismatch_field_is_individually_detected(self):
        for field in TC.IDENTITY_MISMATCH_FIELDS:
            with self.subTest(field=field):
                tmp = tempfile.TemporaryDirectory()
                try:
                    root = Path(tmp.name)
                    drifted = {**_SAMPLE_IDENTITY, field: "definitely-not-the-current-value"}
                    staged = root / ".staging" / "checkpoint-1"
                    _write_fake_checkpoint_state_files(staged)
                    TC.atomic_save_checkpoint(staged_dir=staged, final_dir=root / "checkpoint-1", identity=drifted)
                    with self.assertRaises(ValueError):
                        TC._select_resume_checkpoint(root, _SAMPLE_IDENTITY)
                finally:
                    tmp.cleanup()


class EnforceModeFlagContractTest(unittest.TestCase):
    """TASK-07R1 production-training gate: --mode train requires --enable-production-training."""

    def test_mode_train_without_the_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            TC._enforce_mode_flag_contract("train", False, None)

    def test_mode_train_with_the_flag_is_accepted(self):
        TC._enforce_mode_flag_contract("train", True, None)  # must not raise

    def test_non_train_modes_reject_the_flag(self):
        for mode in ("validate", "tokenizer_stats", "smoke"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    TC._enforce_mode_flag_contract(mode, True, None)

    def test_smoke_mode_without_the_flag_and_within_the_ceiling_is_accepted(self):
        TC._enforce_mode_flag_contract("smoke", False, TC.MAX_SMOKE_STEPS)  # must not raise

    def test_smoke_mode_cannot_exceed_the_hard_step_ceiling(self):
        with self.assertRaises(ValueError):
            TC._enforce_mode_flag_contract("smoke", False, TC.MAX_SMOKE_STEPS + 1)

    def test_smoke_mode_far_beyond_the_ceiling_is_still_rejected(self):
        with self.assertRaises(ValueError):
            TC._enforce_mode_flag_contract("smoke", False, 100_000)

    def test_validate_and_tokenizer_stats_ignore_max_steps_entirely(self):
        TC._enforce_mode_flag_contract("validate", False, 100_000)  # must not raise
        TC._enforce_mode_flag_contract("tokenizer_stats", False, 100_000)  # must not raise


class ResumeEquivalenceTest(unittest.TestCase):
    """Deterministic proof that TC's own atomic checkpoint save/resume mechanics reproduce an
    uninterrupted run's final state exactly, using a tiny real torch model/optimizer/scheduler -
    no sentence_transformers/datasets/accelerate and no GPU, per TASK-07R1's synthetic-only bound.
    """

    IDENTITY_EXTRA = {
        "schema_version": TC.CHECKPOINT_IDENTITY_SCHEMA_VERSION, "training_config_sha256": "cfg-digest",
        "batch_plan_digest": "batch-digest", "model_revision": "rev-1", "code_revision": None,
        "runtime_lock_sha256": "runtime-digest", "objective": "mnrl", "precision": "float32",
        "max_seq_length_applied": 512, "seed": 123,
    }

    def _build(self, seed):
        import torch
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        return model, optimizer, scheduler

    def _run_steps(self, model, optimizer, scheduler, start_step, end_step, seed):
        import torch
        loss_fn = torch.nn.MSELoss()
        losses = []
        for step in range(start_step, end_step):
            generator = torch.Generator().manual_seed(seed * 1000 + step)
            x = torch.randn(4, 4, generator=generator)
            y = torch.randn(4, 4, generator=generator)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(round(loss.item(), 8))
        return losses

    def _state_digest(self, state_dict) -> str:
        import io
        import torch
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    def _save_checkpoint(self, checkpoint_root, model, optimizer, scheduler, step):
        import safetensors.torch
        import torch
        staged = checkpoint_root / ".staging" / f"checkpoint-{step}"
        staged.mkdir(parents=True)
        # Real safetensors bytes, not torch.save() bytes under a .safetensors name:
        # torch.load() dispatches on that suffix to the safetensors reader, so a
        # mislabeled pickle is unreadable on the resume path this test exists to prove.
        safetensors.torch.save_file(model.state_dict(), str(staged / "model.safetensors"))
        torch.save(optimizer.state_dict(), staged / "optimizer.pt")
        torch.save(scheduler.state_dict(), staged / "scheduler.pt")
        torch.save({"cpu": torch.random.get_rng_state()}, staged / "rng_state.pth")
        (staged / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        identity = {**self.IDENTITY_EXTRA, "global_step": step}
        return TC.atomic_save_checkpoint(
            staged_dir=staged, final_dir=checkpoint_root / f"checkpoint-{step}", identity=identity,
        )

    def test_interrupted_and_resumed_run_matches_an_uninterrupted_run(self):
        import torch
        seed = 123

        model_a, opt_a, sched_a = self._build(seed)
        losses_a = self._run_steps(model_a, opt_a, sched_a, 0, 6, seed)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)

            model_b, opt_b, sched_b = self._build(seed)
            losses_first_half = self._run_steps(model_b, opt_b, sched_b, 0, 3, seed)
            checkpoint_dir = self._save_checkpoint(checkpoint_root, model_b, opt_b, sched_b, 3)

            resumed = TC._select_resume_checkpoint(
                checkpoint_root, {k: v for k, v in self.IDENTITY_EXTRA.items() if k in TC.IDENTITY_MISMATCH_FIELDS},
            )
            self.assertEqual(Path(resumed), checkpoint_dir)

            model_c, opt_c, sched_c = self._build(seed)  # simulates a fresh process before loading
            model_c.load_state_dict(torch.load(checkpoint_dir / "model.safetensors", weights_only=True))
            opt_c.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", weights_only=True))
            sched_c.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", weights_only=True))
            rng = torch.load(checkpoint_dir / "rng_state.pth", weights_only=True)
            torch.random.set_rng_state(rng["cpu"])

            losses_second_half = self._run_steps(model_c, opt_c, sched_c, 3, 6, seed)

        losses_b = losses_first_half + losses_second_half
        self.assertEqual(losses_a, losses_b)
        self.assertEqual(self._state_digest(model_a.state_dict()), self._state_digest(model_c.state_dict()))
        self.assertEqual(self._state_digest(opt_a.state_dict()), self._state_digest(opt_c.state_dict()))
        self.assertEqual(self._state_digest(sched_a.state_dict()), self._state_digest(sched_c.state_dict()))
        self.assertEqual(sched_a.last_epoch, sched_c.last_epoch)

    def test_resume_rejects_a_checkpoint_from_a_differently_configured_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            model, opt, sched = self._build(123)
            self._run_steps(model, opt, sched, 0, 2, 123)
            self._save_checkpoint(checkpoint_root, model, opt, sched, 2)

            current_run_identity = {
                k: v for k, v in self.IDENTITY_EXTRA.items() if k in TC.IDENTITY_MISMATCH_FIELDS
            }
            current_run_identity["seed"] = 999  # a differently-configured run
            with self.assertRaises(ValueError):
                TC._select_resume_checkpoint(checkpoint_root, current_run_identity)


class _FakeParam:
    def __init__(self, count):
        self._count = count

    def numel(self):
        return self._count


class _FakeTrainableModel:
    def __init__(self):
        self.prompts = {"retrieval.query": "Represent the query: ", "retrieval.passage": "Represent the passage: "}
        self.save_calls: list[tuple[str, dict]] = []
        self.forward_calls = []

    def parameters(self):
        return [_FakeParam(100), _FakeParam(100), _FakeParam(100)]

    def save(self, path, **kwargs):
        self.save_calls.append((str(path), kwargs))
        _write_fake_final_export_files(Path(path))

    def __call__(self, features, task=None):
        self.forward_calls.append((features, task))
        return {"sentence_embedding": f"embedding::{features}::{task}"}


class _FakeTrainerState:
    def __init__(self):
        self.global_step = 0
        self.epoch = 0.0
        self.log_history = []


class _FakeSentenceTransformerTrainer:
    """train() advances global_step to ``steps_to_run``, so run_training()'s post-training
    final-export contract can be exercised for both a complete and a short run."""

    steps_to_run = 5

    def __init__(self, model=None, args=None, train_dataset=None, loss=None, callbacks=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.loss = loss
        self.callbacks = callbacks or []
        self.state = _FakeTrainerState()
        self.train_called_with = "not_called"

    def train(self, resume_from_checkpoint=None):
        self.train_called_with = resume_from_checkpoint
        self.state.global_step = type(self).steps_to_run


class _FakeTrainingArguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeMultipleNegativesRankingLoss:
    def __init__(self, model, scale=None):
        self.model = model
        self.scale = scale
        self.compute_calls = []

    def compute_loss_from_embeddings(self, embeddings, labels):
        self.compute_calls.append((list(embeddings), labels))
        return "fake-loss"

    def get_config_dict(self):
        return {"scale": self.scale}


class _FakeTransformersTrainerCallback:
    """Stand-in for transformers.TrainerCallback: a plain base class is enough for subclassing."""


class RunTrainingWiringTest(unittest.TestCase):
    """Fakes the entire training stack to prove run_training() assembles its arguments correctly -
    no real optimizer step runs here, which is out of scope for TASK-07R1.
    """

    def _fake_modules(self):
        losses_module = types.SimpleNamespace(MultipleNegativesRankingLoss=_FakeMultipleNegativesRankingLoss)
        sentence_transformer_submodule = types.SimpleNamespace(losses=losses_module)
        sentence_transformers_module = types.SimpleNamespace(
            SentenceTransformerTrainer=_FakeSentenceTransformerTrainer,
            SentenceTransformerTrainingArguments=_FakeTrainingArguments,
            sentence_transformer=sentence_transformer_submodule,
        )
        transformers_module = types.SimpleNamespace(TrainerCallback=_FakeTransformersTrainerCallback)
        return {
            "sentence_transformers": sentence_transformers_module,
            "sentence_transformers.sentence_transformer": sentence_transformer_submodule,
            "sentence_transformers.sentence_transformer.losses": losses_module,
            "transformers": transformers_module,
        }

    def _config(self):
        return {
            "schedule": {"seed": 1, "num_train_epochs": 1, "max_steps": -1},
            "optimizer": {
                "learning_rate": 1e-5, "weight_decay": 0.0, "warmup_ratio": 0.0,
                "lr_scheduler_type": "linear", "max_grad_norm": 1.0,
            },
            "runtime": {"precision": "bfloat16", "gradient_checkpointing": True},
            "checkpoint": {"save_steps": 10, "save_total_limit": 2},
            "objective": {"loss": "multiple_negatives_ranking_loss", "scale": 20.0},
            "models": {"m": {"per_device_train_batch_size": 4}},
            "batch_plan": {"expected_digest": "digest-abc"},
        }

    def _metadata(self):
        return {
            "dtype_applied": "bfloat16", "revision_applied": "rev-1", "code_revision_applied": None,
            "max_seq_length_applied": 512, "license_label": None, "query_adapter_applied": None,
            "passage_adapter_applied": None,
        }

    def _metadata_with_lora(self):
        return {
            **self._metadata(),
            "trust_remote_code_applied": True,
            "query_adapter_applied": "retrieval.query", "passage_adapter_applied": "retrieval.passage",
            "query_lora_task_applied": "retrieval.query", "passage_lora_task_applied": "retrieval.passage",
            "lora_task_routing_behaviorally_verified": True,
        }

    def test_wires_staging_output_dir_batch_sampler_prompts_and_callback(self):
        model = _FakeTrainableModel()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp) / "m"
            with mock.patch.dict(sys.modules, self._fake_modules()):
                trainer, disk_info = TC.run_training(
                    model, self._metadata(), dataset=["row"], group_sizes=[1], config=self._config(),
                    checkpoint_root=checkpoint_root, model_key="m", git_commit="deadbeef", max_steps=5,
                )
        self.assertEqual(trainer.args.output_dir, str(checkpoint_root / ".staging"))
        self.assertIsInstance(trainer.args.batch_sampler, TC._FrozenBatchSamplerFactory)
        self.assertIsNone(trainer.args.prompts)
        self.assertEqual(len(trainer.callbacks), 1)
        self.assertIsNone(trainer.train_called_with)  # fresh checkpoint_root: nothing to resume
        self.assertIn("disk_estimate", disk_info)
        self.assertIn("checkpoint_identity_context", disk_info)
        self.assertEqual(disk_info["checkpoint_identity_context"]["model_key"], "m")
        self.assertEqual(disk_info["checkpoint_identity_context"]["git_commit"], "deadbeef")

    def test_final_state_is_exported_with_the_run_identity_after_training(self):
        model = _FakeTrainableModel()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp) / "m"
            with mock.patch.dict(sys.modules, self._fake_modules()):
                _trainer, disk_info = TC.run_training(
                    model, self._metadata(), dataset=["row"], group_sizes=[1], config=self._config(),
                    checkpoint_root=checkpoint_root, model_key="m", git_commit="deadbeef", max_steps=5,
                )
            export = disk_info["final_export"]
            self.assertEqual(disk_info["planned_optimizer_steps"], 5)
            self.assertEqual(export["global_step"], 5)
            self.assertEqual(Path(export["path"]), checkpoint_root / "final-step-5")
            self.assertTrue((checkpoint_root / "final-step-5" / TC.CHECKPOINT_COMPLETE_MARKER).is_file())
            identity = json.loads(
                (checkpoint_root / "final-step-5" / TC.CHECKPOINT_IDENTITY_FILENAME).read_text(encoding="utf-8")
            )
            # The export carries the same run identity the periodic checkpoints do,
            # so a dev result can be tied back to this exact config/data/model/runtime.
            self.assertEqual(identity["batch_plan_digest"], "digest-abc")
            self.assertEqual(identity["model_revision"], "rev-1")
            self.assertEqual(identity["git_commit"], "deadbeef")
            self.assertEqual(identity["export_kind"], TC.FINAL_EXPORT_KIND)
            self.assertEqual(len(model.save_calls), 1)

    def test_a_run_that_stops_short_exports_nothing(self):
        model = _FakeTrainableModel()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp) / "m"
            with mock.patch.dict(sys.modules, self._fake_modules()):
                with mock.patch.object(_FakeSentenceTransformerTrainer, "steps_to_run", 3):
                    with self.assertRaises(RuntimeError):
                        TC.run_training(
                            model, self._metadata(), dataset=["row"], group_sizes=[1], config=self._config(),
                            checkpoint_root=checkpoint_root, model_key="m", git_commit="deadbeef", max_steps=5,
                        )
            self.assertEqual(model.save_calls, [])
            self.assertFalse((checkpoint_root / "final-step-3").exists())

    def test_declared_expected_optimizer_steps_must_match_the_real_schedule(self):
        config = self._config()
        config["schedule"]["expected_optimizer_steps"] = 694
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, self._fake_modules()):
                with self.assertRaises(ValueError) as ctx:
                    TC.run_training(
                        _FakeTrainableModel(), self._metadata(), dataset=["row", "row"], group_sizes=[1, 1],
                        config=config, checkpoint_root=Path(tmp) / "m", model_key="m", git_commit="deadbeef",
                    )
        self.assertIn("expected_optimizer_steps", str(ctx.exception))

    def test_declared_expected_optimizer_steps_that_match_are_accepted(self):
        config = self._config()
        config["schedule"]["expected_optimizer_steps"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, self._fake_modules()):
                with mock.patch.object(_FakeSentenceTransformerTrainer, "steps_to_run", 2):
                    _trainer, disk_info = TC.run_training(
                        _FakeTrainableModel(), self._metadata(), dataset=["a", "b"], group_sizes=[1, 1],
                        config=config, checkpoint_root=Path(tmp) / "m", model_key="m", git_commit="deadbeef",
                    )
        self.assertEqual(disk_info["planned_optimizer_steps"], 2)
        self.assertEqual(disk_info["final_export"]["global_step"], 2)

    def test_non_jina_metadata_leaves_the_loss_unwrapped(self):
        model = _FakeTrainableModel()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, self._fake_modules()):
                trainer, _disk_info = TC.run_training(
                    model, self._metadata(), dataset=["row"], group_sizes=[1], config=self._config(),
                    checkpoint_root=Path(tmp) / "m", model_key="m", git_commit="deadbeef", max_steps=5,
                )
        self.assertIsInstance(trainer.loss, _FakeMultipleNegativesRankingLoss)
        self.assertFalse(hasattr(trainer.loss, "query_task"))

    def test_jina_lora_task_metadata_wraps_the_loss_and_attests_identity(self):
        model = _FakeTrainableModel()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, self._fake_modules()):
                trainer, disk_info = TC.run_training(
                    model, self._metadata_with_lora(), dataset=["row"], group_sizes=[1], config=self._config(),
                    checkpoint_root=Path(tmp) / "m", model_key="m", git_commit="deadbeef", max_steps=5,
                )
        self.assertNotIsInstance(trainer.loss, _FakeMultipleNegativesRankingLoss)
        self.assertEqual(trainer.loss.query_task, "retrieval.query")
        self.assertEqual(trainer.loss.passage_task, "retrieval.passage")
        self.assertIs(trainer.loss.model, model)
        identity = disk_info["checkpoint_identity_context"]
        self.assertTrue(identity["trust_remote_code"])
        self.assertEqual(identity["query_adapter"], "retrieval.query")
        self.assertEqual(identity["passage_adapter"], "retrieval.passage")
        self.assertEqual(identity["query_lora_task"], "retrieval.query")
        self.assertEqual(identity["passage_lora_task"], "retrieval.passage")

    def test_task_aware_loss_calls_query_then_passage_passage_in_order(self):
        model = _FakeTrainableModel()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, self._fake_modules()):
                trainer, _disk_info = TC.run_training(
                    model, self._metadata_with_lora(), dataset=["row"], group_sizes=[1], config=self._config(),
                    checkpoint_root=Path(tmp) / "m", model_key="m", git_commit="deadbeef", max_steps=5,
                )
        # Mirrors the frozen batch plan's dataset column order (query, positive, negative):
        # sentence_features[0] is the anchor/query, every later entry is a document.
        result = trainer.loss(["query-features", "positive-features", "negative-features"], labels="labels-sentinel")
        self.assertEqual(result, "fake-loss")
        self.assertEqual(
            [task for _features, task in model.forward_calls],
            ["retrieval.query", "retrieval.passage", "retrieval.passage"],
        )
        self.assertEqual(
            [features for features, _task in model.forward_calls],
            ["query-features", "positive-features", "negative-features"],
        )
        base_loss = trainer.loss.base_loss
        embeddings_passed, labels_passed = base_loss.compute_calls[0]
        self.assertEqual(labels_passed, "labels-sentinel")
        self.assertEqual(embeddings_passed, [
            "embedding::query-features::retrieval.query",
            "embedding::positive-features::retrieval.passage",
            "embedding::negative-features::retrieval.passage",
        ])

    def test_asymmetric_lora_task_metadata_is_rejected(self):
        metadata = {**self._metadata_with_lora(), "passage_lora_task_applied": None}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, self._fake_modules()):
                with self.assertRaises(ValueError) as ctx:
                    TC.run_training(
                        _FakeTrainableModel(), metadata, dataset=["row"], group_sizes=[1], config=self._config(),
                        checkpoint_root=Path(tmp) / "m", model_key="m", git_commit="deadbeef", max_steps=5,
                    )
        self.assertIn("query_lora_task_applied", str(ctx.exception))

    def test_final_export_restores_the_base_snapshots_auto_map(self):
        class _FakeJinaTrainableModel(_FakeTrainableModel):
            def save(self, path, **kwargs):
                self.save_calls.append((str(path), kwargs))
                _write_fake_final_export_files(Path(path))
                (Path(path) / "config.json").write_text(json.dumps({
                    "auto_map": {"AutoConfig": "configuration_xlm_roberta.XLMRobertaFlashConfig"},
                }), encoding="utf-8")

        model = _FakeJinaTrainableModel()
        config = self._config()
        with tempfile.TemporaryDirectory() as snapshot_dir:
            source_auto_map = {
                "AutoConfig": (
                    "jinaai/xlm-roberta-flash-implementation--configuration_xlm_roberta.XLMRobertaFlashConfig"
                ),
            }
            (Path(snapshot_dir) / "config.json").write_text(
                json.dumps({"auto_map": source_auto_map}), encoding="utf-8",
            )
            config["models"]["m"]["local_model_path"] = snapshot_dir
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint_root = Path(tmp) / "m"
                with mock.patch.dict(sys.modules, self._fake_modules()):
                    _trainer, disk_info = TC.run_training(
                        model, self._metadata(), dataset=["row"], group_sizes=[1], config=config,
                        checkpoint_root=checkpoint_root, model_key="m", git_commit="deadbeef", max_steps=5,
                    )
                export_path = Path(disk_info["final_export"]["path"])
                final_config = json.loads((export_path / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(final_config["auto_map"], source_auto_map)

    def test_unverified_lora_task_routing_is_rejected(self):
        metadata = {**self._metadata_with_lora(), "lora_task_routing_behaviorally_verified": False}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, self._fake_modules()):
                with self.assertRaises(ValueError) as ctx:
                    TC.run_training(
                        _FakeTrainableModel(), metadata, dataset=["row"], group_sizes=[1], config=self._config(),
                        checkpoint_root=Path(tmp) / "m", model_key="m", git_commit="deadbeef", max_steps=5,
                    )
        self.assertIn("behaviorally verify", str(ctx.exception))


class _FakeTorchDtype:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return f"torch.{self._name}"

    def __eq__(self, other):
        return isinstance(other, _FakeTorchDtype) and self._name == other._name

    def __hash__(self):
        return hash(("_FakeTorchDtype", self._name))


def _fake_torch_module():
    return types.SimpleNamespace(
        float32=_FakeTorchDtype("float32"), float16=_FakeTorchDtype("float16"), bfloat16=_FakeTorchDtype("bfloat16"),
    )


class _FakeParameter:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakePoolingModule:
    def __init__(self, pooling_mode):
        self.pooling_mode = pooling_mode


class _FakeJinaTransformerModule:
    """Stand-in for custom_st.Transformer at a loaded model's ``[0]`` index - mirrors
    tests/test_evaluate_retrieval.py's fixture of the same shape, since
    build_trainable_model() reuses evaluate_retrieval.SentenceTransformerEncoder unmodified and
    therefore exercises the exact same behavioral LoRA-task-routing probe.
    """

    def __init__(self, lora_adaptations):
        self._lora_adaptations = list(lora_adaptations)

    def forward(self, features, task=None, **kwargs):
        return {"token_embeddings": features, "task_applied": task}


class _FakeSentenceTransformerModel:
    def __init__(self, pooling_mode="mean", max_seq_length=512, prompts=None, lora_adaptations=None):
        self.max_seq_length = max_seq_length
        self._dtype = _FakeTorchDtype("float32")
        self._lora_adaptations = lora_adaptations
        first_module = (
            _FakeJinaTransformerModule(lora_adaptations) if lora_adaptations is not None else types.SimpleNamespace()
        )
        self._modules = {"0": first_module, "1": _FakePoolingModule(pooling_mode)}
        self.prompts = prompts
        self.train_called = False

    def __getitem__(self, index):
        return self._modules[str(index)]

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self._dtype = dtype
        return self

    def parameters(self):
        yield _FakeParameter(self._dtype)

    def train(self):
        self.train_called = True
        return self

    def encode(self, texts, prompt_name=None, task=None, **kwargs):
        if self._lora_adaptations is None:
            raise AssertionError("encode() must not run unless the fixture declares lora_adaptations")
        self[0].forward({"texts": list(texts)}, task=task)
        return [[0.0, 0.0] for _ in texts]


def _fake_sentence_transformers_module(constructor):
    return types.SimpleNamespace(SentenceTransformer=constructor, __version__="9.9.9-fake")


class BuildTrainableModelTest(unittest.TestCase):
    """Proves scripts/train_contrastive.py reuses evaluate_retrieval's fail-closed contract, not a copy of it."""

    UPSKYY_SETTINGS = {
        "model_name_or_path": "upskyy/bge-m3-korean",
        "local_model_path": "/tmp/fake-upskyy",
        "revision": None, "trust_remote_code": False, "code_revision": None,
        "query_adapter": None, "passage_adapter": None, "pooling": "mean", "max_seq_length": 512,
        "dtype": "bfloat16", "license_label": "LICENSE UNKNOWN — INTERNAL EXPERIMENT ONLY",
    }

    def test_upskyy_constructs_with_license_label_and_no_adapters(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel(pooling_mode="mean", max_seq_length=512))
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            model, metadata = TC.build_trainable_model(self.UPSKYY_SETTINGS)
        self.assertTrue(model.train_called)
        self.assertEqual(metadata["license_label"], "LICENSE UNKNOWN — INTERNAL EXPERIMENT ONLY")
        self.assertFalse(metadata["trust_remote_code_applied"])
        self.assertIsNone(metadata["query_adapter_applied"])
        self.assertEqual(metadata["max_seq_length_applied"], 512)

    def test_upskyy_requesting_trust_remote_code_is_refused(self):
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        settings = {**self.UPSKYY_SETTINGS, "trust_remote_code": True}
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                TC.build_trainable_model(settings)
        self.assertIn("KURE and BAAI", str(ctx.exception))
        constructor.assert_not_called()

    def test_upskyy_native_cap_below_requested_length_is_never_silently_truncated(self):
        # Simulates a hypothetical snapshot whose applied max_seq_length (256) is
        # already below the config's requested 512 - the failure this project's
        # own documented Upskyy 512-vs-8194 case is designed to guard against.
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel(max_seq_length=256))
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError) as ctx:
                TC.build_trainable_model(self.UPSKYY_SETTINGS)
        self.assertIn("silently truncate", str(ctx.exception))

    def test_kure_and_baai_reject_trust_remote_code_too(self):
        for model_id in ("nlpai-lab/KURE-v1", "BAAI/bge-m3"):
            constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
            settings = {
                "model_name_or_path": model_id, "local_model_path": "/tmp/fake-model",
                "trust_remote_code": True, "pooling": "cls", "max_seq_length": 512, "dtype": "bfloat16",
            }
            with mock.patch.dict(sys.modules, {
                "sentence_transformers": _fake_sentence_transformers_module(constructor),
                "torch": _fake_torch_module(),
            }):
                with self.assertRaises(ValueError):
                    TC.build_trainable_model(settings)
            constructor.assert_not_called()

    def test_jina_with_the_exact_reviewed_pin_constructs_and_resolves_prompts(self):
        pin = TC.EV.JINA_REMOTE_CODE_PIN
        fake_model = _FakeSentenceTransformerModel(
            pooling_mode="mean",
            prompts={pin["query_adapter"]: "Represent the query: ", pin["passage_adapter"]: "Represent the passage: "},
            lora_adaptations=[pin["query_adapter"], pin["passage_adapter"]],
        )
        constructor = mock.Mock(return_value=fake_model)
        settings = {
            "model_name_or_path": pin["model_id"], "local_model_path": "/tmp/fake-jina",
            "revision": pin["model_revision"], "trust_remote_code": True, "code_revision": pin["code_revision"],
            "query_adapter": pin["query_adapter"], "passage_adapter": pin["passage_adapter"],
            "pooling": "mean", "max_seq_length": 512, "dtype": "bfloat16",
        }
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            model, metadata = TC.build_trainable_model(settings)
            prompts = TC.resolve_training_prompts(model, metadata)
        self.assertTrue(metadata["trust_remote_code_applied"])
        self.assertEqual(prompts, {
            "query": "Represent the query: ", "positive": "Represent the passage: ", "negative": "Represent the passage: ",
        })
        # query_adapter_applied/passage_adapter_applied alone are prompt-name selection evidence;
        # query_lora_task_applied/passage_lora_task_applied are the separate, behaviorally verified
        # proof that the real LoRA adapter (not just its text prompt) was reachable.
        self.assertEqual(metadata["query_lora_task_applied"], pin["query_adapter"])
        self.assertEqual(metadata["passage_lora_task_applied"], pin["passage_adapter"])
        self.assertTrue(metadata["lora_task_routing_behaviorally_verified"])

    def test_jina_with_a_drifted_revision_is_refused_before_construction(self):
        pin = TC.EV.JINA_REMOTE_CODE_PIN
        constructor = mock.Mock(return_value=_FakeSentenceTransformerModel())
        settings = {
            "model_name_or_path": pin["model_id"], "local_model_path": "/tmp/fake-jina",
            "revision": "not-the-reviewed-revision", "trust_remote_code": True,
            "code_revision": pin["code_revision"], "query_adapter": pin["query_adapter"],
            "passage_adapter": pin["passage_adapter"], "pooling": "mean", "max_seq_length": 512, "dtype": "bfloat16",
        }
        with mock.patch.dict(sys.modules, {
            "sentence_transformers": _fake_sentence_transformers_module(constructor),
            "torch": _fake_torch_module(),
        }):
            with self.assertRaises(ValueError):
                TC.build_trainable_model(settings)
        constructor.assert_not_called()


class ResolveTrainingPromptsTest(unittest.TestCase):
    def test_no_adapters_returns_none(self):
        model = types.SimpleNamespace(prompts={})
        self.assertIsNone(TC.resolve_training_prompts(model, {"query_adapter_applied": None, "passage_adapter_applied": None}))

    def test_adapters_are_mapped_to_dataset_columns(self):
        model = types.SimpleNamespace(prompts={"q": "Q: ", "p": "P: "})
        prompts = TC.resolve_training_prompts(model, {"query_adapter_applied": "q", "passage_adapter_applied": "p"})
        self.assertEqual(prompts, {"query": "Q: ", "positive": "P: ", "negative": "P: "})


class BuildTrainingDatasetTest(unittest.TestCase):
    def test_builds_a_dataset_with_the_expected_columns_in_frozen_order(self):
        rows = [
            {"query": "q1", "positive": "p1", "negative": "n1"},
            {"query": "q2", "positive": "p2", "negative": "n2"},
        ]
        dataset = TC.build_training_dataset(rows)
        self.assertEqual(list(dataset.column_names), ["query", "positive", "negative"])
        self.assertEqual(dataset["query"], ["q1", "q2"])
        self.assertEqual(dataset["negative"], ["n1", "n2"])

    def test_missing_datasets_package_raises_an_actionable_runtime_error(self):
        with mock.patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaises(RuntimeError):
                TC.build_training_dataset([{"query": "q", "positive": "p", "negative": "n"}])


class CollectEnvironmentManifestTest(unittest.TestCase):
    def test_reports_git_commit_and_python_version_for_the_real_repository(self):
        manifest = TC.collect_environment_manifest(PROJECT_ROOT)
        expected_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        self.assertEqual(manifest["git_commit"], expected_commit)
        self.assertIn("python_version", manifest)
        self.assertIsInstance(manifest.get("cuda_available", False), bool)


class TrainContrastiveConfigContractTest(unittest.TestCase):
    """Static checks on the real configs/train_contrastive_base.json - no data/, no GPU, no network."""

    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_all_four_task06_model_families_are_present(self):
        self.assertEqual(set(self.config["models"]), REQUIRED_MODEL_KEYS)

    def test_upskyy_is_local_only_and_labeled_internal_experiment(self):
        upskyy = self.config["models"]["upskyy_bge_m3_korean"]
        self.assertFalse(upskyy["trust_remote_code"])
        self.assertEqual(upskyy["license_label"], "LICENSE UNKNOWN — INTERNAL EXPERIMENT ONLY")
        self.assertEqual(upskyy["pooling"], "mean")
        self.assertEqual(upskyy["max_seq_length"], 512)
        self.assertIsNone(upskyy["query_adapter"])
        self.assertIsNone(upskyy["passage_adapter"])

    def test_kure_and_baai_do_not_request_remote_code(self):
        for key in ("kure_v1", "baai_bge_m3"):
            self.assertFalse(self.config["models"][key]["trust_remote_code"])

    def test_jina_settings_match_the_evaluator_pin_exactly(self):
        pin = TC.EV.JINA_REMOTE_CODE_PIN
        jina = self.config["models"]["jina_embeddings_v3"]
        self.assertEqual(jina["model_name_or_path"], pin["model_id"])
        self.assertEqual(jina["revision"], pin["model_revision"])
        self.assertEqual(jina["code_revision"], pin["code_revision"])
        self.assertEqual(jina["query_adapter"], pin["query_adapter"])
        self.assertEqual(jina["passage_adapter"], pin["passage_adapter"])

    def test_common_runner_contract_is_identical_across_all_four_models(self):
        shared_fields = ("per_device_train_batch_size", "dtype", "normalize_embeddings", "similarity", "max_seq_length")
        values = {field: {self.config["models"][key][field] for key in REQUIRED_MODEL_KEYS} for field in shared_fields}
        for field, distinct_values in values.items():
            self.assertEqual(len(distinct_values), 1, f"{field} differs across model families: {distinct_values}")

    def test_batch_plan_digest_is_a_64_character_hex_string(self):
        digest = self.config["batch_plan"]["expected_digest"]
        self.assertEqual(len(digest), 64)
        int(digest, 16)  # raises ValueError if not valid hex


class Task08aKureConfigContractTest(unittest.TestCase):
    """Static checks that configs/train_contrastive_task08a_kure.json still carries the exact
    frozen TASK-08A contract - no data/, no GPU, no network. A frozen value may not be changed
    after loss or dev results are observed, so a drifted config must fail here first.
    """

    FROZEN_TRAINING_CONTRACT = {
        "batch_plan": {
            "expected_digest": "28809c25fb0898a6a682c9bcfb0f7a9654e4291c9f0c3aed3ebc0fafe554e708",
            "expected_rows": 22204,
            "expected_batch_groups": 694,
            "train_only": True,
        },
        "objective": {"loss": "multiple_negatives_ranking_loss", "similarity": "cosine", "scale": 20.0},
        "optimizer": {
            "name": "adamw_torch", "learning_rate": 2e-05, "weight_decay": 0.01, "warmup_ratio": 0.1,
            "lr_scheduler_type": "linear", "max_grad_norm": 1.0,
        },
        "schedule": {"seed": 20260810, "num_train_epochs": 1, "expected_optimizer_steps": 694},
        "runtime": {"precision": "bf16", "gradient_checkpointing": True},
        "checkpoint": {"save_steps": 50, "save_total_limit": 3},
    }
    FROZEN_MODEL_CONTRACT = {
        "model_name_or_path": "nlpai-lab/KURE-v1",
        "revision": "d14c8a9423946e268a0c9952fecf3a7aabd73bd9",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "pooling": "cls",
        "max_seq_length": 512,
        "per_device_train_batch_size": 32,
        "normalize_embeddings": True,
        "similarity": "cosine",
    }

    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (PROJECT_ROOT / "configs" / "train_contrastive_task08a_kure.json").read_text(encoding="utf-8")
        )
        cls.eval_config = json.loads(
            (PROJECT_ROOT / "configs" / "eval_contrastive_task08a_kure.json").read_text(encoding="utf-8")
        )

    def test_every_frozen_training_value_is_preserved(self):
        for section, expected in self.FROZEN_TRAINING_CONTRACT.items():
            for key, value in expected.items():
                with self.subTest(section=section, key=key):
                    self.assertEqual(self.config[section][key], value)

    def test_only_kure_is_authorized_for_this_task(self):
        self.assertEqual(set(self.config["models"]), {"kure_v1"})
        self.assertEqual(self.config["_authorized_model_keys"], ["kure_v1"])

    def test_every_frozen_kure_model_value_is_preserved(self):
        kure = self.config["models"]["kure_v1"]
        for key, value in self.FROZEN_MODEL_CONTRACT.items():
            with self.subTest(key=key):
                self.assertEqual(kure[key], value)
        self.assertIn("KURE-v1/snapshots/d14c8a9423946e268a0c9952fecf3a7aabd73bd9", kure["local_model_path"])

    def test_schedule_and_batch_groups_agree_on_694_steps(self):
        schedule = self.config["schedule"]
        planned = TC._resolve_planned_optimizer_steps(
            schedule, [32] * self.config["batch_plan"]["expected_batch_groups"]
        )
        self.assertEqual(planned, schedule["expected_optimizer_steps"])
        self.assertEqual(planned, 694)

    def test_shares_the_base_config_batch_plan_source(self):
        base = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for key in ("dir", "batches_file", "summary_file", "source_run_id", "expected_digest"):
            self.assertEqual(self.config["batch_plan"][key], base["batch_plan"][key])

    def test_checkpoint_root_is_outside_the_repository_and_untracked(self):
        root = Path(self.config["checkpoint"]["root_dir"]).expanduser()
        self.assertFalse(str(root).startswith(str(PROJECT_ROOT)))

    def test_dev_eval_config_keeps_the_test_split_locked_and_dev_only(self):
        self.assertEqual(self.eval_config["locked_splits"], ["test"])
        self.assertEqual([task["split"] for task in self.eval_config["tasks"]], ["dev", "dev"])
        self.assertEqual({task["name"] for task in self.eval_config["tasks"]}, {"exam_mcq", "ko_passage_strict"})

    def test_dev_eval_config_matches_the_zero_shot_comparison_contract(self):
        zero_shot = json.loads(
            (PROJECT_ROOT / "configs" / "eval_zero_shot_dev.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.eval_config["cutoffs"], zero_shot["cutoffs"])
        self.assertEqual(self.eval_config["bootstrap"]["iterations"], zero_shot["bootstrap"]["iterations"])
        self.assertEqual(self.eval_config["bootstrap"]["confidence"], zero_shot["bootstrap"]["confidence"])
        self.assertEqual(self.eval_config["bootstrap"]["seed"], zero_shot["bootstrap"]["seed"])
        self.assertEqual(self.eval_config["code_switch_thresholds"], zero_shot["code_switch_thresholds"])
        for task, zero_shot_task in zip(self.eval_config["tasks"], zero_shot["tasks"]):
            self.assertEqual(task["type"], zero_shot_task["type"])
            self.assertEqual(task["headline_metric"], zero_shot_task["headline_metric"])

    def test_dev_eval_reads_the_final_export_and_never_a_periodic_checkpoint(self):
        retriever = self.eval_config["retriever"]
        model_path = retriever["local_model_path"]
        self.assertTrue(model_path.endswith(f"/{TC.FINAL_EXPORT_DIR_PREFIX}694"))
        self.assertNotIn("checkpoint-", model_path)
        self.assertEqual(retriever["provider"], "sentence_transformers")
        self.assertFalse(retriever["trust_remote_code"])
        self.assertEqual(retriever["pooling"], self.config["models"]["kure_v1"]["pooling"])
        expected_root = Path(self.config["checkpoint"]["root_dir"]).expanduser() / "kure_v1"
        self.assertEqual(Path(model_path).expanduser().parent, expected_root)

    def test_dev_eval_records_the_accepted_zero_shot_baseline_for_delta_reporting(self):
        baseline = self.eval_config["_zero_shot_baseline"]
        self.assertEqual(baseline["exam_mcq"], {"accuracy@1": 0.2308, "MRR": 0.4948})
        self.assertEqual(
            baseline["ko_passage_strict"], {"nDCG@10": 0.8178, "MRR@10": 0.7865, "Recall@20": 0.9534}
        )


if __name__ == "__main__":
    unittest.main()
