#!/usr/bin/env python3
"""Contrastive training runner shared by all four TASK-06 model candidates.

One config (configs/train_contrastive_base.json) and one code path train
``nlpai-lab/KURE-v1``, ``BAAI/bge-m3``, ``jinaai/jina-embeddings-v3``, and the
internal-only ``upskyy/bge-m3-korean`` candidate, selected with ``--model-key``
the same way ``scripts/evaluate_retrieval.py`` selects a zero-shot candidate
with ``--model-key``. Only the model settings change between runs; the batch
source, digest preflight, objective, optimizer, precision, and checkpoint
policy stay fixed, which is what makes TASK-08's four-model comparison
meaningful.

This script never rebuilds or substitutes the TASK-04 collision-safe,
length-balanced batch plan. It reads exactly ``data/contrastive_batches_v1/``
and refuses to train unless both the plan's own recorded digest and an
independent recompute over the current on-disk bytes match the frozen
contract digest in the config - see ``load_batch_plan``. Model construction
reuses ``evaluate_retrieval.SentenceTransformerEncoder`` unmodified so the
Jina remote-code/adapter pin and the offline/local-files-only/dtype/pooling
checks can never drift between evaluation and training.

Four run modes, selected with ``--mode``:

* ``validate`` - digest preflight plus fail-closed model construction only.
* ``tokenizer_stats`` - read-only positive/negative token-length aggregate
  analysis for the selected model's tokenizer, never mutating the batch plan.
* ``smoke`` - a handful of real optimizer steps (default 5) to prove the loop
  runs end to end and loss is finite; this is what TASK-07 implementation
  review runs, never full training.
* ``train`` - the full configured schedule; this is TASK-08's job.

Both training modes end by writing a weights-only ``final-step-<N>`` export
(see ``export_final_model``). Periodic ``checkpoint-<step>`` directories are
save_steps behind the state training actually finished in, so they are resume
sources only - the final export is the artifact dev evaluation and any later
release must consume.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EVALUATOR_MODULE_PATH = PROJECT_ROOT / "scripts" / "evaluate_retrieval.py"

# Hard ceiling for --mode smoke: no --max-steps override can turn a smoke run into
# sustained training. See _enforce_mode_flag_contract().
MAX_SMOKE_STEPS = 20

CHECKPOINT_IDENTITY_SCHEMA_VERSION = 1
CHECKPOINT_IDENTITY_FILENAME = "kbmem_checkpoint_identity.json"
CHECKPOINT_COMPLETE_MARKER = "kbmem_checkpoint_complete.json"
# The final trained state is NOT one of the periodic checkpoint-<step> directories.
# With save_steps=50 and 694 optimizer steps the newest periodic checkpoint is
# checkpoint-650, which is 44 steps of training short of the model the run
# actually produced; HF Trainer performs no end-of-training save under
# save_strategy="steps". export_final_model() therefore persists the exact final
# state as a separate weights-only export directory named
# final-step-<global_step>, and that export - never checkpoint-650, never an
# in-memory-only model - is what dev evaluation consumes. The prefix is
# deliberately not "checkpoint-", so _select_resume_checkpoint() and
# _AtomicCheckpointCallback._enforce_retention() ignore it: the final export is
# never a resume source and is never rotated away by save_total_limit.
FINAL_EXPORT_DIR_PREFIX = "final-step-"
FINAL_EXPORT_KIND = "final_weights_only"
# Sentence Transformers' save() writes all of these for a Transformer+Pooling+Normalize
# model; an export missing any of them could not be reloaded offline by
# evaluate_retrieval.SentenceTransformerEncoder, so it must never be renamed into place.
REQUIRED_FINAL_EXPORT_FILES = (
    "model.safetensors", "config.json", "modules.json", "config_sentence_transformers.json",
    "sentence_bert_config.json", "tokenizer_config.json",
)
# Files a checkpoint must contain before it is considered a validly staged
# checkpoint. model.safetensors is the Sentence Transformers weight file for a
# real run; the resume-equivalence synthetic test writes real safetensors bytes
# under the same name so both paths exercise identical staging logic.
REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth")
# Resume must reject a mismatch on any of these identity fields rather than
# silently continuing under different config/data/model/runtime settings.
# TASK-08B-R1: trust_remote_code/query_adapter/passage_adapter/query_lora_task/passage_lora_task
# are included so a resume can never silently continue under a different (or missing) Jina
# task-adapter routing contract than the one the original run's checkpoints were produced under.
IDENTITY_MISMATCH_FIELDS = (
    "schema_version", "training_config_sha256", "batch_plan_digest", "model_revision",
    "code_revision", "runtime_lock_sha256", "objective", "precision", "max_seq_length_applied", "seed",
    "trust_remote_code", "query_adapter", "passage_adapter", "query_lora_task", "passage_lora_task",
)


def _load_evaluator_module():
    """Load evaluate_retrieval.py as a standalone module without touching sys.modules.

    Reusing its fail-closed construction contract (Jina remote-code/adapter
    pin, local_files_only, revision matching, dtype/pooling verification) is
    the single point that decides what a "safe model construction" means for
    this project; duplicating it here would risk the two paths silently
    drifting apart. Importing it only defines classes/functions - identical to
    how tests/test_evaluate_retrieval.py loads it - so this stays offline-safe
    even though scripts/train_contrastive.py itself needs no network access
    to import.
    """
    spec = importlib.util.spec_from_file_location(
        "train_contrastive_evaluate_retrieval", _EVALUATOR_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EV = _load_evaluator_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to configs/train_contrastive_base.json")
    parser.add_argument("--model-key", required=True, help="Key under 'models' in --config")
    parser.add_argument(
        "--mode", required=True, choices=["validate", "tokenizer_stats", "smoke", "train"],
        help="validate | tokenizer_stats | smoke | train - see module docstring",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Limit to the first N frozen batch groups")
    parser.add_argument("--max-steps", type=int, default=None, help="Override the run's step count (smoke defaults to 5)")
    parser.add_argument("--checkpoint-root", default=None, help="Override checkpoint.root_dir for this run")
    parser.add_argument(
        "--enable-production-training", action="store_true",
        help="Required for --mode train and rejected by every other mode; see _enforce_mode_flag_contract().",
    )
    return parser.parse_args()


def _enforce_mode_flag_contract(mode: str, enable_production_training: bool, max_steps: int | None) -> None:
    """Fail closed before any digest/model/GPU work if the mode/flag combination is unsafe.

    - --mode train is the only mode allowed to run a full/sustained schedule,
      and only when the caller explicitly opts in with
      --enable-production-training; a bare --mode train, or the default
      config invoked without the flag, must never silently start real
      training.
    - validate/tokenizer_stats/smoke all reject the flag outright - it is
      irrelevant to them, and accepting it silently would let a caller think
      a non-training mode had production authority it does not have.
    - smoke has a hard step ceiling (MAX_SMOKE_STEPS): --max-steps can raise
      it within that ceiling for a slightly longer sanity check, but can
      never turn smoke into sustained training.
    """
    if mode == "train" and not enable_production_training:
        raise ValueError(
            "--mode train requires --enable-production-training. This is a deliberate gate: the "
            "default config and a bare --mode train must never silently start a full/sustained "
            "training run."
        )
    if mode != "train" and enable_production_training:
        raise ValueError(
            f"--enable-production-training was passed with --mode {mode!r}, which does not accept it. "
            "Only --mode train may request production training."
        )
    if mode == "smoke" and max_steps is not None and max_steps > MAX_SMOKE_STEPS:
        raise ValueError(
            f"--mode smoke requested max_steps={max_steps}, which exceeds the hard smoke ceiling of "
            f"{MAX_SMOKE_STEPS}. --max-steps cannot turn smoke mode into sustained training; use "
            "--mode train --enable-production-training for a real run."
        )


def _recompute_batch_plan_digest(batches_path: Path, summary: dict[str, Any]) -> str:
    """Reproduce build_contrastive_batches.py's digest formula over the current on-disk bytes."""
    canonical_summary = {key: value for key, value in summary.items() if key != "deterministic_digest"}
    encoded = json.dumps(canonical_summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(batches_path.read_bytes() + encoded).hexdigest()


def _group_sizes_by_contiguous_batch_id(rows: list[dict[str, Any]]) -> list[int]:
    """Recover each TASK-04 batch's row count, requiring every batch_id to be one contiguous run.

    FrozenGroupBatchSampler assumes a batch's rows occupy a single contiguous
    index range; this is the check that assumption actually holds for the
    file currently on disk, rather than trusting the builder's own ordering.
    """
    sizes: list[int] = []
    seen_ids: set[str] = set()
    current_id: str | None = None
    for row in rows:
        batch_id = row["batch_id"]
        if batch_id != current_id:
            if batch_id in seen_ids:
                raise ValueError(
                    f"Batch id {batch_id!r} is not contiguous in the audited batch plan; refusing to "
                    "train with a sampler that assumes each TASK-04A batch occupies one contiguous run "
                    "of rows."
                )
            seen_ids.add(batch_id)
            sizes.append(0)
            current_id = batch_id
        sizes[-1] += 1
    return sizes


def load_batch_plan(
    project_root: Path, batch_plan_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """Load and digest-verify the TASK-04A audited batch plan.

    Returns rows in their frozen file order, the row count of each contiguous
    batch group in that same order, and the plan's own summary dict. Refuses
    to return anything unless both the plan's stored digest and an
    independent sha256 recompute over the current batches.jsonl.gz bytes
    match the frozen contract digest - there is no override that skips this.
    """
    plan_dir = project_root / batch_plan_config["dir"]
    batches_path = plan_dir / batch_plan_config["batches_file"]
    summary_path = plan_dir / batch_plan_config["summary_file"]
    if not batches_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Batch plan not found at {plan_dir}. scripts/train_contrastive.py only trains from the "
            "exact TASK-04A audited output; it never rebuilds or substitutes a different batch source."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_digest = batch_plan_config["expected_digest"]
    stored_digest = summary.get("deterministic_digest")
    recomputed_digest = _recompute_batch_plan_digest(batches_path, summary)
    for label, digest in (
        ("summary.json's stored digest", stored_digest),
        ("an independent recompute over the current batches.jsonl.gz bytes", recomputed_digest),
    ):
        if digest != expected_digest:
            raise ValueError(
                f"Batch-plan digest preflight failed: {label} is {digest!r}, expected the frozen "
                f"TASK-04A contract digest {expected_digest!r} (source run "
                f"{batch_plan_config.get('source_run_id')!r}). Refusing to train on an unverified or "
                "drifted batch plan."
            )
    rows = list(EV.iter_jsonl(batches_path))
    group_sizes = _group_sizes_by_contiguous_batch_id(rows)
    # A task config may additionally freeze the exact population it was written
    # against; the digest already pins the bytes, and this pins the reader's
    # interpretation of them (22,204 rows in 694 contiguous groups) so a silent
    # parsing/grouping change cannot pass unnoticed.
    for label, expected, observed in (
        ("expected_rows", batch_plan_config.get("expected_rows"), len(rows)),
        ("expected_batch_groups", batch_plan_config.get("expected_batch_groups"), len(group_sizes)),
    ):
        if expected is not None and int(expected) != observed:
            raise ValueError(
                f"Batch-plan population preflight failed: config declares {label}={int(expected)} but "
                f"the digest-verified plan yields {observed}."
            )
    return rows, group_sizes, summary


DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def estimate_training_disk_bytes(
    *,
    param_count: int,
    bytes_per_param: int,
    retained_checkpoints: int,
    optimizer_state_multiplier: float = 2.0,
    fixed_overhead_bytes: int = 1_048_576,
    staging_overhead_checkpoints: int = 1,
    run_overhead_bytes: int = 10_485_760,
    safety_margin: float = 0.2,
) -> dict[str, Any]:
    """Estimate disk bytes a training run needs, replacing a single fixed free-space threshold.

    Every component is named and returned so the run manifest records exactly
    how the number was derived, not just its final value.

    - model_weight_bytes: one copy of the model's parameters at the applied precision.
    - optimizer_state_bytes: AdamW keeps two per-parameter moment buffers
      (exp_avg, exp_avg_sq) at the same dtype as the parameter by default, so
      this is ``optimizer_state_multiplier`` (2.0) times the weight bytes.
    - gradient_bytes: gradients are never written to a saved checkpoint (they
      are transient and zeroed every step), so this is an explicit, honest
      zero rather than an omitted or fabricated line item.
    - fixed_overhead_bytes: scheduler.pt, trainer_state.json, rng_state.pth,
      and the Sentence Transformers module/tokenizer config files, which are
      small and roughly constant regardless of model size.
    - checkpoint_pool_bytes: one checkpoint's bytes times
      (retained_checkpoints + staging_overhead_checkpoints), because the
      atomic rename in atomic_save_checkpoint() briefly leaves
      save_total_limit + 1 complete checkpoints on disk before the oldest one
      is pruned.
    - final_export_bytes: one weights-only copy for TASK-08's eventual release
      artifact (no optimizer state).
    - run_overhead_bytes: flat allowance for runs/<run_id>/ logs and manifests.
    - safety_margin: multiplicative headroom applied to the whole subtotal.
    """
    if param_count <= 0:
        raise ValueError(f"param_count must be positive, got {param_count}")
    if bytes_per_param <= 0:
        raise ValueError(f"bytes_per_param must be positive, got {bytes_per_param}")
    if retained_checkpoints <= 0:
        raise ValueError(f"retained_checkpoints must be positive, got {retained_checkpoints}")
    if staging_overhead_checkpoints < 0:
        raise ValueError(f"staging_overhead_checkpoints must be non-negative, got {staging_overhead_checkpoints}")
    if safety_margin < 0:
        raise ValueError(f"safety_margin must be non-negative, got {safety_margin}")

    model_weight_bytes = param_count * bytes_per_param
    optimizer_state_bytes = int(param_count * bytes_per_param * optimizer_state_multiplier)
    gradient_bytes = 0
    one_checkpoint_bytes = model_weight_bytes + optimizer_state_bytes + gradient_bytes + fixed_overhead_bytes
    checkpoint_pool_bytes = one_checkpoint_bytes * (retained_checkpoints + staging_overhead_checkpoints)
    final_export_bytes = model_weight_bytes
    subtotal_bytes = checkpoint_pool_bytes + final_export_bytes + run_overhead_bytes
    required_bytes = int(subtotal_bytes * (1.0 + safety_margin))
    return {
        "model_weight_bytes": model_weight_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "gradient_bytes": gradient_bytes,
        "fixed_overhead_bytes": fixed_overhead_bytes,
        "one_checkpoint_bytes": one_checkpoint_bytes,
        "retained_checkpoints": retained_checkpoints,
        "staging_overhead_checkpoints": staging_overhead_checkpoints,
        "checkpoint_pool_bytes": checkpoint_pool_bytes,
        "final_export_bytes": final_export_bytes,
        "run_overhead_bytes": run_overhead_bytes,
        "subtotal_bytes": subtotal_bytes,
        "safety_margin": safety_margin,
        "required_bytes": required_bytes,
    }


def disk_preflight(path: Path, estimate: dict[str, Any]) -> dict[str, Any]:
    """Refuse to start writing checkpoints unless free space covers the full estimate.

    Replaces a single fixed free-space threshold: ``estimate`` comes from
    estimate_training_disk_bytes() and accounts for the retained-checkpoint
    pool, staging overhead, final export, run overhead, and safety margin.
    """
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    required = estimate["required_bytes"]
    if usage.free < required:
        raise RuntimeError(
            f"Disk preflight failed for {path}: {usage.free:,} bytes free, estimated requirement is "
            f"{required:,} bytes. Refusing to start training and risk a partial checkpoint. "
            f"Estimate breakdown: {estimate}"
        )
    return {"total": usage.total, "used": usage.used, "free": usage.free, "estimate": estimate}


def load_tokenizer(settings: dict[str, Any]):
    """Load a model's tokenizer under the same fail-closed contract as model construction.

    Only used for read-only tokenizer-length analysis, never for training
    itself - the trainable model's own tokenizer comes from
    build_trainable_model()/EV.SentenceTransformerEncoder instead.
    """
    from transformers import AutoTokenizer

    model_path = settings.get("local_model_path")
    if not model_path:
        raise ValueError(
            "tokenizer settings require 'local_model_path' pointing at an approved local snapshot; "
            "this script never downloads a tokenizer from the hub."
        )
    resolved_model_path = Path(str(model_path)).expanduser()
    model_id = str(settings.get("model_name_or_path", model_path))
    requested_revision = settings.get("revision")
    if resolved_model_path.exists() and requested_revision:
        applied_snapshot_revision = resolved_model_path.name
        if applied_snapshot_revision != requested_revision:
            raise ValueError(
                f"Local snapshot revision mismatch for {model_id!r}: requested {requested_revision!r} "
                f"but local_model_path resolves to snapshot {applied_snapshot_revision!r}."
            )
    remote_code = EV._validate_remote_code_controls(
        model_id=model_id,
        requested_trust_remote_code=bool(settings.get("trust_remote_code", False)),
        requested_model_revision=requested_revision,
        requested_code_revision=settings.get("code_revision"),
        requested_query_adapter=settings.get("query_adapter"),
        requested_passage_adapter=settings.get("passage_adapter"),
    )
    return AutoTokenizer.from_pretrained(
        str(resolved_model_path),
        revision=requested_revision,
        trust_remote_code=remote_code["trust_remote_code"],
        local_files_only=True,
    )


def _percentile(ordered: list[int], fraction: float) -> int:
    if not ordered:
        return 0
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def tokenizer_length_stats(
    tokenizer,
    rows: list[dict[str, Any]],
    fields: list[str],
    sample_size: int | None,
    seed: int,
    max_seq_length: int | None = None,
) -> dict[str, Any]:
    """Measure per-field token-length distributions (count/mean/p50/p95/p99/max) and, when
    max_seq_length is given, how many rows would exceed it - read-only, over query/positive/negative.

    Character-length balance (TASK-04A) does not imply tokenizer-length
    balance, because each model segments Korean/Latin text differently. This
    reads the already digest-verified rows in memory only; results are
    reported by the caller and never written back into
    data/contrastive_batches_v1. sample_size=None (the config default) uses
    the full population; a smaller sample is deterministic for a fixed seed.
    """
    import random

    sample = rows
    if sample_size is not None and sample_size < len(rows):
        sample = random.Random(seed).sample(rows, sample_size)

    lengths: dict[str, list[int]] = {field: [] for field in fields}
    for row in sample:
        for field in fields:
            lengths[field].append(len(tokenizer(row[field], add_special_tokens=True)["input_ids"]))

    def summarize(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        summary: dict[str, Any] = {
            "count": len(values),
            "mean": round(statistics.fmean(values), 4),
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
            "max": ordered[-1],
        }
        if max_seq_length is not None:
            exceeding = sum(1 for value in values if value > max_seq_length)
            summary["requested_max_seq_length"] = max_seq_length
            summary["applied_max_seq_length"] = max_seq_length
            summary["count_exceeding_max"] = exceeding
            summary["percentage_exceeding_max"] = round(100.0 * exceeding / len(values), 4)
        return summary

    stats: dict[str, Any] = {field: summarize(values) for field, values in lengths.items()}
    if "positive" in stats and "negative" in stats and stats["positive"]["count"] and stats["negative"]["count"]:
        stats["gap"] = {
            "mean_gap": round(stats["positive"]["mean"] - stats["negative"]["mean"], 4),
            "p50_gap": stats["positive"]["p50"] - stats["negative"]["p50"],
            "p95_gap": stats["positive"]["p95"] - stats["negative"]["p95"],
            "p99_gap": stats["positive"]["p99"] - stats["negative"]["p99"],
        }
    stats["sample_size"] = len(sample)
    stats["population_size"] = len(rows)
    return stats


def build_trainable_model(settings: dict[str, Any]):
    """Construct a real, gradient-enabled SentenceTransformer for training.

    Reuses evaluate_retrieval.SentenceTransformerEncoder unmodified for the
    fail-closed revision/remote-code/dtype/pooling/adapter construction
    contract, then continues with its already-verified ``._model`` in train
    mode. The inference-only wrapper (its batched encode() method) is
    discarded; only the underlying torch module continues into training.
    """
    encoder = EV.SentenceTransformerEncoder(settings)
    model = encoder._model
    requested_max = int(settings.get("max_seq_length", 512))
    applied_max = min(requested_max, encoder._native_max_seq_length)
    if applied_max != requested_max:
        raise ValueError(
            f"{encoder.model_id!r} requested max_seq_length={requested_max} but the loaded model's "
            f"native cap is {encoder._native_max_seq_length}; applying {applied_max} would silently "
            f"truncate below the requested length. Set max_seq_length to {applied_max} explicitly in "
            "the config if that is the intended training length."
        )
    model.max_seq_length = applied_max
    model.train()
    metadata = {
        "model_id": encoder.model_id,
        "revision_applied": encoder.model_revision_applied,
        "trust_remote_code_applied": encoder.trust_remote_code_applied,
        "code_revision_applied": encoder.code_revision_applied,
        "dtype_applied": encoder.dtype_applied,
        "pooling_applied": encoder.pooling_applied,
        "device_applied": encoder.device_applied,
        "query_adapter_applied": encoder.query_adapter_applied,
        "passage_adapter_applied": encoder.passage_adapter_applied,
        "query_lora_task_applied": encoder.query_lora_task_applied,
        "passage_lora_task_applied": encoder.passage_lora_task_applied,
        "query_prompt_text_applied": encoder.query_prompt_text_applied,
        "passage_prompt_text_applied": encoder.passage_prompt_text_applied,
        "lora_task_routing_behaviorally_verified": encoder.lora_task_routing_behaviorally_verified,
        "max_seq_length_requested": requested_max,
        "max_seq_length_native": encoder._native_max_seq_length,
        "max_seq_length_applied": applied_max,
        "architecture_max_position_embeddings": encoder.architecture_max_position_embeddings,
        "sentence_transformers_version": encoder.library_version,
        "license_label": settings.get("license_label"),
    }
    return model, metadata


def resolve_training_prompts(model, metadata: dict[str, Any]) -> dict[str, str] | None:
    """Map the frozen dataset columns to Jina's query/passage adapters, or None for plain models.

    query_adapter_applied/passage_adapter_applied are already verified against
    the loaded model's real .prompts mapping by
    evaluate_retrieval.SentenceTransformerEncoder before this runs, mirroring
    the same query-vs-passage role split evaluate_mcq/DenseRetriever use at
    inference time.
    """
    query_adapter = metadata.get("query_adapter_applied")
    passage_adapter = metadata.get("passage_adapter_applied")
    if query_adapter is None and passage_adapter is None:
        return None
    prompts = model.prompts
    return {"query": prompts[query_adapter], "positive": prompts[passage_adapter], "negative": prompts[passage_adapter]}


def make_task_aware_mnrl_loss(base_loss, *, query_task: str, passage_task: str):
    """Wrap an installed MultipleNegativesRankingLoss so every sentence_features entry is routed
    through Jina's real forward(task=...) LoRA adapter, without patching sentence_transformers.

    TASK-08B-R1: the installed loss's own ``forward()`` calls ``self.model(sentence_feature)``
    with no ``task`` kwarg, which - for Jina - leaves every column routed through the base
    backbone with no LoRA delta applied at all (the same query/passage routing mismatch
    behaviorally verified and closed for inference in
    ``evaluate_retrieval.SentenceTransformerEncoder``). This wrapper is the training-side fix: it
    reuses the installed loss's ``compute_loss_from_embeddings()`` unmodified, so the scoring math
    (in-batch negatives, temperature scale, directions, partition mode) is byte-for-byte the same
    objective TASK-08A trained under - only how each embedding is produced changes. The frozen
    batch plan's dataset column order (``build_training_dataset``: query, positive, negative) means
    ``sentence_features[0]`` is always the anchor/query and every later entry is a
    positive/hard-negative document, so position 0 routes through ``query_task`` and every
    subsequent position routes through ``passage_task``.
    """
    from torch import nn

    class _TaskAwareMultipleNegativesRankingLoss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_loss = base_loss
            self.model = base_loss.model
            self.query_task = query_task
            self.passage_task = passage_task

        def forward(self, sentence_features, labels):
            embeddings = []
            for position, sentence_feature in enumerate(sentence_features):
                task = self.query_task if position == 0 else self.passage_task
                embeddings.append(self.model(sentence_feature, task=task)["sentence_embedding"])
            return self.base_loss.compute_loss_from_embeddings(embeddings, labels)

        def get_config_dict(self) -> dict[str, Any]:
            base_config = self.base_loss.get_config_dict() if hasattr(self.base_loss, "get_config_dict") else {}
            return {**base_config, "query_lora_task": self.query_task, "passage_lora_task": self.passage_task}

    return _TaskAwareMultipleNegativesRankingLoss()


def build_training_dataset(rows: list[dict[str, Any]]):
    """Wrap the frozen-order rows in a datasets.Dataset with columns [query, positive, negative]."""
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError(
            "training requires the optional 'datasets' package, which is not importable in this "
            "environment. This script does not install packages on its own."
        ) from error
    return Dataset.from_dict({
        "query": [row["query"] for row in rows],
        "positive": [row["positive"] for row in rows],
        "negative": [row["negative"] for row in rows],
    })


class FrozenGroupBatchSampler:
    """Replays the TASK-04A batch groups exactly; only their presentation order is shuffled.

    Sentence Transformers' built-in batch samplers draw a fresh random subset
    of dataset rows for every step, which would destroy the collision-safe,
    length-balanced composition TASK-04A audited (105,749 -> 0 collisions,
    every batch's positive/negative length gap bounded by config). Each
    contiguous batch_id run in the frozen plan is treated as one indivisible
    unit here: every training step is exactly one of the audited batches,
    never a reshuffled mix of rows drawn from different batches.
    """

    def __init__(
        self,
        dataset,
        batch_size=None,
        drop_last=None,
        valid_label_columns=None,
        generator=None,
        seed=0,
        group_sizes: list[int] | None = None,
    ) -> None:
        del batch_size, drop_last, valid_label_columns, seed
        if not group_sizes:
            raise ValueError("FrozenGroupBatchSampler requires the audited batch plan's group_sizes.")
        if sum(group_sizes) != len(dataset):
            raise ValueError(
                f"group_sizes sum to {sum(group_sizes)} but the dataset has {len(dataset)} rows; the "
                "frozen batch plan and the training dataset have drifted apart."
            )
        self.generator = generator
        boundaries: list[tuple[int, int]] = []
        start = 0
        for size in group_sizes:
            boundaries.append((start, start + size))
            start += size
        self._boundaries = boundaries

    def __len__(self) -> int:
        return len(self._boundaries)

    def __iter__(self):
        order = list(range(len(self._boundaries)))
        if self.generator is not None:
            import torch

            order = torch.randperm(len(order), generator=self.generator).tolist()
        for group_index in order:
            start, end = self._boundaries[group_index]
            yield list(range(start, end))


class _FrozenBatchSamplerFactory:
    """Picklable stand-in for a closure, since HF Trainer checkpoints torch.save() self.args.

    A local `def factory(dataset, **kwargs): ...` closure cannot be pickled, and
    Trainer._save() persists the full TrainingArguments - including
    args.batch_sampler - into every checkpoint. Storing group_sizes as plain
    instance state on a module-level class keeps checkpointing working.
    """

    def __init__(self, group_sizes: list[int]) -> None:
        self.group_sizes = group_sizes

    def __call__(self, dataset, **kwargs):
        return FrozenGroupBatchSampler(dataset, group_sizes=self.group_sizes, **kwargs)


def make_frozen_batch_sampler_factory(group_sizes: list[int]) -> _FrozenBatchSamplerFactory:
    return _FrozenBatchSamplerFactory(group_sizes)


def resolve_checkpoint_root(checkpoint_cfg: dict[str, Any], model_key: str) -> Path:
    """Stable per model_key, not per run_id.

    Each CLI invocation gets a fresh runs/<run_id>/ for its own log/manifest,
    but a rerun after an interruption must land on the same checkpoint_root as
    the interrupted attempt to find and resume from its newest complete
    checkpoint - see _latest_complete_checkpoint(). Folding run_id into this
    path would make every restart start over from step 0.
    """
    return Path(checkpoint_cfg["root_dir"]).expanduser() / model_key


def _config_digest(config: dict[str, Any]) -> str:
    """Canonical sha256 over the effective training config, matching _recompute_batch_plan_digest's style."""
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_lock_fingerprint(python_executable: str | None = None) -> dict[str, Any]:
    """Sha256 over this interpreter's exact installed-distribution set (pip freeze, minus pip itself).

    Computed live from the running venv so a checkpoint's identity manifest
    always reflects the runtime it was actually produced under, rather than
    trusting a separately-maintained lock file that could drift out from
    under it. ``pip`` itself is excluded as the documented venv-bootstrap
    exception, matching the TASK-07R1 runtime-freeze convention.
    """
    executable = python_executable or sys.executable
    result = subprocess.run(
        [executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True, check=True,
    )
    lines = sorted(line for line in result.stdout.splitlines() if line and not line.startswith("pip=="))
    text = "\n".join(lines) + "\n"
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "package_count": len(lines)}


def resolve_checkpoint_identity(
    *, config: dict[str, Any], config_digest: str, batch_plan_digest: str, metadata: dict[str, Any],
    runtime_lock_sha256: str,
) -> dict[str, Any]:
    """The subset of checkpoint identity fields resume compares for mismatch, independent of run state.

    Shared by the real training path (to compare against a candidate
    checkpoint) and by tests (to build synthetic identity dicts) so both use
    exactly the same field set as IDENTITY_MISMATCH_FIELDS.
    """
    return {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA_VERSION,
        "training_config_sha256": config_digest,
        "batch_plan_digest": batch_plan_digest,
        "model_revision": metadata["revision_applied"],
        "code_revision": metadata["code_revision_applied"],
        "runtime_lock_sha256": runtime_lock_sha256,
        "objective": config["objective"]["loss"],
        "precision": config["runtime"]["precision"],
        "max_seq_length_applied": metadata["max_seq_length_applied"],
        "seed": int(config["schedule"]["seed"]),
        # TASK-08B-R1: prompt-name selection (query_adapter/passage_adapter) and behaviorally
        # verified LoRA task routing (query_lora_task/passage_lora_task) are two distinct claims -
        # see evaluate_retrieval.SentenceTransformerEncoder._verify_lora_task_routing - and both are
        # attested here so a checkpoint's identity manifest and any dev result linked to it can never
        # be mistaken for having applied a routing contract it did not actually run under.
        "trust_remote_code": metadata.get("trust_remote_code_applied", False),
        "query_adapter": metadata.get("query_adapter_applied"),
        "passage_adapter": metadata.get("passage_adapter_applied"),
        "query_lora_task": metadata.get("query_lora_task_applied"),
        "passage_lora_task": metadata.get("passage_lora_task_applied"),
    }


def _fsync_directory_tree(path: Path) -> None:
    """Fsync every file under path, then the directory itself, before it is trusted as durable."""
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_descriptor = os.open(str(Path(root) / name), os.O_RDONLY)
            try:
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
    dir_descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(dir_descriptor)
    finally:
        os.close(dir_descriptor)


def atomic_save_checkpoint(*, staged_dir: Path, final_dir: Path, identity: dict[str, Any]) -> Path:
    """Validate a freshly-written staged checkpoint, attach its identity manifest, fsync, then
    atomically rename it into its final checkpoint-<step> location.

    Shared by the real HF Trainer checkpoint callback and by the synthetic
    resume-equivalence test, so both exercise exactly this logic: neither
    ``final_dir`` nor anything resumable ever exists until every required
    state file, the identity manifest, and the completion marker have all
    been written and fsynced. A crash at any point before the final
    os.rename() leaves ``staged_dir`` behind - never a partially-written
    ``final_dir`` - and _select_resume_checkpoint() only ever looks at
    ``final_dir``-shaped paths with a valid completion marker.
    """
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (staged_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Staged checkpoint {staged_dir} is missing required file(s): {missing}.")

    identity_text = json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (staged_dir / CHECKPOINT_IDENTITY_FILENAME).write_text(identity_text, encoding="utf-8")

    _fsync_directory_tree(staged_dir)

    marker = {
        "complete": True,
        "identity_sha256": hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
        "written_at": dt.datetime.now().astimezone().isoformat(),
    }
    marker_path = staged_dir / CHECKPOINT_COMPLETE_MARKER
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with marker_path.open("rb") as handle:
        os.fsync(handle.fileno())

    if final_dir.exists():
        raise RuntimeError(f"Refusing to overwrite an already-completed checkpoint at {final_dir}.")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staged_dir, final_dir)
    return final_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_export_final_model(
    *, save_model, staged_dir: Path, final_dir: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    """Write a weights-only export through staging, validate it, fsync it, then atomically rename it.

    Deliberately mirrors atomic_save_checkpoint()'s contract - stage, require
    every file that makes the artifact usable, attach an identity manifest,
    fsync, write the completion marker, refuse to overwrite, rename - so the
    final trained state gets exactly the same durability guarantees the
    periodic checkpoints already had. ``save_model`` is a callable that writes
    the model into a directory (SentenceTransformer.save for a real run, a
    synthetic writer in tests), which keeps this function testable without
    sentence_transformers, torch, or a GPU.

    A crash at any point before the final os.rename() leaves only
    ``staged_dir`` behind; ``final_dir`` never exists in a partial state, and
    it only ever appears complete because the marker is the last file written
    before the rename.
    """
    if final_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite an existing final export at {final_dir}. A completed export is "
            "immutable evidence of one specific trained state; remove or archive it deliberately "
            "rather than letting a rerun silently replace it."
        )
    if staged_dir.exists():
        raise RuntimeError(
            f"Refusing to reuse leftover export staging at {staged_dir}; it is the residue of an "
            "interrupted export and its contents cannot be trusted. Inspect and remove it "
            "deliberately."
        )
    staged_dir.mkdir(parents=True)
    save_model(staged_dir)

    missing = [name for name in REQUIRED_FINAL_EXPORT_FILES if not (staged_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Staged final export {staged_dir} is missing required file(s): {missing}. Refusing to "
            "publish an export that could not be reloaded offline for dev evaluation."
        )

    identity_text = json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (staged_dir / CHECKPOINT_IDENTITY_FILENAME).write_text(identity_text, encoding="utf-8")

    _fsync_directory_tree(staged_dir)

    marker = {
        "complete": True,
        "export_kind": FINAL_EXPORT_KIND,
        "global_step": identity.get("global_step"),
        "identity_sha256": hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
        "written_at": dt.datetime.now().astimezone().isoformat(),
    }
    marker_path = staged_dir / CHECKPOINT_COMPLETE_MARKER
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with marker_path.open("rb") as handle:
        os.fsync(handle.fileno())

    if final_dir.exists():
        raise RuntimeError(f"Refusing to overwrite an existing final export at {final_dir}.")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staged_dir, final_dir)

    total_bytes = sum(path.stat().st_size for path in final_dir.rglob("*") if path.is_file())
    weights_path = final_dir / "model.safetensors"
    return {
        "path": str(final_dir),
        "global_step": identity.get("global_step"),
        "export_kind": FINAL_EXPORT_KIND,
        "identity": identity,
        "identity_sha256": marker["identity_sha256"],
        "completion_marker": CHECKPOINT_COMPLETE_MARKER,
        "total_bytes": total_bytes,
        "model_safetensors_sha256": _sha256_file(weights_path),
        "files": sorted(str(path.relative_to(final_dir)) for path in final_dir.rglob("*") if path.is_file()),
    }


def _read_base_snapshot_auto_map(model_settings: dict[str, Any]) -> dict[str, Any] | None:
    """Read the base snapshot's own config.json ``auto_map`` (None if absent, e.g. KURE/BAAI/Upskyy).

    This is the pre-training, pre-save ``auto_map`` - the exact repo-qualified form the pinned
    Jina snapshot already loads successfully with - captured before ``model.save()`` has a chance
    to rewrite it into the bare local form that trips the transformers local-directory bug. See
    ``_save_model_and_restore_auto_map``.
    """
    local_model_path = model_settings.get("local_model_path")
    if not local_model_path:
        return None
    config_path = Path(str(local_model_path)).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    auto_map = config.get("auto_map")
    return dict(auto_map) if isinstance(auto_map, dict) else None


def _save_model_and_restore_auto_map(model, directory: Path, source_auto_map: dict[str, Any] | None) -> None:
    """Save the model, then restore config.json's pre-save ``auto_map`` if the save rewrote it.

    TASK-08B-R2 discovery: for a trust_remote_code model, ``SentenceTransformer.save()`` rewrites
    the classes it actually loaded (``AutoConfig``/``AutoModel`` here) from their original
    ``<repo_id>--<module>.<Class>`` form to a bare local ``<module>.<Class>`` form, to make the
    export self-contained. That bare form then makes
    ``transformers.dynamic_module_utils.get_cached_module_file`` treat the export as a plain local
    directory (``is_local=True``), whose copy step only copies the entry file's *direct* relative
    imports, not the full transitive closure Jina's modeling_xlm_roberta.py -> mha/block/mlp/
    rotary/stochastic_depth/xlm_padding/embedding chain needs - an upstream transformers bug this
    project cannot patch. Restoring the original repo-qualified auto_map (the exact string the
    pinned, statically-reviewed jinaai/xlm-roberta-flash-implementation snapshot already uses
    successfully) routes the reload through the correctly-recursing non-local branch instead,
    resolving from that same already-cached, offline, pinned code_revision. A no-op for any model
    (KURE, BAAI, Upskyy) whose config.json has no auto_map at all.
    """
    model.save(str(directory), create_model_card=False, safe_serialization=True)
    if not source_auto_map:
        return
    config_path = directory / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "auto_map" not in config:
        return
    config["auto_map"] = source_auto_map
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _restore_periodic_checkpoint_auto_map(staged_dir: Path, source_auto_map: dict[str, Any] | None) -> bool:
    """Apply the same TASK-08B-R2 auto_map restoration to a periodic (save_steps) checkpoint,
    fixing TASK-08G-R2-R1's discovered resume-time gap: only the final export previously received
    this treatment, so resuming from any periodic checkpoint hit the same shallow-recursion bug
    _save_model_and_restore_auto_map's docstring describes.

    Verified no-op (returns False, touches nothing) when ``source_auto_map`` is falsy - i.e. every
    model whose base snapshot config.json has no auto_map at all (KURE, BAAI, Upskyy). For a model
    that does have one (Jina), this fails closed - raises RuntimeError - rather than silently
    skipping, if the checkpoint Trainer just staged does not have the config.json/auto_map shape
    this project's own save path is known to produce: missing config.json, unparseable JSON, no
    ``auto_map`` key, or an auto_map whose key set does not match the pinned source (any of these
    signal something unexpected happened during the library's own save, which this restoration
    step must not paper over by guessing or partially overwriting).
    """
    if not source_auto_map:
        return False
    config_path = staged_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Refusing periodic-checkpoint auto_map restoration: expected {config_path} to exist "
            "for a model with a pinned source auto_map, but it is missing."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Refusing periodic-checkpoint auto_map restoration: {config_path} is not valid JSON."
        ) from error
    staged_auto_map = config.get("auto_map")
    if not isinstance(staged_auto_map, dict) or not staged_auto_map:
        raise RuntimeError(
            f"Refusing periodic-checkpoint auto_map restoration: {config_path} has no non-empty "
            "'auto_map' object, but this model's pinned source config.json has one."
        )
    if set(staged_auto_map.keys()) != set(source_auto_map.keys()):
        raise RuntimeError(
            f"Refusing periodic-checkpoint auto_map restoration: {config_path}'s auto_map keys "
            f"{sorted(staged_auto_map.keys())} do not match the pinned source auto_map keys "
            f"{sorted(source_auto_map.keys())} - refusing to restore over drifted content."
        )
    config["auto_map"] = dict(source_auto_map)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def export_final_model(
    model, *, checkpoint_root: Path, identity_context: dict[str, Any], global_step: int,
    expected_global_step: int | None = None, source_auto_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the exact final trained state, refusing to publish a step count that was not reached.

    ``expected_global_step`` is the run's planned optimizer-step total. If
    training stopped early - an interruption, an early stop, a drifted
    schedule - the in-memory model is not the state this run was authorized to
    produce, so nothing is exported and every existing checkpoint is preserved
    for inspection rather than being papered over with a mislabeled "final"
    artifact. ``source_auto_map`` is the base snapshot's own config.json auto_map (if any), applied
    after save() - see ``_save_model_and_restore_auto_map``.
    """
    if expected_global_step is not None and int(global_step) != int(expected_global_step):
        raise RuntimeError(
            f"Refusing to export a final model at global_step={global_step} when the run's planned "
            f"schedule was {expected_global_step} optimizer steps. The trained state is incomplete; "
            "existing checkpoints are left untouched for inspection."
        )
    identity = {
        **identity_context,
        "global_step": int(global_step),
        "export_kind": FINAL_EXPORT_KIND,
        "created_at": dt.datetime.now().astimezone().isoformat(),
    }
    final_dir = checkpoint_root / f"{FINAL_EXPORT_DIR_PREFIX}{int(global_step)}"
    staged_dir = checkpoint_root / ".staging" / f"{FINAL_EXPORT_DIR_PREFIX}{int(global_step)}"
    return atomic_export_final_model(
        save_model=lambda directory: _save_model_and_restore_auto_map(model, directory, source_auto_map),
        staged_dir=staged_dir, final_dir=final_dir, identity=identity,
    )


def _resolve_planned_optimizer_steps(
    schedule: dict[str, Any], group_sizes: list[int], max_steps: int | None = None
) -> int:
    """How many optimizer steps this run is supposed to take, given the frozen batch groups.

    Every training step consumes exactly one audited TASK-04A batch group (see
    FrozenGroupBatchSampler), so one epoch over 694 groups is 694 steps. A
    positive --max-steps/config max_steps overrides that, which is what smoke
    runs use.
    """
    override = max_steps if max_steps is not None else schedule.get("max_steps", -1)
    if override is not None and int(override) > 0:
        return int(override)
    return int(len(group_sizes) * float(schedule["num_train_epochs"]))


def resolve_checkpoint_root(checkpoint_cfg: dict[str, Any], model_key: str) -> Path:
    """Stable per model_key, not per run_id.

    Each CLI invocation gets a fresh runs/<run_id>/ for its own log/manifest,
    but a rerun after an interruption must land on the same checkpoint_root as
    the interrupted attempt to find and resume from its newest complete
    checkpoint - see _select_resume_checkpoint(). Folding run_id into this
    path would make every restart start over from step 0.
    """
    return Path(checkpoint_cfg["root_dir"]).expanduser() / model_key


def _select_resume_checkpoint(checkpoint_root: Path, expected_identity: dict[str, Any]) -> str | None:
    """Find the newest checkpoint with a valid completion marker, and reject it if its identity
    disagrees with the current run's config/data/model/runtime on any IDENTITY_MISMATCH_FIELDS entry.

    Returns None only when there is no valid checkpoint at all (a fresh
    start). A checkpoint that exists but was staged and never completed - no
    CHECKPOINT_COMPLETE_MARKER - is silently skipped, matching the atomicity
    contract: it was never resumable in the first place. A checkpoint that
    completed but does not match the current run raises instead of being
    silently ignored, because silently falling back to "start fresh" would
    hide exactly the kind of misconfiguration (wrong model revision, wrong
    batch digest, wrong seed) this check exists to catch.
    """
    if not checkpoint_root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        if not (path.is_dir() and path.name.startswith("checkpoint-")):
            continue
        if not (path / CHECKPOINT_COMPLETE_MARKER).is_file() or not (path / CHECKPOINT_IDENTITY_FILENAME).is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    newest_step, newest_path = candidates[-1]
    identity = json.loads((newest_path / CHECKPOINT_IDENTITY_FILENAME).read_text(encoding="utf-8"))
    mismatches = [
        (field, identity.get(field), expected_identity.get(field))
        for field in IDENTITY_MISMATCH_FIELDS
        if identity.get(field) != expected_identity.get(field)
    ]
    if mismatches:
        detail = "; ".join(f"{field}: checkpoint={got!r} current={want!r}" for field, got, want in mismatches)
        raise ValueError(
            f"Refusing to resume from {newest_path} (step {newest_step}): identity mismatch on "
            f"{len(mismatches)} field(s) - {detail}. Resume never silently continues a run under "
            "different config/data/model/runtime settings; use a different checkpoint_root or "
            "resolve the mismatch."
        )
    return str(newest_path)


class _AtomicCheckpointCallback:
    """Stages every HF Trainer checkpoint under checkpoint_root/.staging, then hands it to
    atomic_save_checkpoint() to gain an identity manifest and land atomically in checkpoint_root.

    training_args.output_dir is pointed at checkpoint_root/.staging, so
    Trainer._save_checkpoint() - the well-tested library code that writes
    weights/optimizer/scheduler/rng/trainer_state - writes there under its
    own naming exactly as it always does; nothing about that logic is
    reimplemented. Trainer's own internal _rotate_checkpoints() runs before
    this callback fires and only ever sees the single checkpoint currently in
    staging (this callback empties staging again immediately after every
    save), so save_total_limit is enforced here instead, against the real
    checkpoint_root.
    """

    def __init__(
        self, *, checkpoint_root: Path, staging_root: Path, save_total_limit: int, identity_context: dict[str, Any],
        source_auto_map: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_root = checkpoint_root
        self.staging_root = staging_root
        self.save_total_limit = save_total_limit
        self.identity_context = identity_context
        self.source_auto_map = source_auto_map

    def on_save(self, args, state, control, **kwargs):
        del args, kwargs
        checkpoint_folder = f"checkpoint-{state.global_step}"
        staged_dir = self.staging_root / checkpoint_folder
        final_dir = self.checkpoint_root / checkpoint_folder
        if not staged_dir.is_dir():
            raise RuntimeError(f"Expected a staged checkpoint at {staged_dir}, but it does not exist.")

        # TASK-08G-R2-R2: restore the pinned repo-qualified auto_map in the staged
        # checkpoint before it is atomically finalized, so a later resume from this
        # exact checkpoint reconstructs the model through the same fully-recursing
        # code path the original snapshot and the final export already use.
        _restore_periodic_checkpoint_auto_map(staged_dir, self.source_auto_map)

        identity = {
            **self.identity_context,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "batch_position": state.global_step,
            "created_at": dt.datetime.now().astimezone().isoformat(),
        }
        atomic_save_checkpoint(staged_dir=staged_dir, final_dir=final_dir, identity=identity)
        self._enforce_retention()
        return control

    def _enforce_retention(self) -> None:
        if self.save_total_limit <= 0:
            return
        candidates: list[tuple[int, Path]] = []
        for path in self.checkpoint_root.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-") and (path / CHECKPOINT_COMPLETE_MARKER).is_file():
                try:
                    step = int(path.name.rsplit("-", 1)[-1])
                except ValueError:
                    continue
                candidates.append((step, path))
        candidates.sort(key=lambda item: item[0])
        for _step, path in candidates[: max(0, len(candidates) - self.save_total_limit)]:
            shutil.rmtree(path)


def run_training(
    model,
    metadata: dict[str, Any],
    dataset,
    group_sizes: list[int],
    config: dict[str, Any],
    checkpoint_root: Path,
    model_key: str,
    git_commit: str,
    max_steps: int | None = None,
    model_reconstruction_context: contextlib.AbstractContextManager | None = None,
):
    """``model_reconstruction_context``, if given, is entered only around
    ``trainer.train(resume_from_checkpoint=...)`` - the one call that can
    internally reconstruct the model a second time (sentence_transformers'
    Trainer._load_from_checkpoint, for a resumed run). Every other model here
    (KURE, BAAI, Upskyy) passes None and this function's behavior for them is
    byte-for-byte unchanged: ``contextlib.nullcontext()`` wraps the same call
    with no effect. Only a caller building the pinned Jina model supplies a
    real context (TASK-08G-R2-R1's tied-weights compatibility shim), keeping
    this runner itself model-agnostic.
    """
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from transformers import TrainerCallback

    schedule = config["schedule"]
    optimizer_cfg = config["optimizer"]
    runtime_cfg = config["runtime"]
    checkpoint_cfg = config["checkpoint"]
    model_settings = config["models"][model_key]

    param_count = sum(p.numel() for p in model.parameters())
    disk_estimate = estimate_training_disk_bytes(
        param_count=param_count,
        bytes_per_param=DTYPE_BYTES[metadata["dtype_applied"]],
        retained_checkpoints=int(checkpoint_cfg["save_total_limit"]),
        **checkpoint_cfg.get("disk_estimate", {}),
    )
    disk_before = disk_preflight(checkpoint_root, disk_estimate)

    config_digest = _config_digest(config)
    runtime_lock = runtime_lock_fingerprint()
    identity_context = {
        **resolve_checkpoint_identity(
            config=config, config_digest=config_digest, batch_plan_digest=config["batch_plan"]["expected_digest"],
            metadata=metadata, runtime_lock_sha256=runtime_lock["sha256"],
        ),
        "model_key": model_key,
        "license_label": metadata["license_label"],
        "git_commit": git_commit,
        "runtime_lock_package_count": runtime_lock["package_count"],
    }
    expected_identity = {field: identity_context[field] for field in IDENTITY_MISMATCH_FIELDS}
    resume_from = _select_resume_checkpoint(checkpoint_root, expected_identity)

    # Fail before the first optimizer step if the schedule would not produce the
    # exact step count the task config froze, rather than discovering it 694
    # steps later at export time.
    planned_steps = _resolve_planned_optimizer_steps(schedule, group_sizes, max_steps)
    declared_steps = schedule.get("expected_optimizer_steps")
    if declared_steps is not None and int(declared_steps) != planned_steps:
        raise ValueError(
            f"Frozen schedule contract violated: config declares expected_optimizer_steps="
            f"{int(declared_steps)} but this run would take {planned_steps} optimizer steps over "
            f"{len(group_sizes)} frozen batch groups and {schedule['num_train_epochs']} epoch(s)."
        )

    # Captured once, before training starts, so both the periodic-checkpoint
    # callback (every save_steps) and the final export apply the identical
    # pinned value - never re-read mid-run, never allowed to drift between them.
    source_auto_map = _read_base_snapshot_auto_map(model_settings)

    staging_root = checkpoint_root / ".staging"
    atomic_checkpoint = _AtomicCheckpointCallback(
        checkpoint_root=checkpoint_root, staging_root=staging_root,
        save_total_limit=int(checkpoint_cfg["save_total_limit"]), identity_context=identity_context,
        source_auto_map=source_auto_map,
    )

    class _TrainerCallbackAdapter(TrainerCallback):
        """Thin adapter so _AtomicCheckpointCallback stays importable/testable without transformers.

        Only overrides on_save; every other on_* event falls through to
        TrainerCallback's default no-op implementations.
        """

        def on_save(self, args, state, control, **kwargs):
            return atomic_checkpoint.on_save(args, state, control, **kwargs)

    callback = _TrainerCallbackAdapter()

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(staging_root),
        num_train_epochs=float(schedule["num_train_epochs"]),
        max_steps=int(max_steps) if max_steps is not None else int(schedule.get("max_steps", -1)),
        per_device_train_batch_size=int(model_settings.get("per_device_train_batch_size", 32)),
        learning_rate=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        warmup_ratio=float(optimizer_cfg["warmup_ratio"]),
        lr_scheduler_type=str(optimizer_cfg["lr_scheduler_type"]),
        max_grad_norm=float(optimizer_cfg["max_grad_norm"]),
        bf16=runtime_cfg["precision"] == "bf16",
        fp16=runtime_cfg["precision"] == "float16",
        gradient_checkpointing=bool(runtime_cfg["gradient_checkpointing"]),
        dataloader_drop_last=bool(runtime_cfg.get("dataloader_drop_last", False)),
        dataloader_num_workers=int(runtime_cfg.get("dataloader_num_workers", 0)),
        save_strategy="steps",
        save_steps=int(checkpoint_cfg["save_steps"]),
        save_total_limit=int(checkpoint_cfg["save_total_limit"]),
        seed=int(schedule["seed"]),
        data_seed=int(schedule["seed"]),
        batch_sampler=make_frozen_batch_sampler_factory(group_sizes),
        prompts=resolve_training_prompts(model, metadata),
        report_to=[],
        logging_steps=1,
        disable_tqdm=True,
    )
    loss = MultipleNegativesRankingLoss(model, scale=float(config["objective"].get("scale", 20.0)))
    query_lora_task = metadata.get("query_lora_task_applied")
    passage_lora_task = metadata.get("passage_lora_task_applied")
    if query_lora_task is not None or passage_lora_task is not None:
        if query_lora_task is None or passage_lora_task is None:
            raise ValueError(
                "Jina LoRA task routing requires both query_lora_task_applied and "
                f"passage_lora_task_applied; got query={query_lora_task!r} passage={passage_lora_task!r}."
            )
        if not metadata.get("lora_task_routing_behaviorally_verified"):
            raise ValueError(
                "Refusing to build a task-aware training loss: build_trainable_model() did not "
                "behaviorally verify LoRA task routing for this model (see "
                "evaluate_retrieval.SentenceTransformerEncoder._verify_lora_task_routing). Training "
                "with an unverified task adapter is exactly the query/passage routing mismatch "
                "TASK-08B-R1 exists to catch."
            )
        loss = make_task_aware_mnrl_loss(loss, query_task=query_lora_task, passage_task=passage_lora_task)
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=dataset, loss=loss, callbacks=[callback],
    )
    with (model_reconstruction_context or contextlib.nullcontext()):
        trainer.train(resume_from_checkpoint=resume_from)

    # The newest periodic checkpoint is save_steps behind the state that just
    # finished training (checkpoint-650 for a 694-step run), and it only exists
    # in memory until this call. Everything downstream - dev evaluation above
    # all - must consume this export, not a checkpoint and not the live process.
    final_export = export_final_model(
        trainer.model, checkpoint_root=checkpoint_root, identity_context=identity_context,
        global_step=int(trainer.state.global_step), expected_global_step=planned_steps,
        source_auto_map=source_auto_map,
    )

    disk_after_usage = shutil.disk_usage(checkpoint_root)
    disk_info = {
        "disk_before": disk_before,
        "disk_after": {"total": disk_after_usage.total, "used": disk_after_usage.used, "free": disk_after_usage.free},
        "resumed_from": resume_from,
        "disk_estimate": disk_estimate,
        "checkpoint_identity_context": identity_context,
        "planned_optimizer_steps": planned_steps,
        "final_export": final_export,
    }
    return trainer, disk_info


def collect_environment_manifest(project_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest: dict[str, Any] = {"git_commit": commit, "python_version": sys.version, "platform": platform.platform()}
    try:
        import torch
    except ImportError:
        return manifest
    manifest["torch_version"] = torch.__version__
    manifest["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        manifest["gpu"] = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(0)),
            "cuda_version": torch.version.cuda,
        }
    return manifest


def main() -> int:
    args = parse_args()
    _enforce_mode_flag_contract(args.mode, args.enable_production_training, args.max_steps)
    project_root = PROJECT_ROOT
    run_id, run_dir = EV.make_run_dir(project_root / "runs")
    command = shlex.join(["frozen-python", *sys.argv])
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    log_handle = (run_dir / "console.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = EV.Tee(original_stdout, log_handle)
    sys.stderr = EV.Tee(original_stderr, log_handle)
    started = dt.datetime.now().astimezone()
    status = "failed"
    result_summary: dict[str, Any] = {}
    env_manifest: dict[str, Any] = {}
    try:
        config = json.loads((project_root / args.config).read_text(encoding="utf-8"))
        if args.model_key not in config["models"]:
            raise ValueError(f"Unknown model key {args.model_key!r}; available: {sorted(config['models'])}")
        if args.checkpoint_root:
            config["checkpoint"] = {**config["checkpoint"], "root_dir": args.checkpoint_root}

        rows, group_sizes, batch_summary = load_batch_plan(project_root, config["batch_plan"])
        if args.max_batches is not None:
            group_sizes = group_sizes[: args.max_batches]
            rows = rows[: sum(group_sizes)]
        print(f"[batch-plan] digest verified; {len(group_sizes)} groups, {len(rows)} rows in use")

        model_settings = config["models"][args.model_key]
        env_manifest = collect_environment_manifest(project_root)

        if args.mode == "tokenizer_stats":
            analysis_cfg = config["tokenizer_length_analysis"]
            tokenizer = load_tokenizer(model_settings)
            stats = tokenizer_length_stats(
                tokenizer, rows, analysis_cfg["fields"], analysis_cfg.get("sample_size"),
                int(config["schedule"]["seed"]), max_seq_length=int(model_settings.get("max_seq_length", 512)),
            )
            result_summary = {"tokenizer_length_stats": stats}
            print(f"       tokenizer_length_stats: {stats}")

        elif args.mode == "validate":
            _model, metadata = build_trainable_model(model_settings)
            result_summary = {"model_metadata": metadata}
            print(f"       model construction validated: {metadata}")

        else:  # smoke or train
            model, metadata = build_trainable_model(model_settings)
            dataset = build_training_dataset(rows)
            checkpoint_root = resolve_checkpoint_root(config["checkpoint"], args.model_key)
            default_smoke_steps = 5
            resolved_max_steps = args.max_steps if args.max_steps is not None else (
                default_smoke_steps if args.mode == "smoke" else None
            )
            trainer, disk_info = run_training(
                model, metadata, dataset, group_sizes, config, checkpoint_root, args.model_key,
                git_commit=env_manifest["git_commit"], max_steps=resolved_max_steps,
            )
            loss_history = [
                {"step": entry["step"], "loss": entry["loss"]}
                for entry in trainer.state.log_history if "loss" in entry
            ]
            result_summary = {
                "model_metadata": metadata,
                "global_step": trainer.state.global_step,
                "loss_history": loss_history,
                "checkpoint_root": str(checkpoint_root),
                **disk_info,
            }
            final_export = disk_info["final_export"]
            print(f"       global_step={trainer.state.global_step} losses={len(loss_history)} logged")
            print(
                f"       final export: {final_export['path']} "
                f"(global_step={final_export['global_step']}, {final_export['total_bytes']:,} bytes, "
                f"model.safetensors sha256={final_export['model_safetensors_sha256']})"
            )

        EV.write_json(run_dir / "summary.json", {
            "run_id": run_id, "mode": args.mode, "model_key": args.model_key,
            "batch_plan_digest": batch_summary["deterministic_digest"],
            "batch_groups_used": len(group_sizes), "batch_rows_used": len(rows),
            "license_label": model_settings.get("license_label"),
            **result_summary, **env_manifest,
        })
        print(f"\nOutput: {run_dir.relative_to(project_root)}")
        status = "completed"
        return 0
    except Exception as error:  # noqa: BLE001 - surfaced in the run summary
        print(f"[!] Training run failed: {error}")
        raise
    finally:
        finished = dt.datetime.now().astimezone()
        EV.write_json(run_dir / "run_manifest.json", {
            "run_id": run_id, "script": "scripts/train_contrastive.py", "command": command,
            "started_at": started.isoformat(), "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - started).total_seconds(), 3),
            "mode": args.mode, "model_key": args.model_key, "python_version": sys.version,
            "status": status, **env_manifest,
        })
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
