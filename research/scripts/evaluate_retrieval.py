#!/usr/bin/env python3
"""Evaluate retrieval and multiple-choice discrimination over K-BMEM datasets.

The retriever is deliberately behind a small interface so that BM25, the
zero-shot embedding baselines and any fine-tuned checkpoint are all scored by
the same code path. Only the retriever changes between runs; the corpus loading,
metric computation and reporting stay fixed, which is what makes the numbers
comparable across models.

Two task shapes are supported:

* ``retrieval`` - rank a corpus for each query and score nDCG/MRR/Recall.
* ``mcq`` - rank the options belonging to one question and score accuracy@1.
  Recall over five candidates is meaningless, so it is not reported.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Iterable, Iterator


HANGUL_RE = re.compile(r"[가-힣]")
LATIN_NUM_RE = re.compile(r"[A-Za-z0-9]+")
NON_WORD_RE = re.compile(r"[^가-힣A-Za-z0-9]+")


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, message: str) -> int:
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the evaluation JSON configuration")
    parser.add_argument("--models-config", help="Optional frozen model-matrix JSON configuration")
    parser.add_argument("--model-key", help="Model key selected from --models-config")
    args = parser.parse_args()
    if bool(args.models_config) != bool(args.model_key):
        parser.error("--models-config and --model-key must be provided together")
    return args


def select_model_config(
    base_config: dict[str, Any], models_config: dict[str, Any], model_key: str
) -> dict[str, Any]:
    """Merge one frozen real-model entry into the dev-only evaluation contract."""
    models = models_config.get("models", {})
    if model_key not in models:
        raise ValueError(f"Unknown model key {model_key!r}; available: {sorted(models)}")
    selected = dict(models[model_key])
    merged = json.loads(json.dumps(base_config))
    merged["retriever"] = {**merged.get("retriever", {}), **selected}
    merged["_model_key"] = model_key
    merged["_license_label"] = selected.get("license_label")
    return merged


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")
    temporary.replace(path)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dig(record: dict[str, Any], field: str) -> Any:
    current: Any = record
    for part in field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

def tokenize(text: str, mode: str) -> list[str]:
    """Split text into index terms.

    Korean is written without spaces between morphemes, so whitespace tokens
    make poor index terms and a morphological analyser would be an extra
    dependency. Character bigrams are the standard analyser-free substitute and
    they degrade gracefully on the mixed Korean/English strings this project
    targets: Latin runs stay whole words, Hangul runs become bigrams.
    """
    if mode == "whitespace":
        return [t for t in NON_WORD_RE.split(text.lower()) if t]

    tokens: list[str] = [match.group(0).lower() for match in LATIN_NUM_RE.finditer(text)]
    for run in NON_WORD_RE.split(text):
        hangul = "".join(ch for ch in run if HANGUL_RE.match(ch))
        if len(hangul) == 1:
            tokens.append(hangul)
        for index in range(len(hangul) - 1):
            tokens.append(hangul[index: index + 2])
    return tokens


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------

class BM25Retriever:
    """Okapi BM25 over an in-memory inverted index (standard library only)."""

    name = "bm25"

    def __init__(self, settings: dict[str, Any]) -> None:
        self.k1 = float(settings.get("k1", 1.2))
        self.b = float(settings.get("b", 0.75))
        self.tokenizer = str(settings.get("tokenizer", "char_bigram"))
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avg_len = 0.0
        self.idf: dict[str, float] = {}

    def index(self, documents: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in documents:
            counts = Counter(tokenize(text, self.tokenizer))
            position = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            self.doc_len.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((position, frequency))
        total = len(self.doc_ids)
        self.avg_len = (sum(self.doc_len) / total) if total else 0.0
        for term, entries in self.postings.items():
            df = len(entries)
            self.idf[term] = math.log(1.0 + (total - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term, query_frequency in Counter(tokenize(query, self.tokenizer)).items():
            entries = self.postings.get(term)
            if not entries:
                continue
            idf = self.idf[term]
            for position, frequency in entries:
                norm = 1.0 - self.b + self.b * (self.doc_len[position] / self.avg_len if self.avg_len else 0.0)
                scores[position] += idf * (frequency * (self.k1 + 1.0)) / (frequency + self.k1 * norm)
        # Ties are broken by doc id so a run is reproducible.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[position], score) for position, score in ranked[:top_k]]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else list(vector)


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot_product(a, b) / (norm_a * norm_b)


SIMILARITY_FUNCTIONS = {"cosine": _cosine_similarity, "dot_product": _dot_product}


def _similarity_fn(name: str):
    if name not in SIMILARITY_FUNCTIONS:
        raise ValueError(f"Unsupported similarity function: {name}. Available: {sorted(SIMILARITY_FUNCTIONS)}")
    return SIMILARITY_FUNCTIONS[name]


def _config_digest(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# docs/task06_phase_b_gate.md Sections 1 and 3 freeze this exact model+code
# revision pair as the only combination ever allowed to run with trusted remote
# code. Moving either pin requires a new gate record, not a config edit here.
JINA_REMOTE_CODE_PIN = {
    "model_id": "jinaai/jina-embeddings-v3",
    "model_revision": "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
    "code_revision": "845308d0fd72a8406a3e378450e1a09522790419",
    "query_adapter": "retrieval.query",
    "passage_adapter": "retrieval.passage",
}


class _PinnedCodeRevisionKwargs(dict[str, Any]):
    """Preserve Jina code_revision across SentenceTransformers 5.7 legacy loading.

    SentenceTransformers consumes ``model_kwargs.pop("code_revision")`` while
    importing ``custom_st.py``, then reuses the same mapping for the custom
    module model arguments. A normal dict therefore loses the independently
    pinned implementation revision before ``AutoModel.from_pretrained`` sees it.
    This Jina-only mapping lets that one security pin be read without deletion;
    every other key keeps normal dict.pop semantics.
    """

    def pop(self, key: str, *default: Any) -> Any:
        if key == "code_revision" and key in self:
            return self[key]
        return super().pop(key, *default)


def _validate_remote_code_controls(
    model_id: str,
    requested_trust_remote_code: bool,
    requested_model_revision: str | None,
    requested_code_revision: str | None,
    requested_query_adapter: str | None,
    requested_passage_adapter: str | None,
) -> dict[str, Any]:
    """Fail closed on Jina's trusted-code/adapter controls before any model construction.

    This is the single point that decides whether `trust_remote_code=True` is ever
    passed to the model constructor. Only the exact reviewed Jina model+code revision
    pair from `JINA_REMOTE_CODE_PIN` may request it, and only together with both its
    query/passage task adapters - every other model, including KURE and BAAI, is
    refused before construction rather than allowed to silently run without remote
    code review coverage.
    """
    pin = JINA_REMOTE_CODE_PIN
    is_pinned_jina = model_id == pin["model_id"]

    if not requested_trust_remote_code:
        if requested_query_adapter or requested_passage_adapter or requested_code_revision:
            raise ValueError(
                f"Model '{model_id}' requested a Jina task adapter or code_revision without "
                "trust_remote_code=True. retrieval.query/retrieval.passage adapters and a pinned "
                "code_revision only exist behind Jina's reviewed remote-code path - set "
                "trust_remote_code=True with the exact reviewed model/code revisions and both "
                "adapters, or drop these settings entirely for a standard-module model."
            )
        return {
            "trust_remote_code": False,
            "model_revision": requested_model_revision,
            "code_revision": None,
            "query_adapter": None,
            "passage_adapter": None,
        }

    if not is_pinned_jina:
        raise ValueError(
            f"trust_remote_code=True was requested for '{model_id}', but only the reviewed Jina "
            f"candidate ('{pin['model_id']}') is permitted to execute remote code "
            "(docs/task06_phase_b_gate.md Section 3). KURE and BAAI must run with "
            "trust_remote_code=False; refusing to construct this model with remote code trusted."
        )
    if requested_model_revision != pin["model_revision"]:
        raise ValueError(
            f"Jina trust_remote_code=True requires the exact reviewed model revision "
            f"'{pin['model_revision']}'; config requested {requested_model_revision!r}. A changed "
            "upstream branch or pin requires a new gate review before this evaluator will "
            "construct it."
        )
    if requested_code_revision != pin["code_revision"]:
        raise ValueError(
            f"Jina trust_remote_code=True requires the exact reviewed code_revision "
            f"'{pin['code_revision']}'; config requested {requested_code_revision!r}."
        )
    if requested_query_adapter != pin["query_adapter"]:
        raise ValueError(
            f"Jina requires query_adapter '{pin['query_adapter']}' for queries; config requested "
            f"{requested_query_adapter!r}. Refusing to run Jina without its reviewed query task "
            "adapter."
        )
    if requested_passage_adapter != pin["passage_adapter"]:
        raise ValueError(
            f"Jina requires passage_adapter '{pin['passage_adapter']}' for passages and MCQ "
            f"options; config requested {requested_passage_adapter!r}. Refusing to run Jina "
            "without its reviewed passage task adapter."
        )
    return {
        "trust_remote_code": True,
        "model_revision": requested_model_revision,
        "code_revision": requested_code_revision,
        "query_adapter": requested_query_adapter,
        "passage_adapter": requested_passage_adapter,
    }


class MockEncoder:
    """Deterministic, dependency-free stand-in for a real embedding model.

    Hashes character bigrams into a fixed-size vector (the hashing trick), which
    is enough to exercise the full batching/index/search/normalization path with
    no ML package and no network access. It is never a substitute for a real
    zero-shot score - `library_version` reports the literal string "mock" so
    that fact cannot be lost downstream.
    """

    provider_name = "mock"
    library_version = "mock"

    def __init__(self, settings: dict[str, Any]) -> None:
        self.model_id = str(settings.get("model_name_or_path", "mock-encoder"))
        self.revision_or_path = str(settings.get("revision") or settings.get("local_model_path") or "mock")
        self.dim = int(settings.get("mock_embedding_dim", 32))
        native = settings.get("mock_native_max_seq_length")
        self._native_max_seq_length = int(native) if native is not None else None
        # The hashing trick has no dtype/pooling/remote-code/adapter concept to
        # honor or violate, so it never fails closed on any of these settings -
        # it just says so plainly rather than echoing back the request as if it
        # had been applied.
        self.dtype_applied = "not_applicable (mock backend)"
        self.pooling_applied = "not_applicable (mock backend)"
        self.model_revision_applied = "not_applicable (mock backend)"
        self.trust_remote_code_applied = "not_applicable (mock backend)"
        self.code_revision_applied = "not_applicable (mock backend)"
        self.query_adapter_applied = "not_applicable (mock backend)"
        self.passage_adapter_applied = "not_applicable (mock backend)"
        self.query_lora_task_applied = "not_applicable (mock backend)"
        self.passage_lora_task_applied = "not_applicable (mock backend)"
        self.query_prompt_text_applied = "not_applicable (mock backend)"
        self.passage_prompt_text_applied = "not_applicable (mock backend)"
        self.lora_task_routing_behaviorally_verified = "not_applicable (mock backend)"
        self.use_flash_attn_requested = "not_applicable (mock backend)"
        self.use_flash_attn_applied = "not_applicable (mock backend)"

    def applied_max_seq_length(self, requested: int) -> int:
        if self._native_max_seq_length is None:
            return requested
        return min(requested, self._native_max_seq_length)

    def encode(self, texts: list[str], max_length: int, role: str | None = None) -> list[list[float]]:
        applied = self.applied_max_seq_length(max_length)
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text[:applied], "char_bigram"):
                slot = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vector[slot] += 1.0
            vectors.append(vector)
        return vectors


class SentenceTransformerEncoder:
    """Real dense encoder backend. Never imported or constructed by BM25/mock paths.

    The `sentence-transformers` import happens here, inside __init__, and only
    when a caller explicitly asks for this provider - importing
    evaluate_retrieval.py, or running the BM25/mock tests, must not load torch,
    transformers, or sentence-transformers, and must never touch the network.

    Construction happens exactly once per encoder instance and does all of the
    work that can only be verified once a real model is loaded: passing
    `local_files_only=True` so no request can escape to the hub, applying the
    requested dtype and reading it back from the model's actual parameters
    instead of assuming the request succeeded, and confirming the requested
    pooling mode against the pretrained model's real pooling module rather than
    pretending this backend can override it. Callers that score many
    short-lived items - see DenseRetriever.build_shared_encoder - build one
    encoder and hand it to many DenseRetriever instances, so this expensive
    construction happens once per task, not once per item.
    """

    provider_name = "sentence_transformers"

    def __init__(self, settings: dict[str, Any]) -> None:
        try:
            import sentence_transformers
        except ImportError as error:
            raise RuntimeError(
                "retriever provider 'sentence_transformers' requires the optional "
                "'sentence-transformers' package, which is not importable in this environment. "
                "This script does not install packages or download models on its own - install "
                "the dependency and provide an approved local model path yourself, or use "
                "provider 'mock' for offline-compatible development and testing."
            ) from error
        model_path = settings.get("local_model_path")
        if not model_path:
            raise ValueError(
                "dense retriever settings require 'local_model_path' pointing at an approved "
                "local snapshot for provider 'sentence_transformers'. This script never downloads "
                "models, so a bare 'model_name_or_path' hub id is not accepted here - "
                "'model_name_or_path' is metadata only."
            )
        resolved_model_path = Path(str(model_path)).expanduser()
        self.model_id = str(settings.get("model_name_or_path", model_path))
        requested_model_revision = settings.get("revision")
        # Report the immutable revision (or the non-identifying configured ~ path
        # when no revision exists), never the expanded host-specific home path.
        self.revision_or_path = str(requested_model_revision or model_path)
        if resolved_model_path.exists() and requested_model_revision:
            # A contrastive-training final-step-<N> export directory is named after its
            # optimizer step, not the base model revision it was trained from - see
            # train_contrastive.py's FINAL_EXPORT_DIR_PREFIX. Its real provenance lives in
            # kbmem_checkpoint_identity.json (written atomically alongside the export), so when
            # that file is present it is the authoritative source for this check instead of the
            # directory name. Absent that file (a real HF hub snapshot directory), the directory
            # name is still required to equal the requested revision, unchanged from before.
            identity_path = resolved_model_path / "kbmem_checkpoint_identity.json"
            if identity_path.is_file():
                recorded_revision = json.loads(identity_path.read_text(encoding="utf-8")).get("model_revision")
                if recorded_revision != requested_model_revision:
                    raise ValueError(
                        f"Checkpoint identity revision mismatch for {self.model_id!r}: requested "
                        f"{requested_model_revision!r} but {identity_path} records model_revision "
                        f"{recorded_revision!r}. Refusing to construct from a different revision."
                    )
            else:
                applied_snapshot_revision = resolved_model_path.name
                if applied_snapshot_revision != requested_model_revision:
                    raise ValueError(
                        f"Local snapshot revision mismatch for {self.model_id!r}: requested "
                        f"{requested_model_revision!r} but local_model_path resolves to snapshot "
                        f"{applied_snapshot_revision!r}. Refusing to construct from a different revision."
                    )

        # Validated and resolved before any constructor call - trust_remote_code
        # must never reach SentenceTransformer() for a model/revision pair that
        # has not passed the Jina-only pin check in _validate_remote_code_controls.
        remote_code = _validate_remote_code_controls(
            model_id=self.model_id,
            requested_trust_remote_code=bool(settings.get("trust_remote_code", False)),
            requested_model_revision=requested_model_revision,
            requested_code_revision=settings.get("code_revision"),
            requested_query_adapter=settings.get("query_adapter"),
            requested_passage_adapter=settings.get("passage_adapter"),
        )

        self.library_version = sentence_transformers.__version__
        pinned_code_kwargs = (
            {"code_revision": remote_code["code_revision"]}
            if remote_code["code_revision"]
            else None
        )
        # TASK-08B-R2: Jina's config.json declares use_flash_attn=true, and its custom mha.py
        # hard-asserts qkv.dtype in (float16, bfloat16) whenever flash_attn is actually importable
        # - a plain float32 encode (this project's fixed dev/zero-shot evaluation dtype, chosen to
        # keep trained-vs-zero-shot arithmetic comparable) now crashes once flash_attn is installed
        # for training. use_flash_attn is a real XLMRobertaFlashConfig field, so overriding it via
        # AutoConfig.from_pretrained's config_kwargs (not model_kwargs - get_use_flash_attn() reads
        # config.use_flash_attn, not a constructor argument) reliably routes construction back
        # through the PyTorch-native attention path this project's fp32 evaluation contract
        # depends on, independent of whether flash_attn is installed in the running interpreter.
        requested_use_flash_attn = settings.get("use_flash_attn")
        config_overrides = dict(pinned_code_kwargs) if pinned_code_kwargs else {}
        if requested_use_flash_attn is not None:
            config_overrides["use_flash_attn"] = bool(requested_use_flash_attn)
        self._model = sentence_transformers.SentenceTransformer(
            str(resolved_model_path),
            device=settings.get("device", "cpu"),
            revision=requested_model_revision,
            trust_remote_code=remote_code["trust_remote_code"],
            model_kwargs=_PinnedCodeRevisionKwargs(pinned_code_kwargs) if pinned_code_kwargs else None,
            config_kwargs=config_overrides or None,
            local_files_only=True,
        )
        self.model_revision_applied = remote_code["model_revision"]
        self.trust_remote_code_applied = remote_code["trust_remote_code"]
        self.code_revision_applied = remote_code["code_revision"]
        self.use_flash_attn_requested = requested_use_flash_attn
        self.use_flash_attn_applied = self._read_applied_use_flash_attn()
        self._native_max_seq_length = int(getattr(self._model, "max_seq_length", 512))
        tokenizer = getattr(self._model, "tokenizer", None)
        self.tokenizer_class = type(tokenizer).__name__ if tokenizer is not None else None
        first_module = None
        try:
            first_module = self._model[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        architecture_config = getattr(getattr(first_module, "auto_model", None), "config", None)
        self.architecture_max_position_embeddings = getattr(
            architecture_config, "max_position_embeddings", None
        )
        self.adapter_route_counts: dict[str, int] = {}
        self.dtype_applied = self._apply_dtype(str(settings.get("dtype", "float32")))
        try:
            parameter_device = getattr(next(self._model.parameters()), "device", None)
        except StopIteration:
            parameter_device = None
        self.device_applied = str(parameter_device or settings.get("device", "cpu"))
        self.pooling_applied = self._verify_pooling(str(settings.get("pooling", "mean")))
        # query_adapter_applied/passage_adapter_applied record prompt-NAME selection only: proof
        # that the requested name exists as a key in the loaded model's `.prompts` text-prefix
        # mapping. That is necessary but not sufficient evidence that Jina's actual task-specific
        # LoRA adapter ran - see _verify_lora_task_routing below, which is the only source of
        # query_lora_task_applied/passage_lora_task_applied.
        self.query_adapter_applied, self.passage_adapter_applied = self._verify_task_adapters(
            remote_code["query_adapter"], remote_code["passage_adapter"]
        )
        self.query_prompt_text_applied = (
            self._model.prompts.get(self.query_adapter_applied) if self.query_adapter_applied else None
        )
        self.passage_prompt_text_applied = (
            self._model.prompts.get(self.passage_adapter_applied) if self.passage_adapter_applied else None
        )
        self.lora_task_routing_behaviorally_verified = False
        self.query_lora_task_applied, self.passage_lora_task_applied = self._verify_lora_task_routing(
            remote_code["query_adapter"], remote_code["passage_adapter"]
        )

    def _apply_dtype(self, requested_dtype: str) -> str:
        import torch

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if requested_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype '{requested_dtype}' for provider 'sentence_transformers'. "
                f"Available: {sorted(dtype_map)}. Refusing to guess or silently fall back to a "
                "different dtype before scoring."
            )
        try:
            self._model = self._model.to(dtype=dtype_map[requested_dtype])
            applied = next(self._model.parameters()).dtype
        except (StopIteration, RuntimeError, AttributeError) as error:
            raise ValueError(
                f"Could not apply and then confirm dtype '{requested_dtype}' on the loaded model: "
                f"{error}. Refusing to report an unverified dtype rather than guessing it worked."
            ) from error
        applied_canonical = str(applied).replace("torch.", "")
        if applied_canonical != requested_dtype:
            raise ValueError(
                f"Requested dtype '{requested_dtype}' was not applied: after .to(dtype=...) the "
                f"loaded model's parameters report dtype '{applied_canonical}' instead. This backend "
                "refuses to score with a dtype that does not match what was requested - the cast may "
                "have been silently ignored by the model or device. Set 'dtype' to the model's real "
                f"applied dtype ('{applied_canonical}'), or investigate why the cast did not take."
            )
        return applied_canonical

    def _verify_pooling(self, requested_pooling: str) -> str:
        applied = None
        for module in getattr(self._model, "_modules", {}).values():
            mode = getattr(module, "pooling_mode", None)
            if mode:
                applied = str(mode)
                break
        if applied is None:
            raise ValueError(
                f"Could not confirm the loaded model's actual pooling mode to compare against the "
                f"requested '{requested_pooling}'. Refusing to report an unverified pooling setting - "
                "this backend only proceeds when the applied mode is directly readable from the "
                "loaded model, not assumed from the request."
            )
        if applied != requested_pooling:
            raise ValueError(
                f"Requested pooling '{requested_pooling}' does not match the loaded model's actual "
                f"pooling mode '{applied}'. This backend cannot override a pretrained "
                "SentenceTransformer's pooling architecture - set 'pooling' in the config to the "
                f"model's real mode ('{applied}'), or choose a different model. Refusing to score "
                "with a pooling label that would not match what the model actually does."
            )
        return applied

    def _verify_task_adapters(
        self, requested_query_adapter: str | None, requested_passage_adapter: str | None
    ) -> tuple[str | None, str | None]:
        """Confirm the loaded model actually exposes the requested Jina task adapters.

        `_validate_remote_code_controls` only checked that the *config* asked for
        the reviewed adapter names; this checks the *loaded model object* the same
        way `_verify_pooling` checks pooling - an adapter name that is not present
        in the model's real `.prompts` mapping must fail before any encode() call
        rather than silently running a generic, unprompted encode path.
        """
        if requested_query_adapter is None and requested_passage_adapter is None:
            return None, None
        prompts = getattr(self._model, "prompts", None)
        if not isinstance(prompts, dict):
            raise ValueError(
                "A query/passage task adapter was requested, but the loaded model does not expose "
                "a '.prompts' mapping to select it from. Refusing to run a generic shared encode "
                "path without the reviewed task adapter actually being available on this model."
            )
        for label, adapter in (
            ("query_adapter", requested_query_adapter),
            ("passage_adapter", requested_passage_adapter),
        ):
            if adapter is not None and adapter not in prompts:
                raise ValueError(
                    f"Requested {label} '{adapter}' is not present in the loaded model's prompts "
                    f"({sorted(prompts)}). Refusing to score without the reviewed task adapter "
                    "rather than silently falling back to an unprompted encode call."
                )
        return requested_query_adapter, requested_passage_adapter

    def _verify_lora_task_routing(
        self, query_task: str | None, passage_task: str | None
    ) -> tuple[str | None, str | None]:
        """Behaviorally prove Jina's actual task-specific LoRA adapter - not just its text prompt
        prefix - is reachable from a real encode() call.

        TASK-08B-R1: prompt_name alone only selects which text is prepended
        (`SentenceTransformer._resolve_prompt`); it is never forwarded into
        `custom_st.Transformer.forward`'s `task` parameter, which is what
        actually selects the task-specific LoRA delta named in the model's
        `config.json` `lora_adaptations`. TASK-08B's own behavioral check
        proved passing `prompt_name` alone leaves the LoRA adapter inactive
        (embeddings from `prompt_name`-only and `task`-set calls differ
        materially), contradicting docs/task06_phase_b_gate.md Section 3-6's
        "fail closed rather than silently run Jina without the task adapter."
        This method is the fail-closed gate: it requires both a structural
        check (the requested task names are really present in the loaded
        model's `_lora_adaptations` list) and a live behavioral probe (a real
        forward call, observed by temporarily wrapping the loaded Transformer
        module, must actually receive the exact requested `task` value).
        Returns (None, None) for every non-Jina model, exactly mirroring
        `_verify_task_adapters`.
        """
        if query_task is None and passage_task is None:
            return None, None
        try:
            first_module = self._model[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            first_module = None
        lora_adaptations = getattr(first_module, "_lora_adaptations", None)
        if not isinstance(lora_adaptations, list) or not lora_adaptations:
            raise ValueError(
                f"Model {self.model_id!r} requested Jina LoRA tasks {query_task!r}/{passage_task!r}, but "
                "its loaded transformer module does not expose a non-empty '_lora_adaptations' list. "
                "Refusing to treat prompt-name-only routing as proof that a task-specific LoRA adapter "
                "was applied."
            )
        for label, task in (("query_adapter", query_task), ("passage_adapter", passage_task)):
            if task is not None and task not in lora_adaptations:
                raise ValueError(
                    f"Requested {label} LoRA task {task!r} is not present in the loaded model's real "
                    f"lora_adaptations ({sorted(lora_adaptations)}). Refusing to run without the reviewed "
                    "task adapter."
                )

        received: dict[str, Any] = {}
        original_forward = first_module.forward

        def _probe_forward(features, task=None, **kwargs):
            received["task"] = task
            return original_forward(features, task=task, **kwargs)

        first_module.forward = _probe_forward
        try:
            probe_text = ["k-bmem lora task routing behavioral probe"]
            for label, task in (("query", query_task), ("passage", passage_task)):
                if task is None:
                    continue
                received.clear()
                self._model.encode(
                    probe_text, prompt_name=task, task=task, batch_size=1, show_progress_bar=False,
                    convert_to_numpy=True,
                )
                if received.get("task") != task:
                    raise ValueError(
                        f"Behavioral probe failed for {label}: requested LoRA task {task!r} did not reach "
                        f"the loaded model's Transformer.forward (observed {received.get('task')!r} "
                        "instead). Refusing to trust prompt-name-only routing as evidence of LoRA "
                        "application - this is exactly the query/passage routing mismatch TASK-08B-R1's "
                        "gate exists to catch."
                    )
        finally:
            del first_module.forward
        self.lora_task_routing_behaviorally_verified = True
        return query_task, passage_task

    def _read_applied_use_flash_attn(self) -> bool | None:
        """Read back config.use_flash_attn actually applied to the loaded model's own config.

        None when no override was requested, or the loaded architecture has no such field at all
        (KURE/BAAI/Upskyy never request one). Raises if a requested override silently failed to
        apply - a mismatch here would silently invalidate the dtype/precision contract this
        evaluation's fp32-vs-zero-shot comparability depends on.
        """
        if self.use_flash_attn_requested is None:
            return None
        try:
            first_module = self._model[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            return None
        config = getattr(getattr(first_module, "auto_model", None), "config", None)
        applied = getattr(config, "use_flash_attn", None)
        if applied is None:
            return None
        applied = bool(applied)
        if applied != bool(self.use_flash_attn_requested):
            raise ValueError(
                f"Requested use_flash_attn={self.use_flash_attn_requested!r} but the loaded model's "
                f"config reports use_flash_attn={applied!r}. Refusing to report an unverified "
                "attention-kernel setting."
            )
        return applied

    def applied_max_seq_length(self, requested: int) -> int:
        return min(requested, self._native_max_seq_length)

    def encode(self, texts: list[str], max_length: int, role: str | None = None) -> list[list[float]]:
        """Encode a batch, routing to the query/passage task adapter for this role.

        `role` comes from the caller (DenseRetriever.index()/encode_queries()), not
        guessed here - see the asymmetric-adapter contract in
        docs/task06_phase_b_gate.md Section 3. A model with no adapters requested
        (KURE, BAAI) ignores `role` entirely and encodes exactly as before.
        """
        applied = self.applied_max_seq_length(max_length)
        original_max_seq_length = self._model.max_seq_length
        self._model.max_seq_length = applied
        adapter: str | None = None
        lora_task: str | None = None
        if role == "query":
            adapter = self.query_adapter_applied
            lora_task = self.query_lora_task_applied
        elif role == "passage":
            adapter = self.passage_adapter_applied
            lora_task = self.passage_lora_task_applied
        route_key = f"{role or 'unspecified'}:{adapter or 'no_adapter'}:{lora_task or 'no_lora_task'}"
        self.adapter_route_counts[route_key] = self.adapter_route_counts.get(route_key, 0) + 1
        encode_kwargs: dict[str, Any] = {
            "batch_size": max(len(texts), 1), "show_progress_bar": False, "convert_to_numpy": False,
        }
        if adapter is not None:
            encode_kwargs["prompt_name"] = adapter
        # TASK-08B-R1: prompt_name alone only selects the text prefix - it never reaches Jina's
        # custom_st.Transformer.forward's `task` parameter, which is what actually activates the
        # task-specific LoRA delta. Both must be passed together so a real score/train forward
        # pass genuinely applies the reviewed task adapter, not just its text prompt.
        if lora_task is not None:
            encode_kwargs["task"] = lora_task
        try:
            embeddings = self._model.encode(texts, **encode_kwargs)
        finally:
            self._model.max_seq_length = original_max_seq_length
        return [[float(x) for x in vector] for vector in embeddings]


ENCODER_PROVIDERS = {"mock": MockEncoder, "sentence_transformers": SentenceTransformerEncoder}


def build_encoder(settings: dict[str, Any]):
    provider = str(settings.get("provider", "mock"))
    if provider not in ENCODER_PROVIDERS:
        raise ValueError(f"Unknown dense encoder provider: {provider}. Available: {sorted(ENCODER_PROVIDERS)}")
    return ENCODER_PROVIDERS[provider](settings)


class DenseRetriever:
    """Batched dense retrieval behind the same index()/search() shape as BM25Retriever.

    Query and document encoding are configured and tracked separately - batch
    size, max sequence length, prefix - because collapsing them hides exactly
    the risk this project already hit: `upskyy/bge-m3-korean` reports
    `max_position_embeddings=8194` but `sentence_bert_config.json` locks
    `max_seq_length` to 512, and that only shows up if query/document length
    handling is inspectable independently.
    """

    name = "dense"

    def __init__(self, settings: dict[str, Any], encoder: Any = None) -> None:
        self.settings = dict(settings)
        self.query_batch_size = int(settings.get("query_batch_size", 32))
        self.document_batch_size = int(settings.get("document_batch_size", 32))
        self.query_max_seq_length = int(settings.get("query_max_seq_length", 512))
        self.document_max_seq_length = int(settings.get("document_max_seq_length", 512))
        self.query_prefix = str(settings.get("query_prefix", ""))
        self.document_prefix = str(settings.get("document_prefix", ""))
        self.pooling = str(settings.get("pooling", "mean"))
        self.normalize = bool(settings.get("normalize_embeddings", True))
        self.similarity = str(settings.get("similarity", "cosine"))
        self._similarity_fn = _similarity_fn(self.similarity)
        self.encoder = encoder if encoder is not None else build_encoder(settings)
        self.doc_ids: list[str] = []
        self.doc_vectors: list[list[float]] = []

    def _encode_batched(
        self, texts: list[str], batch_size: int, max_length: int, label: str, role: str
    ) -> list[list[float]]:
        applied = self.encoder.applied_max_seq_length(max_length)
        if applied != max_length:
            raise ValueError(
                f"{label} max_seq_length mismatch: requested {max_length} but provider "
                f"'{self.encoder.provider_name}' model '{self.encoder.model_id}' enforces {applied}. "
                "This is the upskyy/bge-m3-korean 512-lock risk (sentence_bert_config.json overrides "
                "the model's max_position_embeddings) - the harness refuses to silently truncate. "
                f"Either lower the requested {label.lower()} max length to {applied}, or confirm the "
                "provider's real cap and update the config explicitly."
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            vectors.extend(self.encoder.encode(batch, max_length=max_length, role=role))
        if self.normalize:
            vectors = [_l2_normalize(vector) for vector in vectors]
        return vectors

    def index(self, documents: Iterable[tuple[str, str]]) -> None:
        """Encode the document/passage side, including MCQ options.

        `evaluate_mcq()` indexes each question's options through this same method,
        so `role="passage"` covers both a retrieval corpus and MCQ candidates alike
        - Jina's `retrieval.passage` adapter applies to both without a separate path.
        """
        self.doc_ids = []
        texts: list[str] = []
        for doc_id, text in documents:
            self.doc_ids.append(doc_id)
            texts.append(self.document_prefix + text)
        self.doc_vectors = self._encode_batched(
            texts, self.document_batch_size, self.document_max_seq_length, "Document", role="passage"
        )

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.query_prefix + text for text in texts]
        return self._encode_batched(
            prefixed, self.query_batch_size, self.query_max_seq_length, "Query", role="query"
        )

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        scored = [(position, self._similarity_fn(query_vector, doc_vector)) for position, doc_vector in enumerate(self.doc_vectors)]
        # Ties are broken by doc id, same convention as BM25Retriever, so a run is reproducible.
        ranked = sorted(scored, key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[position], score) for position, score in ranked[:top_k]]

    @classmethod
    def build_shared_encoder(cls, settings: dict[str, Any]):
        """Build this retriever's encoder exactly once.

        Callers that construct many short-lived DenseRetriever instances - one
        per MCQ item, each with its own isolated option index - pass the same
        encoder into every instance instead of rebuilding (and, for a real
        backend, reloading) it per item. See evaluate_mcq().
        """
        return build_encoder(settings)

    def reproducibility_metadata(self, top_k_applied: Any) -> dict[str, Any]:
        """Reproducibility metadata, with requested vs. applied kept distinct.

        `top_k_applied` must come from the caller: this retriever alone cannot
        know whether it is being used for whole-corpus retrieval (a fixed
        cutoff) or per-question MCQ ranking (candidate count varies by
        question), so it is never guessed here.
        """
        top_k_requested = int(self.settings["top_k"]) if "top_k" in self.settings else None
        peak_vram_bytes = None
        if str(self.settings.get("device", "cpu")).startswith("cuda"):
            try:
                import torch
                peak_vram_bytes = int(torch.cuda.max_memory_allocated())
            except (ImportError, RuntimeError, AttributeError):
                pass
        return {
            "retriever": self.name,
            "provider": self.encoder.provider_name,
            "model_id": self.encoder.model_id,
            "revision_or_local_path": self.encoder.revision_or_path,
            "model_revision_requested": self.settings.get("revision"),
            "model_revision_applied": self.encoder.model_revision_applied,
            "trust_remote_code_requested": bool(self.settings.get("trust_remote_code", False)),
            "trust_remote_code_applied": self.encoder.trust_remote_code_applied,
            "code_revision_requested": self.settings.get("code_revision"),
            "code_revision_applied": self.encoder.code_revision_applied,
            # query_adapter_applied/passage_adapter_applied are prompt-NAME selection evidence only
            # (the requested name exists as a key in the model's .prompts text-prefix mapping) -
            # kept for backward compatibility with TASK-06/TASK-08A reports. They do not by
            # themselves prove Jina's task-specific LoRA adapter ran; see the *_lora_task_* and
            # *_prompt_text_* fields below, and lora_task_routing_behaviorally_verified, for that.
            "query_adapter_requested": self.settings.get("query_adapter"),
            "query_adapter_applied": self.encoder.query_adapter_applied,
            "passage_adapter_requested": self.settings.get("passage_adapter"),
            "passage_adapter_applied": self.encoder.passage_adapter_applied,
            "query_lora_task_requested": self.settings.get("query_adapter"),
            "query_lora_task_applied": getattr(self.encoder, "query_lora_task_applied", None),
            "passage_lora_task_requested": self.settings.get("passage_adapter"),
            "passage_lora_task_applied": getattr(self.encoder, "passage_lora_task_applied", None),
            "query_prompt_text_applied": getattr(self.encoder, "query_prompt_text_applied", None),
            "passage_prompt_text_applied": getattr(self.encoder, "passage_prompt_text_applied", None),
            "lora_task_routing_behaviorally_verified": getattr(
                self.encoder, "lora_task_routing_behaviorally_verified", False
            ),
            "use_flash_attn_requested": getattr(self.encoder, "use_flash_attn_requested", None),
            "use_flash_attn_applied": getattr(self.encoder, "use_flash_attn_applied", None),
            "device": str(self.settings.get("device", "cpu")),
            "device_applied": getattr(self.encoder, "device_applied", None),
            "dtype_requested": str(self.settings.get("dtype", "float32")),
            "dtype_applied": self.encoder.dtype_applied,
            "query_batch_size": self.query_batch_size,
            "document_batch_size": self.document_batch_size,
            "query_max_seq_length_requested": self.query_max_seq_length,
            "document_max_seq_length_requested": self.document_max_seq_length,
            "query_max_seq_length_applied": self.encoder.applied_max_seq_length(self.query_max_seq_length),
            "document_max_seq_length_applied": self.encoder.applied_max_seq_length(self.document_max_seq_length),
            "snapshot_native_max_seq_length": getattr(self.encoder, "_native_max_seq_length", None),
            "architecture_max_position_embeddings": getattr(
                self.encoder, "architecture_max_position_embeddings", None
            ),
            "tokenizer_class": getattr(self.encoder, "tokenizer_class", None),
            "adapter_route_counts": dict(getattr(self.encoder, "adapter_route_counts", {})),
            "peak_vram_bytes": peak_vram_bytes,
            "license_label": self.settings.get("license_label"),
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "pooling_requested": self.pooling,
            "pooling_applied": self.encoder.pooling_applied,
            "normalize_embeddings": self.normalize,
            "similarity": self.similarity,
            "top_k_requested": top_k_requested,
            "top_k_applied": top_k_applied,
            "provider_library_version": self.encoder.library_version,
            "config_digest": _config_digest(self.settings),
        }


RETRIEVERS = {"bm25": BM25Retriever, "dense": DenseRetriever}


def build_retriever(config: dict[str, Any]):
    settings = config.get("retriever", {"name": "bm25"})
    name = settings.get("name", "bm25")
    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever: {name}. Available: {sorted(RETRIEVERS)}")
    return RETRIEVERS[name](settings)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def dcg(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """Graded nDCG. Binary qrels are the special case where every gain is 1."""
    gains = [relevance.get(doc_id, 0.0) for doc_id in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return (dcg(gains) / denominator) if denominator else 0.0


def reciprocal_rank_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def bootstrap_interval(
    values: list[float], iterations: int, confidence: float, seed: int
) -> dict[str, float]:
    """Percentile bootstrap over per-query scores.

    The evaluation sets here are small - the exam test split holds 266 items,
    where a 95% interval is about six points wide - so a point estimate alone
    cannot support a claim that one model beats another. Every headline number
    is reported with its interval.
    """
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "half_width": 0.0, "n": 0}
    rng = random.Random(seed)
    size = len(values)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[max(int(tail * iterations) - 1, 0)]
    high = means[min(int((1.0 - tail) * iterations), iterations - 1)]
    mean = sum(values) / size
    return {
        "mean": round(mean, 4),
        "low": round(low, 4),
        "high": round(high, 4),
        "half_width": round((high - low) / 2.0, 4),
        "n": size,
    }


def code_switch_level(record: dict[str, Any], threshold: int) -> str:
    """Re-derive the script mix at an arbitrary threshold.

    The builders store `hangul_chars` and `latin_chars` next to the level they
    happened to pick, precisely so that reporting can move the boundary without
    a rebuild. Where "a Korean sentence with an English term" ends and "mixed"
    begins changes the subgroup sizes a lot: on the exam test split the mixed
    group is 75 items at 20 characters but 130 at 5, and the wider group gives a
    meaningfully tighter interval.
    """
    info = dig(record, "code_switch") or {}
    hangul = int(info.get("hangul_chars", 0))
    latin = int(info.get("latin_chars", 0))
    if hangul >= threshold and latin >= threshold:
        return "mixed"
    if hangul >= threshold:
        return "ko"
    if latin >= threshold:
        return "en"
    return "other"


# --------------------------------------------------------------------------
# Task loading
# --------------------------------------------------------------------------

def load_retrieval_task(project_root: Path, task: dict[str, Any]) -> dict[str, Any]:
    split = task["split"]
    query_field = task.get("query_text_field", "text")
    query_split_field = task.get("query_split_field", "split")

    queries: list[dict[str, Any]] = []
    for record in iter_jsonl((project_root / task["queries"]).resolve()):
        if str(dig(record, query_split_field)) != split:
            continue
        queries.append({
            "query_id": record[task.get("query_id_field", "query_id")],
            "text": dig(record, query_field) or "",
            "record": record,
        })

    documents: list[tuple[str, str]] = []
    judged: set[str] = set()
    for source in task["corpus"]:
        path = (project_root / source["path"]).resolve()
        id_field = source.get("id_field", "doc_id")
        text_field = source.get("text_field", "text")
        unjudged = bool(source.get("unjudged_distractors", False))
        for record in iter_jsonl(path):
            doc_id = str(dig(record, id_field))
            documents.append((doc_id, dig(record, text_field) or ""))
            if not unjudged:
                judged.add(doc_id)

    relevance: dict[str, dict[str, float]] = defaultdict(dict)
    qrels_path = (project_root / task["qrels"]).resolve()
    with qrels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if task.get("qrels_split_field") and row.get(task["qrels_split_field"]) != split:
                continue
            grade = float(row.get(task.get("relevance_field", "relevance"), 1))
            if grade > 0:
                relevance[row["query_id"]][row[task.get("qrels_doc_field", "doc_id")]] = grade

    return {"queries": queries, "documents": documents, "relevance": relevance, "judged": judged}


def load_pairs_task(project_root: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Build a retrieval task from records that carry both question and answer.

    MedQuAD ships question/answer on one row rather than as separate corpus and
    qrels files. Materialising those files would duplicate the data, so the
    corpus is assembled here: every answer in the configured splits becomes a
    document, answers with identical text collapse to one document, and each
    query's qrel points at its own answer.
    """
    split = task["split"]
    id_field = task.get("id_field", "record_id")
    question_field = task.get("question_field", "question")
    answer_field = task.get("answer_field", "answer")
    corpus_splits = set(task.get("corpus_splits") or [])

    doc_by_text: dict[str, str] = {}
    documents: list[tuple[str, str]] = []
    queries: list[dict[str, Any]] = []
    relevance: dict[str, dict[str, float]] = defaultdict(dict)

    for record in iter_jsonl((project_root / task["pairs"]).resolve()):
        answer = (dig(record, answer_field) or "").strip()
        record_split = str(record.get("split"))
        if answer and (not corpus_splits or record_split in corpus_splits):
            key = " ".join(answer.split()).lower()
            if key not in doc_by_text:
                doc_id = f"doc-{len(doc_by_text)}"
                doc_by_text[key] = doc_id
                documents.append((doc_id, answer))
        if record_split != split:
            continue
        question = (dig(record, question_field) or "").strip()
        if not question or not answer:
            continue
        query_id = str(record[id_field])
        queries.append({"query_id": query_id, "text": question, "record": record})
        relevance[query_id][doc_by_text[" ".join(answer.split()).lower()]] = 1.0

    judged = {doc_id for _, doc_id in doc_by_text.items()}
    return {"queries": queries, "documents": documents, "relevance": relevance, "judged": judged}


def load_mcq_task(project_root: Path, task: dict[str, Any]) -> list[dict[str, Any]]:
    split = task["split"]
    items: list[dict[str, Any]] = []
    for record in iter_jsonl((project_root / task["questions"]).resolve()):
        if str(record.get("split")) != split:
            continue
        items.append(record)
    return items


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_retrieval(task: dict[str, Any], data: dict[str, Any], retriever, cutoffs: dict[str, int]) -> dict[str, Any]:
    started = time.time()
    retriever.index(data["documents"])
    index_seconds = time.time() - started

    top_k = max(cutoffs.values())
    # Retrievers that expose encode_queries() get their queries embedded in one
    # batched call up front; BM25 has no such method, so it keeps searching by
    # raw query text exactly as before - this is the only branch point between
    # the two retriever families in this function.
    if hasattr(retriever, "encode_queries"):
        query_inputs = retriever.encode_queries([q["text"] for q in data["queries"]])
    else:
        query_inputs = [q["text"] for q in data["queries"]]

    rows: list[dict[str, Any]] = []
    started = time.time()
    for query, query_input in zip(data["queries"], query_inputs):
        relevance = data["relevance"].get(query["query_id"], {})
        ranked = retriever.search(query_input, top_k)
        ranked_ids = [doc_id for doc_id, _ in ranked]
        relevant = set(relevance)
        unjudged_top = sum(
            1 for doc_id in ranked_ids[: cutoffs["ndcg"]]
            if doc_id not in data["judged"] and doc_id not in relevant
        )
        rows.append({
            "query_id": query["query_id"],
            "has_qrel": bool(relevance),
            "ndcg": ndcg_at_k(ranked_ids, relevance, cutoffs["ndcg"]),
            "mrr": reciprocal_rank_at_k(ranked_ids, relevant, cutoffs["mrr"]),
            "recall": recall_at_k(ranked_ids, relevant, cutoffs["recall"]),
            "unjudged_in_top": unjudged_top,
            "top1": ranked_ids[0] if ranked_ids else "",
            "record": query["record"],
        })
    search_seconds = time.time() - started

    scored = [row for row in rows if row["has_qrel"]]
    def mean(field: str) -> float:
        return round(sum(r[field] for r in scored) / len(scored), 4) if scored else 0.0

    summary = {
        "task_type": "retrieval",
        "queries_total": len(rows),
        "queries_scored": len(scored),
        "queries_without_qrel": len(rows) - len(scored),
        "corpus_documents": len(data["documents"]),
        "judged_documents": len(data["judged"]),
        f"nDCG@{cutoffs['ndcg']}": mean("ndcg"),
        f"MRR@{cutoffs['mrr']}": mean("mrr"),
        f"Recall@{cutoffs['recall']}": mean("recall"),
        "mean_unjudged_in_top": round(sum(r["unjudged_in_top"] for r in scored) / len(scored), 3) if scored else 0.0,
        "index_seconds": round(index_seconds, 2),
        "search_seconds": round(search_seconds, 2),
    }
    if hasattr(retriever, "reproducibility_metadata"):
        # top_k here is the one real cutoff every search() call in this task
        # actually used - the true applied value, not the configured request.
        summary["retriever_metadata"] = retriever.reproducibility_metadata(top_k_applied=top_k)
    return {"summary": summary, "rows": rows, "cutoffs": cutoffs}


def evaluate_mcq(items: list[dict[str, Any]], retriever_settings: dict[str, Any]) -> dict[str, Any]:
    """Score each question against its own options.

    Every question gets a fresh index over its five candidates. The corpus is the
    question's own option set, so ranking here measures discrimination between
    deliberately plausible distractors rather than retrieval from a large pool.

    A retriever type may expose `build_shared_encoder()` (DenseRetriever does)
    to say "build my encoder/model once for the whole task, then hand it to a
    fresh instance per question." Without that hook - BM25's case - a brand new
    retriever is built per question exactly as before. Either way, every
    question still gets its own retriever instance indexed only over that
    question's own options, so option indexes never leak between questions.
    """
    rows: list[dict[str, Any]] = []
    retriever_metadata: dict[str, Any] | None = None
    # Validated before the item loop - and therefore before any shared-encoder
    # dispatch - so an unknown retriever name fails the same way build_retriever()
    # would, even for an empty item list, instead of silently defaulting to BM25.
    retriever_name = retriever_settings.get("name", "bm25")
    if retriever_name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever: {retriever_name}. Available: {sorted(RETRIEVERS)}")
    retriever_cls = RETRIEVERS[retriever_name]
    shared_encoder = (
        retriever_cls.build_shared_encoder(retriever_settings)
        if hasattr(retriever_cls, "build_shared_encoder")
        else None
    )
    started = time.time()
    for item in items:
        options = item["options"]
        answer_index = int(item["answer_index"]) - 1
        retriever = (
            retriever_cls(retriever_settings, encoder=shared_encoder)
            if shared_encoder is not None
            else build_retriever({"retriever": retriever_settings})
        )
        if retriever_metadata is None and hasattr(retriever, "reproducibility_metadata"):
            # The candidate count varies per question, so there is no single
            # "applied top_k" number to report here - see rows[].candidate_count
            # for the real per-question value instead of guessing one.
            retriever_metadata = retriever.reproducibility_metadata(
                top_k_applied="per-item candidate count (see rows[].candidate_count)"
            )
        retriever.index([(str(i), text) for i, text in enumerate(options)])
        if hasattr(retriever, "encode_queries"):
            query_input = retriever.encode_queries([item["query"]])[0]
        else:
            query_input = item["query"]
        ranked = retriever.search(query_input, len(options))
        ranked_ids = [doc_id for doc_id, _ in ranked]
        # Options the query shares no term with score zero and drop out entirely;
        # treat them as ranked last so every question still yields a rank.
        missing = [str(i) for i in range(len(options)) if str(i) not in ranked_ids]
        ranked_ids.extend(missing)
        gold = str(answer_index)
        rank = ranked_ids.index(gold) + 1
        # A model that beats chance but not this is exploiting option length,
        # not medical meaning: picking the longest option alone scores 39.8% on
        # the AIHub MCQ set against a 20.1% floor. The exam set does not carry
        # the cue (21.9%), which is part of why it is the primary metric.
        longest = max(len(o) for o in options)
        tied = [i for i, o in enumerate(options) if len(o) == longest]
        rows.append({
            "query_id": item["question_id"],
            "correct": rank == 1,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "longest_option_hit": (1.0 / len(tied)) if answer_index in tied else 0.0,
            "random_hit": 1.0 / len(options),
            "candidate_count": len(options),
            "record": item,
        })
    if shared_encoder is not None:
        final_retriever = retriever_cls(retriever_settings, encoder=shared_encoder)
        retriever_metadata = final_retriever.reproducibility_metadata(
            top_k_applied="per-item candidate count (see rows[].candidate_count)"
        )
    elapsed = time.time() - started

    total = len(rows)
    def mean(field: str) -> float:
        return round(sum(r[field] for r in rows) / total, 4) if total else 0.0

    summary = {
        "task_type": "mcq",
        "questions": total,
        "accuracy@1": mean("correct"),
        "MRR": mean("reciprocal_rank"),
        "random_baseline_accuracy": mean("random_hit"),
        "longest_option_baseline": mean("longest_option_hit"),
        "elapsed_seconds": round(elapsed, 2),
    }
    if retriever_metadata:
        summary["retriever_metadata"] = retriever_metadata
    return {"summary": summary, "rows": rows}


def subgroup_report(
    rows: list[dict[str, Any]],
    fields: list[str],
    metrics: list[str],
    code_switch_thresholds: list[int] | None = None,
    bootstrap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Break the per-query scores down by metadata field and by script mix.

    Script mix is not read from a stored label. It is re-derived from the raw
    character counts at each requested threshold, because the mixed group's size
    - and therefore how much the comparison can support - depends heavily on
    where that boundary sits.
    """
    def summarise(group: list[dict[str, Any]]) -> dict[str, Any]:
        entry: dict[str, Any] = {"n": len(group)}
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            entry[metric] = round(sum(values) / len(values), 4) if values else 0.0
            if bootstrap:
                entry[f"{metric}_ci"] = bootstrap_interval(
                    values,
                    int(bootstrap.get("iterations", 1000)),
                    float(bootstrap.get("confidence", 0.95)),
                    int(bootstrap.get("seed", 0)),
                )
        return entry

    report: dict[str, Any] = {}
    for field in fields:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(dig(row["record"], field))].append(row)
        report[field] = {
            key: summarise(group)
            for key, group in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        }

    for threshold in code_switch_thresholds or []:
        buckets = defaultdict(list)
        for row in rows:
            buckets[code_switch_level(row["record"], threshold)].append(row)
        if set(buckets) == {"other"}:
            continue  # no code_switch field on this dataset
        report[f"code_switch@{threshold}"] = {
            key: summarise(group)
            for key, group in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        }

    return report


def mixed_drop(subgroups: dict[str, Any], metric: str) -> dict[str, Any]:
    """Relative loss on mixed-script queries against Korean-only ones.

    Reported per threshold rather than as a single number: the boundary is a
    reporting choice, and a MixedDrop that flips sign when the threshold moves
    is telling you the effect is not there.

    This is observational, so it carries topic confounding - a mixed query may
    be harder because of its subject rather than its script. A controlled
    MixedDrop needs the same query in both forms, which requires the term
    dictionary from the KCI and Cochrane parallel pairs.
    """
    result: dict[str, Any] = {}
    for key, buckets in subgroups.items():
        if not key.startswith("code_switch@"):
            continue
        korean = buckets.get("ko")
        mixed = buckets.get("mixed")
        if not korean or not mixed or not korean.get(metric):
            continue
        base = float(korean[metric])
        result[key] = {
            "ko": base,
            "ko_n": korean["n"],
            "mixed": float(mixed[metric]),
            "mixed_n": mixed["n"],
            "drop": round((base - float(mixed[metric])) / base, 4) if base else 0.0,
        }
    return result


def write_outputs(output_dir: Path, name: str, result: dict[str, Any], subgroups: dict[str, Any], config: dict[str, Any]) -> None:
    candidates = (
        "ndcg", "mrr", "recall", "unjudged_in_top",
        "correct", "rank", "reciprocal_rank", "longest_option_hit", "random_hit",
    )
    metric_fields = [k for k in candidates if k in result["rows"][0]] if result["rows"] else []
    per_query = output_dir / "per_query.tsv"
    per_query.parent.mkdir(parents=True, exist_ok=True)
    with per_query.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", *metric_fields])
        for row in result["rows"]:
            writer.writerow([row["query_id"], *[row[f] for f in metric_fields]])

    write_json(output_dir / "summary.json", {
        "name": name,
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "config": config,
        "summary": result["summary"],
        "subgroups": subgroups,
    })

    lines = [f"# 검색 평가 — {name}", "", "## 요약", ""]
    for key, value in result["summary"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## 하위 그룹", ""]
    for field, buckets in subgroups.items():
        lines.append(f"### {field}")
        lines.append("")
        for key, stats in list(buckets.items())[:20]:
            detail = " · ".join(f"{m} {v}" for m, v in stats.items() if m != "n")
            lines.append(f"- `{key}` (n={stats['n']}): {detail}")
        lines.append("")
    (output_dir / "report_ko.md").write_text("\n".join(lines), encoding="utf-8")


def run_config(project_root: Path, config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Run every enabled, unlocked task in a config and return its aggregate summaries.

    A task is skipped - and therefore no retriever or encoder is ever
    constructed for it - the moment it is disabled or its split is locked.
    That check happens before `build_retriever`/`evaluate_mcq` are reached for
    every task type, so BM25 and dense retrieval share the exact same lock
    enforcement point; see the behavioral lock tests in
    tests/test_evaluate_retrieval.py.
    """
    cutoffs = config.get("cutoffs", {"ndcg": 10, "mrr": 10, "recall": 20})
    summaries: dict[str, Any] = {}
    for task in config["tasks"]:
        name = task["name"]
        if not task.get("enabled", True):
            print(f"[skip] {name}: disabled — {task.get('_disabled_note', 'no reason given')}")
            continue
        if task["split"] in config.get("locked_splits", ["test"]):
            print(f"[skip] {name}: split '{task['split']}' is locked")
            continue
        print(f"[task] {name} ({task['type']}, split={task['split']})")
        task_retriever = task.get("retriever", config.get("retriever", {}))
        if (
            task_retriever.get("provider") == "sentence_transformers"
            and str(task_retriever.get("device", "cpu")).startswith("cuda")
        ):
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if task["type"] in ("retrieval", "pairs_retrieval"):
            data = (
                load_pairs_task(project_root, task)
                if task["type"] == "pairs_retrieval"
                else load_retrieval_task(project_root, task)
            )
            print(f"       queries {len(data['queries']):,}  corpus {len(data['documents']):,}")
            retriever = build_retriever({**config, "retriever": task.get("retriever", config.get("retriever", {}))})
            result = evaluate_retrieval(task, data, retriever, cutoffs)
            metrics = ["ndcg", "mrr", "recall"]
        elif task["type"] == "mcq":
            items = load_mcq_task(project_root, task)
            print(f"       questions {len(items):,}")
            result = evaluate_mcq(items, task.get("retriever", config.get("retriever", {})))
            metrics = ["correct", "reciprocal_rank"]
        else:
            raise ValueError(f"Unknown task type: {task['type']}")

        scored = [r for r in result["rows"] if r.get("has_qrel", True)]
        thresholds = task.get("code_switch_thresholds", config.get("code_switch_thresholds", []))
        bootstrap = config.get("bootstrap")
        subgroups = subgroup_report(scored, task.get("subgroup_fields", []), metrics, thresholds, bootstrap)

        headline = task.get("headline_metric", metrics[0])
        if bootstrap and scored and headline in scored[0]:
            result["summary"][f"{headline}_ci"] = bootstrap_interval(
                [float(r[headline]) for r in scored],
                int(bootstrap.get("iterations", 1000)),
                float(bootstrap.get("confidence", 0.95)),
                int(bootstrap.get("seed", 0)),
            )
        drops = mixed_drop(subgroups, headline)
        if drops:
            result["summary"]["mixed_drop"] = drops

        write_outputs(output_root / name, name, result, subgroups, task)
        summaries[name] = result["summary"]
        for key, value in result["summary"].items():
            print(f"       {key}: {value}")

    return summaries


def make_run_dir(base: Path) -> tuple[str, Path]:
    run_id = "run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.name, run_dir


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()
    run_id, run_dir = make_run_dir(project_root / "runs")
    command = shlex.join(["frozen-python", *sys.argv])
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    log_handle = (run_dir / "console.log").open("w", encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    started = dt.datetime.now().astimezone()
    status = "failed"
    summaries: dict[str, Any] = {}
    try:
        config_path = (project_root / args.config).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.models_config and args.model_key:
            models_path = (project_root / args.models_config).resolve()
            models_config = json.loads(models_path.read_text(encoding="utf-8"))
            config = select_model_config(config, models_config, args.model_key)
        output_root = (project_root / config["output_dir"]).resolve()
        if args.model_key:
            output_root = output_root / args.model_key
        summaries = run_config(project_root, config, output_root)

        write_json(output_root / "summary.json", {
            "run_id": run_id,
            "generated_at": dt.datetime.now().astimezone().isoformat(),
            "command": command,
            "model_key": args.model_key,
            "license_label": config.get("_license_label"),
            "tasks": summaries,
        })
        print(f"\nOutput: {output_root.relative_to(project_root)}")
        status = "completed"
        return 0
    except Exception as error:  # noqa: BLE001 - surfaced in the run summary
        print(f"[!] Evaluation failed: {error}")
        raise
    finally:
        finished = dt.datetime.now().astimezone()
        write_json(run_dir / "run_manifest.json", {
            "run_id": run_id,
            "script": "scripts/evaluate_retrieval.py",
            "command": command,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - started).total_seconds(), 3),
            "model_key": args.model_key,
            "python_version": sys.version,
            "status": status,
        })
        write_json(run_dir / "summary.json", {"status": status, "run_id": run_id, "tasks": summaries})
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
