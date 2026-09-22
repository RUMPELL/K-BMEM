#!/usr/bin/env python3
"""Build deterministic, collision-safe MCQ contrastive batch plans.

This creates *data plans*, not training examples for a particular library.  A
plan row has one query, positive and selected hard negative; a consumer may use
the explicit negative and the other batch positives as negatives without
creating a known normalized-text false negative.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import shlex
import shutil
import sys
import unicodedata
from typing import Any, Iterator


SPACE_RE = re.compile(r"\s+")
REGISTERED_INPUTS = {
    "data/processed_aihub_qa_v3/mcq/triplets.jsonl.gz",
    "data/processed_exam_v1/triplets.jsonl.gz",
}


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


def clean_text(value: Any) -> str:
    """NFC and whitespace normalization used for every collision key."""
    text = value if isinstance(value, str) else "" if value is None else str(value)
    return unicodedata.normalize("NFC", SPACE_RE.sub(" ", text).strip())


def normalization_key(value: Any) -> str:
    """A conservative Korean-safe key: NFC, casefold, whitespace/punctuation-free."""
    text = clean_text(value).casefold()
    return "".join(char for char in text if char.isalnum())


def stable_int(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=16, person=b"kbmem-batch-v1")
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "big")


def json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


@contextlib.contextmanager
def deterministic_gzip_text(path: Path, compresslevel: int) -> Iterator[io.TextIOWrapper]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=compresslevel, mtime=0)
    wrapper = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    try:
        yield wrapper
    finally:
        wrapper.close()
        raw.close()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Record at {path}:{line_number} is not an object")
            yield record


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def length_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    positives = [len(row["positive"]) for row in records]
    negatives = [len(row["negative"]) for row in records]
    return {
        "positive_mean": round(sum(positives) / len(positives), 6),
        "negative_mean": round(sum(negatives) / len(negatives), 6),
        "mean_gap": round(abs(sum(positives) / len(positives) - sum(negatives) / len(negatives)), 6),
        "positive_quantile": round(percentile(positives, 0.5), 6),
        "negative_quantile": round(percentile(negatives, 0.5), 6),
        "quantile_gap": round(abs(percentile(positives, 0.5) - percentile(negatives, 0.5)), 6),
    }


def meets_length_bound(records: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    stats = length_stats(records)
    return (stats["mean_gap"] <= float(config["max_mean_length_gap"]) and
            stats["quantile_gap"] <= float(config["max_quantile_length_gap"]))


def row_id(record: dict[str, Any], source: str, index: int) -> str:
    supplied = clean_text(record.get("triplet_id") or record.get("question_id"))
    return supplied or f"{source}:{index:09d}"


def candidate_rows(config: dict[str, Any], project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]], Counter[str]]:
    valid: list[dict[str, Any]] = []
    quarantine: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for source_config in config["inputs"]:
        source = str(source_config["path"])
        input_path = (project_root / source).resolve()
        for index, record in enumerate(iter_jsonl(input_path), start=1):
            record_id = row_id(record, source, index)
            split = clean_text(record.get("split"))
            if split != "train":
                quarantine.append({"source": source, "record_id": record_id, "reason": "non_train_split"})
                counts["non_train_split"] += 1
                continue
            positive = clean_text(record.get("positive"))
            positive_key = normalization_key(positive)
            if not positive_key:
                quarantine.append({"source": source, "record_id": record_id, "reason": "empty_positive"})
                counts["empty_positive"] += 1
                continue
            if not isinstance(record.get("option_len_rank"), int):
                quarantine.append({"source": source, "record_id": record_id, "reason": "missing_option_len_rank"})
                counts["missing_option_len_rank"] += 1
                continue
            negatives: list[tuple[str, str]] = []
            seen_negative_keys: set[str] = set()
            for raw_negative in record.get("hard_negatives", []):
                negative = clean_text(raw_negative)
                negative_key = normalization_key(negative)
                if negative_key and negative_key != positive_key and negative_key not in seen_negative_keys:
                    negatives.append((negative, negative_key))
                    seen_negative_keys.add(negative_key)
            if not negatives:
                quarantine.append({"source": source, "record_id": record_id, "reason": "no_usable_hard_negative"})
                counts["no_usable_hard_negative"] += 1
                continue
            valid.append({"source": source, "record_id": record_id, "query": clean_text(record.get("query")),
                          "positive": positive, "positive_key": positive_key, "negatives": negatives,
                          "option_len_rank": record["option_len_rank"]})
    return valid, quarantine, counts


def collision_pairs(rows: list[dict[str, Any]]) -> int:
    positives = Counter(row["positive_key"] for row in rows)
    total = 0
    for row in rows:
        candidates = row["negatives"] if "negatives" in row else [(row["negative"], row["negative_key"])]
        for _, negative_key in candidates:
            total += positives[negative_key] - (1 if row["positive_key"] == negative_key else 0)
    return total


def select_and_pack(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], list[dict[str, str]], Counter[str]]:
    seed = str(config["seed"])
    batch_size = int(config["batch_size"])
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["option_len_rank"]].append(row)
    ordered: list[dict[str, Any]] = []
    # Round-robin ranks so the stored option length rank cannot dominate early batches.
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (stable_int(seed, "row", row["record_id"]), row["record_id"]))
    while any(buckets.values()):
        for rank in sorted(buckets):
            if buckets[rank]:
                ordered.append(buckets[rank].pop(0))

    batches: list[list[dict[str, Any]]] = []
    quarantine: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in ordered:
        choices = sorted(row["negatives"], key=lambda item: (stable_int(seed, "negative", row["record_id"], item[1]), item[1]))
        placed = False
        # Prefer the oldest compatible batch for deterministic, compact output.
        for batch in batches:
            if len(batch) >= batch_size:
                continue
            positive_keys = {item["positive_key"] for item in batch}
            negative_keys = {item["negative_key"] for item in batch}
            if row["positive_key"] in positive_keys or row["positive_key"] in negative_keys:
                continue
            for negative, negative_key in choices:
                if negative_key in positive_keys or negative_key == row["positive_key"]:
                    continue
                proposed = batch + [{**row, "negative": negative, "negative_key": negative_key}]
                if meets_length_bound(proposed, config):
                    batch.append(proposed[-1])
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # A singleton must itself meet the configured bound; otherwise no valid batch exists.
            for negative, negative_key in choices:
                proposed = [{**row, "negative": negative, "negative_key": negative_key}]
                if meets_length_bound(proposed, config):
                    batches.append(proposed)
                    placed = True
                    break
        if not placed:
            quarantine.append({"source": row["source"], "record_id": row["record_id"], "reason": "length_balance_impossible"})
            counts["length_balance_impossible"] += 1

    minimum = int(config["minimum_batch_size"])
    emitted: list[list[dict[str, Any]]] = []
    for batch in batches:
        if len(batch) >= minimum:
            emitted.append(batch)
        else:
            for row in batch:
                quarantine.append({"source": row["source"], "record_id": row["record_id"], "reason": "insufficient_batch_size"})
                counts["insufficient_batch_size"] += 1
    return emitted, quarantine, counts


def verify_batches(batches: list[list[dict[str, Any]]], config: dict[str, Any]) -> None:
    for number, batch in enumerate(batches):
        positives = [row["positive_key"] for row in batch]
        negatives = [row["negative_key"] for row in batch]
        if len(positives) != len(set(positives)):
            raise AssertionError(f"batch {number} has duplicate positive key")
        if set(positives) & set(negatives):
            raise AssertionError(f"batch {number} has positive/negative collision")
        if not meets_length_bound(batch, config):
            raise AssertionError(f"batch {number} exceeds length-balance bound")


def build(config: dict[str, Any], project_root: Path, stage_dir: Path) -> dict[str, Any]:
    validate_config(config, project_root)
    rows, quarantine, reasons = candidate_rows(config, project_root)
    before = collision_pairs(rows)
    batches, packing_quarantine, packing_reasons = select_and_pack(rows, config)
    quarantine.extend(packing_quarantine)
    reasons.update(packing_reasons)
    verify_batches(batches, config)

    plan_path = stage_dir / "batches.jsonl.gz"
    batch_stats: list[dict[str, Any]] = []
    with deterministic_gzip_text(plan_path, int(config.get("gzip_compresslevel", 6))) as handle:
        for batch_number, batch in enumerate(batches):
            batch_id = f"batch_{batch_number:06d}"
            stats = length_stats(batch)
            batch_stats.append({"batch_id": batch_id, "size": len(batch), **stats})
            for row in batch:
                handle.write(json_line({"batch_id": batch_id, "record_id": row["record_id"], "source": row["source"],
                                        "query": row["query"], "positive": row["positive"], "negative": row["negative"],
                                        "option_len_rank": row["option_len_rank"]}))
    quarantine.sort(key=lambda row: (row["source"], row["record_id"], row["reason"]))
    with (stage_dir / "quarantine.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "record_id", "reason"])
        writer.writeheader()
        writer.writerows(quarantine)
    overall = length_stats([row for batch in batches for row in batch]) if batches else {key: 0.0 for key in ("positive_mean", "negative_mean", "mean_gap", "positive_quantile", "negative_quantile", "quantile_gap")}
    summary = {"records_read": len(rows) + sum(reasons.values()) - sum(packing_reasons.values()), "candidate_records": len(rows),
               "emitted_records": sum(len(batch) for batch in batches), "batch_count": len(batches),
               "collision_pairs_before": before, "collision_pairs_after": 0, "quarantined": len(quarantine),
               "quarantine_reasons": dict(sorted(reasons.items())), "length_balance": overall,
               "batch_length_balance": batch_stats, "seed": str(config["seed"])}
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary["deterministic_digest"] = hashlib.sha256(plan_path.read_bytes() + encoded).hexdigest()
    (stage_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def validate_config(config: dict[str, Any], project_root: Path) -> None:
    required = ("inputs", "seed", "batch_size", "minimum_batch_size", "max_mean_length_gap", "max_quantile_length_gap")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    if int(config["batch_size"]) < int(config["minimum_batch_size"]) or int(config["minimum_batch_size"]) < 1:
        raise ValueError("Require batch_size >= minimum_batch_size >= 1")
    if float(config["max_mean_length_gap"]) < 0 or float(config["max_quantile_length_gap"]) < 0:
        raise ValueError("Length gap bounds must be non-negative")
    for source in config["inputs"]:
        path = str(source["path"])
        if config.get("enforce_registered_inputs", True) and path not in REGISTERED_INPUTS:
            raise ValueError(f"Unregistered contrastive input: {path}")
        if not (project_root / path).is_file():
            raise FileNotFoundError(f"Input not found: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = (project_root / config["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    run_id = "run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (project_root / config.get("runs_dir", "runs") / run_id).resolve()
    stage_dir = output_dir.with_name(output_dir.name + ".staging-" + run_id)
    if stage_dir.exists():
        raise FileExistsError(f"Staging path already exists: {stage_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "command.txt").open("w", encoding="utf-8") as command, (run_dir / "console.log").open("w", encoding="utf-8") as log:
        command.write(" ".join(shlex.quote(part) for part in sys.argv) + "\n")
        old_stdout = sys.stdout
        sys.stdout = Tee(old_stdout, log)
        try:
            stage_dir.mkdir(parents=True)
            summary = build(config, project_root, stage_dir)
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(json.dumps({"script": "scripts/build_contrastive_batches.py", "config": config, "output_dir": str(output_dir), "run_id": run_id}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            stage_dir.replace(output_dir)
            print(json.dumps({"batch_count": summary["batch_count"], "deterministic_digest": summary["deterministic_digest"], "quarantined": summary["quarantined"]}, ensure_ascii=False, sort_keys=True))
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        finally:
            sys.stdout = old_stdout
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
