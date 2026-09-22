import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unicodedata
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "build_contrastive_batches.py"
SPEC = importlib.util.spec_from_file_location("build_contrastive_batches", MODULE_PATH)
BATCHES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = BATCHES
SPEC.loader.exec_module(BATCHES)


class ContrastiveBatchesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "fixture.jsonl.gz"

    def write_rows(self, rows):
        with gzip.open(self.input, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def config(self, **changes):
        config = {"inputs": [{"path": "fixture.jsonl.gz"}], "enforce_registered_inputs": False, "seed": "fixed",
                  "batch_size": 3, "minimum_batch_size": 2, "max_mean_length_gap": 2,
                  "max_quantile_length_gap": 2, "gzip_compresslevel": 6}
        config.update(changes)
        return config

    @staticmethod
    def row(identifier, positive, negatives, split="train", rank=1):
        return {"triplet_id": identifier, "split": split, "query": "질문 " + identifier,
                "positive": positive, "hard_negatives": negatives, "option_len_rank": rank}

    def test_korean_nfc_nfd_normalization_collides(self):
        nfd = "셸압"  # NFC 혈압
        nfd = unicodedata.normalize("NFD", "혈압")
        self.assertEqual(BATCHES.normalization_key("혈 압!"), BATCHES.normalization_key(nfd))

    def test_adversarial_collision_is_excluded(self):
        self.write_rows([
            self.row("one", "심근 경색", ["폐렴"]),
            self.row("two", "폐렴", ["심근경색"]),
            self.row("three", "천식", ["기흉"]),
            self.row("four", "기흉", ["천식"]),
        ])
        stage = self.root / "stage"
        stage.mkdir()
        summary = BATCHES.build(self.config(), self.root, stage)
        self.assertGreater(summary["collision_pairs_before"], 0)
        self.assertEqual(summary["collision_pairs_after"], 0)
        rows = []
        with gzip.open(stage / "batches.jsonl.gz", "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        for batch_id in {row["batch_id"] for row in rows}:
            batch = [row for row in rows if row["batch_id"] == batch_id]
            positives = {BATCHES.normalization_key(row["positive"]) for row in batch}
            negatives = {BATCHES.normalization_key(row["negative"]) for row in batch}
            self.assertFalse(positives & negatives)
            self.assertEqual(len(positives), len(batch))

    def test_non_train_is_quarantined(self):
        self.write_rows([
            self.row("one", "가나다", ["라마바사"]), self.row("dev", "라마바사", ["가나다"], split="dev"),
            self.row("two", "라마바사", ["사아자"]),
        ])
        stage = self.root / "stage"
        stage.mkdir()
        summary = BATCHES.build(self.config(), self.root, stage)
        self.assertEqual(summary["quarantine_reasons"]["non_train_split"], 1)

    def test_impossible_length_is_quarantined(self):
        self.write_rows([
            self.row("bad", "가", ["가" * 20]), self.row("one", "나다", ["라마바사"]), self.row("two", "라마바사", ["사아자"]),
        ])
        stage = self.root / "stage"
        stage.mkdir()
        summary = BATCHES.build(self.config(), self.root, stage)
        self.assertEqual(summary["quarantine_reasons"]["length_balance_impossible"], 1)

    def test_same_seed_is_byte_identical(self):
        self.write_rows([
            self.row("one", "가나다", ["라마바사"]), self.row("two", "라마바사", ["사아자"]),
            self.row("three", "사아자", ["차카타"]), self.row("four", "차카타", ["파하"]),
        ])
        first, second = self.root / "one", self.root / "two"
        first.mkdir(); second.mkdir()
        a = BATCHES.build(self.config(), self.root, first)
        b = BATCHES.build(self.config(), self.root, second)
        self.assertEqual(a["deterministic_digest"], b["deterministic_digest"])
        self.assertEqual((first / "batches.jsonl.gz").read_bytes(), (second / "batches.jsonl.gz").read_bytes())


if __name__ == "__main__":
    unittest.main()
