# Research code: batch construction, training, and evaluation pipelines

This directory contains the actual scripts that produced the numbers in the root
README, copied verbatim from the internal research repository at the frozen
release commit. They are published so that the claims in the README can be checked
against code, not only against a results table.

What is **not** here (same boundary as the rest of this repository): model weights,
raw or processed datasets, query IDs / qrels, per-query result rows, embeddings,
indexes, and checkpoints. Every script reads its inputs from paths given in a config
file; the configs shipped here are the frozen contracts used in the study and point
at data locations that are not distributed.

## Claim → code map

| README claim | Script | Tests |
|---|---|---|
| Collision-safe contrastive batches, length balancing, false-negative control | `scripts/build_contrastive_batches.py` | `tests/test_build_contrastive_batches.py` |
| Deterministic local fine-tuning with checkpoint identity and frozen batch plan | `scripts/train_contrastive.py` | `tests/test_train_contrastive.py` (103) |
| nDCG@10 / MRR / Recall, paired bootstrap CI, exact McNemar, locked test splits, subgroup stratification | `scripts/evaluate_retrieval.py` | `tests/test_evaluate_retrieval.py` (161) |
| Multi-candidate comparison with Holm correction and acceptance gates | `scripts/compare_contrastive_candidates.py` | `tests/test_compare_contrastive_candidates.py` |

Total: 4 scripts (~4,300 lines), 4 test modules, 308 unit tests. All tests run on
synthetic in-memory fixtures and temporary directories; none requires the study data,
a GPU, or network access.

The DAPT, reranker-distillation, and hybrid-retrieval ablations reported as negative
results in the root README were run with separate scripts that are not included here;
their aggregate outcomes are in `results/` and the root README tables.

## Reading the stage identifiers

File names, configs and docstrings carry identifiers such as `task08a` or
`TASK-08B-R1`. These are the internal experiment-stage IDs of the research plan and are
kept unchanged for provenance (they appear in output manifests and are asserted by
the tests). Rough legend:

| ID | Stage |
|---|---|
| TASK-04 / 04A | contrastive batch plan (frozen, digest-verified) |
| TASK-05 / 06 | zero-shot baselines, BM25, evaluation harness |
| TASK-07 / 08 | contrastive fine-tuning of the four candidate encoders and their comparison |
| TASK-11 | code-switching (MixedDrop) controls |
| TASK-12 / 13 | DAPT, reranker distillation, hybrid and source ablations (scripts not included) |

## Layout

```text
research/
  scripts/   batch construction, training, evaluation and comparison entry points (argparse + JSON config)
  configs/   frozen config contracts referenced by the scripts and tests
  tests/     unit tests (standard library unittest)
```

Tests locate the scripts relative to this directory, so run them from here:

```bash
cd research
python -m pip install -r requirements-test.txt --extra-index-url https://download.pytorch.org/whl/cpu   # numpy, CPU torch, datasets, safetensors
python -m unittest discover -s tests -p "test_*.py"
```

The `torch`/`datasets` dependent tests skip or error when those packages are absent;
everything else runs on the standard library.

## Running the pipelines

Each script is a CLI that takes `--config <json>`. The shipped configs document the
exact hyper-parameters, seeds, model revisions, and split locks used in the study, but
their `data/…` and `results/…` paths refer to non-distributed assets. To run a
pipeline end to end you must supply your own corpus, queries, and qrels in the formats
described in the config `_note` fields and the script docstrings.
