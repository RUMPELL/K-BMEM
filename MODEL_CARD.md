# Model card: K-BMEM (portfolio record)

## Summary

K-BMEM is a Korean–English medical embedding research prototype based on KURE-v1 and contrastive fine-tuning. This repository records the engineering approach and aggregate evaluation only.

## Release status

Weights are **not included**. The final experimental checkpoint used a training component whose redistribution and derivative-weight implications have not been cleared. This card does not grant access to or a license for those weights.

## Intended use

- Research on local medical semantic retrieval
- Reproducible comparison of embedding approaches
- Portfolio demonstration with synthetic or separately authorized content

## Out-of-scope use

- Diagnosis, triage, treatment recommendations, or autonomous clinical decisions
- Patient-facing medical advice
- Claims of real-world RAG quality based only on the included pair-matching benchmark

## Evaluation

The aggregate table in `results/benchmark_summary.json` compares six frozen embedding systems. Exam is an MCQ option-ranking proxy. AIHub strict passage is auxiliary QA-pair matching. Both splits had prior evaluation exposure. No row-level evidence is distributed.

## Known limitations

- Exam accuracy has wide uncertainty at n=266.
- AIHub strict passage has a small paired corpus and is not open-corpus retrieval.
- The training and evaluation assets are primarily Korean medical QA/MCQ data, not clinical records.
- Performance does not establish safety, fairness, or clinical validity.
