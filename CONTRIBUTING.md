# Contributing

Small, reviewable improvements to the generic retrieval code and documentation are welcome.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-deps -e .
python -m unittest discover -s tests -p "test_*.py" -v
```

## Data and safety boundary

Do not contribute patient data, proprietary or access-controlled medical text, API credentials, model checkpoints, embedding caches, or row-level benchmark evidence. Tests should use synthetic inputs.

Performance claims must identify the dataset, split, sample count, metric, uncertainty method, and limitations. Negative results must not be omitted when they change the interpretation.
