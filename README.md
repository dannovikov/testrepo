# Decoding animal vocalizations → text (unsupervised)

An exploration of unsupervised audio→text "translation" applied to animal
communication: pretrain on an **unlabeled** single-species audio corpus, then
decode into human text with **no parallel data**.

See [`docs/STRATEGY.md`](docs/STRATEGY.md) for the approach, its load-bearing
isometry assumption, and the validation anchors that keep you from mistaking a
fluent hallucination for a decode.

## Data

Staged species: bottlenose dolphin, from the Watkins Marine Mammal Sound Database
(`confit/wmms-parquet` HF mirror; free for academic, non-commercial use).

```bash
pip install datasets soundfile huggingface_hub
python scripts/download_dolphin.py            # native sample rate (default)
python scripts/download_dolphin.py --species Spinner_Dolphin --limit 50
```

Writes WAVs + `manifest.csv` to `data/raw/<species>/` (gitignored).

> **Scale caveat:** this Watkins "best-of" mirror is a *classification* benchmark
> — at most ~91 clips of ~1–2 s per species. It validates the pipeline plumbing
> but is far too small to actually pretrain on. A real run needs a
> pretraining-scale corpus (continuous hydrophone archives, Watkins all-cuts,
> Project CETI, DCLDE). See STRATEGY.md.
