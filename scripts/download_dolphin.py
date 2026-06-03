#!/usr/bin/env python3
"""Pull the single-species (bottlenose dolphin) subset of the Watkins Marine
Mammal Sound Database from the Hugging Face parquet mirror.

Source: confit/wmms-parquet  (Watkins Marine Mammal Sound Database, WHOI +
New Bedford Whaling Museum). Free for personal/academic, non-commercial use.

We stream the dataset so we never materialize all 32 species at once, and we
keep only the target species. Each clip is written as a 16 kHz mono WAV plus a
row in manifest.csv (the unit of training/eval downstream).
"""
import argparse
import csv
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

def resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Cheap linear resample — fine for staging; swap for soxr if quality matters."""
    if not target_sr or sr == target_sr:
        return x, sr
    n = int(round(len(x) * target_sr / sr))
    y = np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x)
    return y.astype(np.float32), target_sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="Bottlenose_Dolphin")
    ap.add_argument("--out", default="data/raw/bottlenose_dolphin")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips of species")
    ap.add_argument(
        "--target-sr",
        type=int,
        default=0,
        help="0 = preserve native sample rate (correct for bioacoustics — dolphin "
        "content lives well above the 8 kHz Nyquist of a 16 kHz speech pipeline). "
        "Set e.g. 16000 only if feeding an off-the-shelf 16 kHz SSL model.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # streaming=True iterates parquet shards lazily instead of downloading 1.22 GB.
    # decode=False hands us the raw encoded bytes so we don't need torchcodec;
    # we decode with soundfile ourselves.
    ds = load_dataset("confit/wmms-parquet", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    manifest = out / "manifest.csv"
    n = 0
    total_secs = 0.0
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "species", "duration_s", "orig_sr"])
        for row in ds:
            if row["species"] != args.species:
                continue
            audio = row["audio"]  # {'bytes': <encoded>, 'path': ...} with decode=False
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if x.ndim > 1:  # downmix to mono
                x = x.mean(axis=1)
            x, out_sr = resample(x, sr, args.target_sr)
            clip = out / f"{args.species.lower()}_{n:05d}.wav"
            sf.write(clip, x, out_sr)
            dur = len(x) / out_sr
            total_secs += dur
            w.writerow([clip.name, args.species, f"{dur:.3f}", sr])
            n += 1
            if n % 10 == 0:
                print(f"  saved {n} clips ({total_secs/60:.1f} min)...", flush=True)
            if args.limit and n >= args.limit:
                break

    print(f"\nDone: {n} clips, {total_secs/60:.1f} min total -> {out}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
