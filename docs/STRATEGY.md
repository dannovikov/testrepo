# Unsupervised audio → text "translation" — strategy

Goal: take an **unlabeled** corpus of one species' vocalizations (here: bottlenose
dolphin) and learn to decode it into human text, with **no parallel data** — no
(audio, gloss) pairs, no transcripts, no dictionary.

This is the animal-communication framing of unsupervised speech translation. The
same recipe as Project CETI / Earth Species Project.

## The pipeline

1. **Self-supervised audio pretraining.** Train a wav2vec2/HuBERT/BEST-RQ-style
   encoder on the raw, unlabeled audio. This buys representations whose geometry
   reflects the acoustic/combinatorial structure of the signal, with zero labels.

2. **Discretization → "pseudo-phonemes".** Quantize/cluster the encoder features
   and segment into a discrete unit sequence. The corpus becomes strings over a
   learned symbol inventory — the analogue of phonemes. (This is where repertoire
   structure — dolphin whistle/click/burst-pulse types — should emerge.)

3. **Unsupervised mapping to text.** Learn units → target-language text with no
   parallel data, via adversarial distribution matching + iterative
   back-translation against a strong target-language LM prior (the wav2vec-U
   recipe, lifted cross-domain).

## The load-bearing assumption (read this before trusting any output)

Unsupervised translation does **not** work by matching surface distributions of
audio and text. It works only when the two latent spaces are approximately
**isometric** — the relational geometry of the source content and the target
language are similar enough that a generator/discriminator game can find the
alignment. Within one human language's speech-vs-text that roughly holds. Across
a real modality gap *and* a non-human source, it is an open empirical question.

The failure mode is **not** "no output." A strong LM prior makes the decoder emit
fluent, well-formed sentences regardless. With no grounding you cannot
distinguish a faithful decode from a confident hallucination — and that is the
entire problem in animal communication.

## Symmetry-breakers / validation anchors

You need *something* to break the alignment symmetry and to validate against.
For dolphins specifically:

- **Signature whistles** — individually distinctive, function like names. A small
  set of known referents you can anchor and check.
- **Context-tagged calls** — vocalizations with known behavioral context
  (foraging, distress, contact) give weak (audio, meaning) supervision.
- **Comparable structure** — Zipfian unit-frequency statistics, sequential
  dependencies (n-gram / entropy-rate analysis) test whether there is
  language-like structure to decode *at all* before claiming to decode it.

Treat 1–2 as held-out validation, never as training signal, or you'll fool
yourself.

## Data

- Source: Watkins Marine Mammal Sound Database (WHOI + New Bedford Whaling Museum),
  via the `confit/wmms-parquet` Hugging Face mirror. Free for personal/academic,
  non-commercial use.
- Species staged: `Bottlenose_Dolphin`. Pull with `scripts/download_dolphin.py`.
- **Bandwidth note:** native sample rates are ~60–82 kHz. Do not blindly
  downsample to the 16 kHz that human-speech SSL models assume — dolphin content
  lives well above an 8 kHz Nyquist. The downloader preserves native SR by default.
