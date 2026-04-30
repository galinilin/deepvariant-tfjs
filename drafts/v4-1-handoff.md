# v4.1 handoff — read this when chat history was rolled back

**Today: 2026-04-30. You are on `feature/v4.1`, freshly branched off `feature/v3.0`.**
The previous attempt at v4 (`feature/v4.0`) was abandoned because the live UI
integration was buggy and synthetic-data predictions didn't make sense. The
encoder + parity infrastructure was validated, though, so v4.1 carries those
forward; the bottom canvas is back to v3.0's placeholder for a clean redesign.

If you are reading this with no recollection of how v4.0 ended, that's
expected — the user explicitly rolled back chat history past that work and
left these notes as the durable record.

## What's on this branch (cherry-picked from v4.0, validated)

```
src/lib/dv-channels.ts         encoding constants — confirmed against golden
src/lib/pileup-encoder.ts      encodePileup(reads, ref, position, candidate) → Float32Array
                               + GOLDEN_CHANNEL_RANGES + validateEncodedTensor()
src/lib/parity.ts              verifyGoldenParity, inspectGoldenChannels,
                               compareEncoderAgainstGolden — all functional
public/fixtures/               5 real DV pileups + their source reads + Keras
                               predictions; pre-existing synthetic parity fixtures
drafts/pileup-encoder-audit.md the encoder findings doc
.gitignore                     modified to allow public/fixtures/ (was blocked)
```

The bottom canvas (`src/sketch/bottom.ts`) is **back to its v3.0 placeholder**.
The top sketch (`src/sketch/top.ts`) does NOT integrate the encoder yet.
`src/lib/sandbox-state.ts` has only `{ candidate }` — no `pileupTensor`,
`prediction`, etc.

The `src/lib/DeepVariantModel.ts` and `src/lib/npy.ts` files were
pre-existing scaffolding — they survive on every branch.

## What was validated on v4.0 (do NOT re-derive these)

1. **Channel order**: `[read_base, base_quality, mapping_quality, strand,
   supports_variant, differs_from_ref, insert_size]`. Channel-7 = insert_size
   = DV channel ID 19 (per `dv-tfjs/testdata/example_info.json`'s
   `channels: [1,2,3,4,5,6,19]`).

2. **Encoding constants** (all verified against golden tensors):
   - `read_base`: A=250, C=30, G=180, T=100, deletion=5 (best-known, golden
     had no deletions to verify), N=0
   - `base_quality`: `floor(q × 254 / 40)`, capped at 254. **NOT round.**
   - `mapping_quality`: `floor(mq × 254 / 60)`, capped at 254
   - `strand`: forward=70, reverse=240
   - `supports_variant`: **per-row** (broadcast to all cells of the read row).
     254 if read carries the candidate alt, 152 if not, 0 if empty
   - `differs_from_ref`: **per-cell**. 254 if differs, **50 if matches** (NOT 0,
     0 is reserved for empty cells), 0 if empty
   - `insert_size`: `floor(|tlen| × 254 / 1000)`, capped at 254

3. **Layout**: 5 reference rows at top (rows 0..4), then read rows (5..min(99,
   5+N)), zero-padded below. Each ref-row cell:
   `(refBase, 254, 254, 70, 152, 50, 254)`.

4. **Validation result**: 5/5 argmax matches against upstream Keras on real
   golden examples. Per-channel value-set diffs collapse to read-selection
   variance (which 95-of-180 reads to pick); no encoding bugs.

5. **Empirical golden round-trip is wired up**: call
   `compareEncoderAgainstGolden({ sampleIndex: 0..4 })` and you'll see argmax
   + per-channel diff against the golden tensor. The 5 hom_alt SNVs are at
   chr20:10000117..10000694 (NA12878).

## Why v4.0's UI integration was abandoned

1. **Predictions on synthetic sandbox data are always hom_alt.** Hypothesis:
   our 221-bp window can pack 3–5 EXTRA_POOL scenarios from
   `placeScenarios`, so the encoded tensor has stacked variant evidence the
   model was never trained on (DV's training data is high-coverage with
   one candidate per pileup). The model sees lots of `differs_from_ref=254`
   cells from neighboring scenarios and biases toward hom_alt regardless of
   which column is "the candidate." Verified empirically on every Randomize.

2. **The bottom canvas live integration was state-heavy and buggy.**
   `sandboxState` accumulated `pileupTensor`, `pileupPosition`,
   `readsGeneration`, `prediction`, `predicting`, `candidateForced`,
   `forcePredict`, `debugLogs`. Bottom.ts had module-globals for
   `modelInstance`, `modelPromise`, `modelError`, `lastPredictedPosition`,
   `lastPredictedGen`, `predictDebounceTimer`, `pendingTarget`. The
   prediction-debounce timer was reset every p.draw frame so it never
   elapsed (caught at the very end). Even after that fix, the live UX
   was busy and the pileup-image strip rendering never looked right (52:1
   aspect ratio of 1571×30 channel image squeezed into a ~300-px-tall
   panel, every channel a ~25-px-tall band).

3. **Coverage mismatch**: our sandbox has ~14 packed reads → ~14 read rows
   in the 100-row tensor → 81% of rows zero-padded. DV trained on near-fully-
   filled tensors. The model's behavior in this sparse regime is unreliable.

## Two cleaner paths to consider for v4.1

### Path α — Prediction tied to golden samples, not synthetic

Synthetic sandbox stays a read/candidate exploration tool only (the v3.0
state). For the model side, build a separate "Golden viewer" panel:
load one of the 5 real DV pileups from `public/fixtures/golden_pileups.npy`,
render its 7 channel strips, show its upstream prediction
(`golden_outputs.npy`) plus our model's prediction. The user can step
through the 5 samples with prev/next buttons, and see exactly what DV's
input + output looks like on real data.

**Pro**: predictions actually correct because input is real DV data. The
sandbox stays simple. Educational about what DV sees.
**Con**: doesn't connect synthetic exploration to model behavior — they're
separate experiences.

### Path β — Pre-bake DV-shaped synthetic scenarios

Add a "high-coverage mode" that generates ~95 reads in a tight window
around a single, isolated variant (no other scenarios in the 221-bp
window). Predictions on those should match expected class. The current
EXTRA_POOL stays for sandbox exploration but is opt-out-of-prediction.

**Pro**: keeps the sandbox→prediction connection. User can build their
own variant and see DV's call.
**Con**: more work — need scenario placement that respects a "lonely
single variant" constraint, plus a coverage knob (95+ reads), plus all
the UI integration to surface this.

Recommend starting with Path α since it's much simpler and gives an
honest picture; Path β can layer on later.

## Things that are not on this branch but might be useful

These exist on `feature/v4.0` and can be cherry-picked if needed:

- `src/sketch/bottom.ts` (full live channel-strip rendering + prediction
  triggering) — only mine if redesigning Path β
- Corner buttons in `src/main.ts` + `index.html` for `verify-golden`,
  `inspect-golden`, `encoder-vs-golden` — useful diagnostics; consider
  re-adding once v4.1 has a basic UI
- `deriveAnyAltCandidate` in v4.0's `src/lib/candidate.ts` — fabricates a
  candidate at any column for force-predict; only useful for the abandoned
  debug toggle

## Sibling-repo state (unaffected by this revert)

`/root/dv-tfjs/` has the Python infrastructure, untouched by anything we
do here. Notable scripts:

- `scripts/extract_reads_for_golden.py` — parses variant proto from the
  golden calling examples, extracts reads from
  `testdata/input/NA12878_S1.chr20.10_10p1mb.bam` via pysam, dumps as
  `golden_reads_<i>.json` in our Read[] schema. Re-run if you need to
  refresh fixtures: `cd /root/dv-tfjs && uv run python
  scripts/extract_reads_for_golden.py
  /root/deepvariant-tf-js/public/fixtures/ 5`
- `scripts/extract_golden.py` — earlier script, dumps `golden_pileups.npy`
  + `golden_outputs.npy` from the same testdata
- `scripts/run_real_pileups.py` — DV's reference for channel ablations
- `vendor/dv_model.py` — the InceptionV3 architecture
- `.venv` — Python 3.11 + tensorflow 2.15 + pysam, all set up

DV r1.8 source is at `/tmp/dv-source/` (cloned for the BAM/FASTA in
`testdata/input/`). Not gitignored; can be deleted and re-cloned.

## The exact selective-revert steps that produced this branch

```bash
# Mark v4.0 as abandoned (committed on v4.0)
git checkout feature/v4.0
git commit -am "docs: mark v4.0 as abandoned in favor of v4.1"

# Branch v4.1 fresh off v3.0
git checkout feature/v3.0
git checkout -b feature/v4.1

# Selectively pull validated files from v4.0
git checkout feature/v4.0 -- \
  src/lib/dv-channels.ts \
  src/lib/pileup-encoder.ts \
  src/lib/parity.ts \
  drafts/pileup-encoder-audit.md \
  .gitignore \
  public/fixtures/golden_pileups.npy \
  public/fixtures/golden_outputs.npy \
  public/fixtures/golden_labels.json \
  public/fixtures/golden_reads_0.json \
  public/fixtures/golden_reads_1.json \
  public/fixtures/golden_reads_2.json \
  public/fixtures/golden_reads_3.json \
  public/fixtures/golden_reads_4.json \
  public/fixtures/parity_inputs.npy \
  public/fixtures/parity_outputs.npy

npm run typecheck   # verify
git add -A
git commit -m "v4.1 clean restart off v3.0 with validated data layer"
```

## Build/run

```
cd /root/deepvariant-tf-js
npm run typecheck  # passes
npm run dev        # vite serves on :5173 (or :5174)
```

The data layer compiles and is callable from anywhere in the codebase.
There's no UI wiring it to anything live yet — that's what v4.1 needs to
design fresh.

---

**One-line summary for the next agent:** `feature/v4.1` carries the
validated DV encoder + golden fixtures forward; the live UI integration
is yours to design from scratch, ideally Path α (golden viewer) since
synthetic-sandbox predictions don't match the model's training
distribution.
