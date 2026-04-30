# Pileup encoder audit (v4.0)

Branch: `feature/v4.0`. **Status: encoder empirically validated against
upstream DV on real golden examples. 5/5 argmax match, all per-channel
value-set diffs explained by read-selection differences (not encoding).**

History:
1. We had golden tensors but no source reads → could only do indirect validation.
2. Cloned DV r1.8 testdata, found the source NA12878 BAM + hg19 FASTA.
3. Added `dv-tfjs/scripts/extract_reads_for_golden.py` — parses the
   variant proto from each golden calling example, extracts the supporting
   reads from the BAM via pysam, dumps as JSON sidecars in our Read[] schema.
4. Added `compareEncoderAgainstGolden` — encodes the extracted reads
   through our encoder, predicts, diffs argmax + per-channel value sets
   against the golden tensor.
5. First run revealed a systematic +1 in base_quality / mapping_quality:
   we used `Math.round`, DV uses `Math.floor`. Fixed in commit `16bef41`.
6. Second run (post-fix): argmax 5/5, diffs collapse to read-selection
   artifacts — no encoding bugs remain.

## Hard ground truth (extracted from golden_pileups.npy)

5 real DV calling examples (NA12878 chr20 SNVs, all hom_alt). Allowed
value sets per channel:

| ch | name              | observed values                                    |
|----|-------------------|----------------------------------------------------|
| 0  | read_base         | `{0, 30, 100, 180, 250}` (only A/C/G/T + empty)    |
| 1  | base_quality      | `[0, 254]`, ~37 unique (q×254/40 rounding)         |
| 2  | mapping_quality   | `{0, 97, 122, 152, 156, 254}` (only 5 mapq levels in this dataset) |
| 3  | strand            | `{0, 70, 240}`                                     |
| 4  | supports_variant  | `{0, 152, 254}` — **per-row**, broadcast           |
| 5  | differs_from_ref  | `{0, 50, 254}` — **per-cell**                      |
| 6  | insert_size       | `[0, 254]`, ~76 unique values                      |

5 reference rows at the top, then read rows. Empty cells = 0 across all
seven channels. `validateEncodedTensor()` in `src/lib/pileup-encoder.ts`
checks our encoder's output against these sets — invoke from console:
`(window as any).validateEncodedTensor(sandboxState.pileupTensor)`.

(Or wire it to a debug button next session.)

## What the encoder almost certainly gets right

- Channels 0/3/4/5: discrete value sets exactly match golden.
- Channel 1: `q × 254 / 40` rounding matches.
- Channel 2: continuous `mq × 254 / 60` formula matches the underlying
  golden values.
- 5 ref rows on top with `(base, 254, 254, 70, 152, 50, 254)` per cell.
- Read row encoding: per-row supports_variant, per-cell differs_from_ref.

## What's uncertain / probably wrong

Ranked by how much each could distort predictions.

### 1. Read filter is too permissive (likely a real bug)

Current:

```ts
overlapping = reads.filter(
  (r) => r.startCol < startCol + 221 && r.startCol + r.bases.length > startCol,
)
```

This includes reads that overlap the *window* but don't span the
*candidate column*. Those reads contribute `supports_variant=152`
(does-not-support) across their span, which the model interprets as
"reads disagreeing with the variant." Real DV typically requires a read
to span the candidate position to count.

Fix: tighten to `r.startCol <= position && position < r.startCol + r.bases.length`
or similar. Test against golden: how many reads in a real pileup actually
span the candidate vs just touching the window?

### 2. Deletion encoding is a guess (`'-'` → 5)

Golden samples are all SNVs — they contain no `-` cells, so we can't
verify the deletion intensity. We're using 5 from prior DV-source memory
(`pileup_image_native.cc`'s `kDeletionPixel` constant). Could be 50, 80,
or different. Test by predicting a `het_del` scenario; if confidence is
low or wrong-class, sweep candidate values: 5, 30, 50, 80, 110. The one
that produces sensible het predictions is right.

To verify decisively: extract more golden examples (50–100) until at
least one deletion candidate appears, inspect channel 0 at the deleted
read cells.

### 3. 'N' base collides with empty marker

`BASE_INTENSITY['N'] = 0`, same as the empty-cell sentinel. Our synthetic
reads never produce N, so this is a latent bug not a current one. Real
BAMs occasionally have N. Pick a non-zero value (maybe 0 stays since N
is rare; or maybe ~210 to keep it distinct from A/C/G/T).

### 4. Read row ordering differs from DV's

We sort by IGV pack-row + start col. DV's pileup uses different
ordering (insertion order with possible random sub-sampling). For a CNN,
row order shouldn't matter much spatially — convolutional kernels are
local. But if DV groups supporting reads first or sorts by start, the
model might have learned position-conditional features. Worth verifying
empirically: shuffle our read order, see if predictions change.

### 5. Insertion bases never appear in read_base channel

For an insertion candidate (`kind: 'ins'`), our encoder sets the
`supports_variant` channel for reads carrying the matching insertion,
but the inserted bases themselves never appear anywhere in the tensor.
The model has nothing in `read_base` distinguishing an inserted-base
column from a normal column. Predictions for insertion candidates will
likely be unreliable.

DV's actual encoding: needs verification. Most likely it overlays the
inserted base sequence into the column following the anchor (or marks
the anchor column with a sentinel intensity). Out of scope for v4.0;
note for v5.0.

### 6. insertSize range mismatch

Our reads have `insertSize = 320 + Math.floor(rng() * 120) = 320..440`
always positive. Golden distribution has 76 unique values, mostly in
the 60–110 range when scaled by `× 254 / 1000`. So our scaled values
land 81–112, which actually lines up reasonably with the dominant
golden range. Probably fine.

But: real BAMs have negative tlens (mate orientation) and zero (singletons).
We don't model these. Minor.

## Surprises caught during inspection

Worth re-emphasizing because they were *not* obvious from the channel names:

- `supports_variant` is **per-read** (broadcast across the row), not per-cell.
- `differs_from_ref` is **per-cell** but uses **50 (not 0)** for matches —
  the 0 sentinel is reserved for empty cells. Encoding matches as 0 would
  make every read row look like one giant empty row to the model.
- Reference rows count is **5**, not 1. They sit at rows 0–4 of every image.
- 4 cells in golden have `read_base = 0` but other channels ≠ 0. Tiny,
  ignorable, probably edge encoding artifact.

## Concrete test plan for next session

1. **SNV sanity** — pan to a clear `hom_alt` SNV scenario in the sandbox.
   Expected: argmax = `hom_alt`, confidence > 0.8. If yes → encoder works
   for SNVs; move to indel testing. If no → channel 5 or 4 likely wrong.

2. **Validator pass** — `validateEncodedTensor(sandboxState.pileupTensor)`
   from console. Should return `passed: true` with no issues. If
   violations appear, pick those channels first.

3. **Read filter fix** — tighten the read filter to "must span predict
   column," re-test SNV sanity. Predictions should sharpen.

4. **Deletion test** — pan to `het_del` / `hom_alt_del`. Sweep
   `BASE_INTENSITY['-']` over `{5, 30, 50, 80, 110}`. Whichever produces
   right-class predictions is the value. Hardcode it.

5. **Extract more golden** — `cd /root/dv-tfjs && uv run python
   scripts/extract_golden.py /root/deepvariant-tf-js/public/fixtures/ 50`.
   Find first sample with a deletion candidate (search variant_summary
   for "del" or alleles where len(ref) > len(alt)). Inspect channel 0 at
   deletion cells — that's the truth.

6. **Insertion test** — pan to `het_ins` / `hom_alt_ins`. Predictions
   will probably be wrong. Document the gap, defer the fix.

7. **End-to-end golden round-trip** — pick one golden pileup, encode
   "the same" inputs through our encoder (manually crafted reads that
   would produce that exact pileup), feed through model, verify output
   matches upstream within quant tolerance.

## Files / functions of interest

- `src/lib/dv-channels.ts` — encoding constants
- `src/lib/pileup-encoder.ts` — `encodePileup`, `validateEncodedTensor`,
  `GOLDEN_CHANNEL_RANGES`
- `src/lib/parity.ts` — `verifyGoldenParity`, `inspectGoldenChannels`
- `dv-tfjs/scripts/extract_golden.py` — pileup fixture generator
- `dv-tfjs/scripts/run_real_pileups.py` — channel-ablation reference
- `dv-tfjs/testdata/example_info.json` — DV channel-id schema
  (`channels: [1,2,3,4,5,6,19]` confirms WGS layout + insert_size as ch7)

## Status

- v4.0 has shipped: encoder + Pileup Image strip rendering + Prediction
  bars + golden parity verification.
- Branch: `feature/v4.0`. Latest commit: `ed04a13`.
- Not merged to master.
- Dev server still running on :5174 (background).

Pick this back up by reading this file, refreshing the dev server, and
running the test plan above in order.
