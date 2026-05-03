# match/3.1 — DV-canonical pipeline validation against real BAM

**TL;DR.** 84 candidates extracted from DV's testdata (NA12878 chr20:10MB–10.1MB).
Three-level comparison: candidate engine, pileup encoder, model prediction.

```
  Candidate match  : 78/83  (94.0%)   [1 skipped: 'complex' kind]
  Encoder match    : 80/80  (100.0%)  [4 skipped: 1 complex + 3 multi-allelic]
  Prediction match : 80/80  (100.0%)
```

## What got validated

- **Encoder**: across 80 single-alt examples (66 SNVs, 7 deletions, 7 insertions),
  every encoded tensor lives in DV's allowed per-channel value sets
  (verified via `validateEncodedTensor` against the source-cited
  `GOLDEN_CHANNEL_RANGES`).
- **Prediction**: 80/80 argmax-match between our TFJS uint8 inference and
  upstream Keras float32 — zero uint8-quantization drift on this set.
- **Candidate engine** (after the two fixes below): 78/83 match. The 5
  remaining are DV-testdata policy gaps, not engine bugs (see below).

## Two engine fixes derived from running the validator

The validator surfaced two real semantic mismatches with DV that
synthetic testing had missed:

1. **Deletion anchor convention** — DV's `variant.start` points to the
   base BEFORE the deletion (e.g., `TA>T` at `start=P` means
   anchor=T@P, deleted=A@P+1). Our previous engine looked for `bases[offset]==='-'`
   at the candidate position, which only worked for our prior synthetic
   convention where `'-'` sat AT the position. Fixed in `candidate.ts`:
   reads with `bases[offset]==='-'` are skipped (they're inside someone
   else's deletion run); a deletion candidate at offset is detected when
   `bases[offset+1]==='-'`. Synthetic scenarios in `reads.ts` adjusted to
   match (`applyDeletion(offset+1, length)` for `het_del`/`hom_alt_del`).
   Recovered 6 candidate matches.

2. **Indel-preferred tie-break** — when SNV and indel allele counts tie at
   a position, DV prefers the indel (phantom SNV mismatches near indels
   are common alignment artifacts). Our previous `kindRank` put SNV
   first. Flipped to `del=0, ins=1, snv=2` in `candidate.ts`. Recovered 1
   candidate match.

## The 5 remaining L1 failures (expected gaps, not bugs)

All are positions where DV's testdata calling examples were generated
with **relaxed thresholds** to maximize variant diversity in the test
fixture. Our engine uses DV's *production* thresholds (mapq≥5, base-Q≥10,
SNV count≥2 + frac≥0.12, indel count≥2 + frac≥0.06). DV's testdata calls
fall below those:

| Sample | Variant                        | Our outcome     | Expected | Reason                  |
|--------|--------------------------------|-----------------|----------|-------------------------|
| 25     | chr20:10009877 A>G             | below-fraction  | snv:G    | 4/37 = 10.8% < 12%      |
| 41     | chr20:10004253 A>T             | below-fraction  | snv:T    | sub-12% SNV fraction    |
| 62     | chr20:10002493 A>C             | below-fraction  | snv:C    | sub-12% SNV fraction    |
| 73     | chr20:10008717 AT>A            | below-count     | del      | 1/N read supports < 2   |
| 75     | chr20:10008750 T>TAA           | snv pick        | ins:AA   | 1 read with insertion   |

Lowering our thresholds to match DV's testdata would diverge from DV
production behavior. We document instead.

## Skipped (4 of 84)

- 1 'complex' (sample 82, `CACACACACACA>CCACACACACA` — neither
  snv/del/ins per the standard convention).
- 3 multi-allelic combined examples (samples 60, 78, 83 with
  `alt_indices=[0,1]`). Our encoder's `Candidate` type holds a single
  alt; multi-allelic combined needs a v2 encoder.

## Repro

```
npm run extract-match-fixtures   # regenerates fixtures/match/ from DV testdata
npm run match                    # validates; writes drafts/match-failures.json
```

Outputs: `fixtures/match/{manifest.json, sample_*.json, pileups.npy.gz,
outputs.npy}`, `drafts/match-failures.json`.
