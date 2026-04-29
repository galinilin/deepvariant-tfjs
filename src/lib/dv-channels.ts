import type { Cell } from './palette';

/**
 * DV 1.8 WGS pileup-image encoding constants. Reverse-engineered by
 * inspecting upstream `golden_pileups.npy` (extracted via
 * `dv-tfjs/scripts/extract_golden.py` from the official
 * `golden.calling_examples.tfrecord`). Channel order matches
 * `testdata/example_info.json`'s `channels: [1,2,3,4,5,6,19]`.
 *
 * IMPORTANT semantic notes (not what the channel names suggest):
 *  - `supports_variant` is **per-read** (broadcast to every column of
 *    the read's row): 254 if the read has the candidate alt at the
 *    candidate column, 152 if it doesn't, 0 if empty.
 *  - `differs_from_ref` is **per-cell**: 254 if base ≠ ref at this
 *    column, 50 if matches, 0 if empty.
 *  - Both channels reserve 0 for empty cells, so the "negative" case
 *    (152 / 50) is non-zero — encoding empty as 0 stays distinguishable.
 */

export const PILEUP_HEIGHT = 100;
export const PILEUP_WIDTH = 221;
export const PILEUP_CHANNELS = 7;
export const REF_ROWS = 5;
export const MAX_READ_ROWS = PILEUP_HEIGHT - REF_ROWS;
export const PREDICT_COL = Math.floor(PILEUP_WIDTH / 2); // 110

export const EMPTY_PIXEL = 0;

// Channel 0 — read_base
export const BASE_INTENSITY: Record<Cell, number> = {
  A: 250,
  C: 30,
  G: 180,
  T: 100,
  // Deletion and N values weren't present in the 5 golden hom_alt SNV
  // examples. Best-known DV constants:
  '-': 5,
  N: 0,
};

// Channel 1 — base_quality (Phred capped at 40, scaled to [0, 254])
export const BASE_QUALITY_CAP = 40;
export function encodeBaseQuality(q: number): number {
  const clamped = Math.max(0, Math.min(BASE_QUALITY_CAP, q));
  return Math.round((clamped * 254) / BASE_QUALITY_CAP);
}

// Channel 2 — mapping_quality (mapq capped at 60, scaled to [0, 254])
export const MAPQ_CAP = 60;
export function encodeMapq(mq: number): number {
  const clamped = Math.max(0, Math.min(MAPQ_CAP, mq));
  return Math.round((clamped * 254) / MAPQ_CAP);
}

// Channel 3 — strand
export const STRAND_FORWARD = 70;
export const STRAND_REVERSE = 240;

// Channel 4 — supports_variant (per-row)
export const SUPPORTS_VARIANT_YES = 254;
export const SUPPORTS_VARIANT_NO = 152;

// Channel 5 — differs_from_ref (per-cell)
export const DIFFERS_FROM_REF_YES = 254;
export const DIFFERS_FROM_REF_NO = 50;

// Channel 6 — insert_size: |template_length| scaled, capped at 1000bp
export const INSERT_SIZE_CAP = 1000;
export function encodeInsertSize(tlen: number): number {
  const clamped = Math.min(INSERT_SIZE_CAP, Math.abs(tlen));
  return Math.round((clamped * 254) / INSERT_SIZE_CAP);
}

// Reference-row defaults. From inspecting golden samples, every cell of a
// reference row has these channel values (with read_base = the ref base for
// that column).
export const REF_ROW_BASE_QUALITY = 254;
export const REF_ROW_MAPQ = 254;
export const REF_ROW_STRAND = STRAND_FORWARD;
export const REF_ROW_SUPPORTS = SUPPORTS_VARIANT_NO;
export const REF_ROW_DIFFERS = DIFFERS_FROM_REF_NO;
export const REF_ROW_INSERT_SIZE = 254;
