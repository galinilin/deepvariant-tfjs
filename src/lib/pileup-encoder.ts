import type { Base, Cell } from './palette';
import type { Read } from './reads';
import { readSupportsCandidate, type Candidate } from './candidate';
import {
  BASE_INTENSITY,
  DIFFERS_FROM_REF_NO,
  DIFFERS_FROM_REF_YES,
  EMPTY_PIXEL,
  MAX_READ_ROWS,
  PILEUP_CHANNELS,
  PILEUP_HEIGHT,
  PILEUP_WIDTH,
  PREDICT_COL,
  REF_ROW_BASE_QUALITY,
  REF_ROW_DIFFERS,
  REF_ROW_INSERT_SIZE,
  REF_ROW_MAPQ,
  REF_ROW_STRAND,
  REF_ROW_SUPPORTS,
  REF_ROWS,
  STRAND_FORWARD,
  STRAND_REVERSE,
  SUPPORTS_VARIANT_NO,
  SUPPORTS_VARIANT_YES,
  encodeBaseQuality,
  encodeInsertSize,
  encodeMapq,
} from './dv-channels';

const SAMPLE_FLOATS = PILEUP_HEIGHT * PILEUP_WIDTH * PILEUP_CHANNELS;

/**
 * Encode a DeepVariant 1.8 WGS pileup tensor centered on `position`.
 *
 * Layout:
 *  - Row 0..4: REF rows (5 copies of the reference sequence in the window)
 *  - Row 5..(5+min(reads, 95)-1): read rows
 *  - Row (5+reads)..99: empty (zeros)
 *
 * Cols span [position - 110, position + 110]; out-of-window cells = 0.
 *
 * Returns a Float32Array of length 100*221*7 = 154,700, ready to feed
 * into `DeepVariantModel.predict()`. Returns null if `candidate` is null
 * (no candidate → no prediction makes sense).
 */
export function encodePileup(
  reads: Read[],
  reference: Base[],
  position: number,
  candidate: Candidate,
): Float32Array | null {
  if (!candidate) return null;

  const out = new Float32Array(SAMPLE_FLOATS);
  // Float32Array is zero-initialized, so empty cells already have 0 across all
  // channels — we only need to write the non-empty cells.

  const startCol = position - PREDICT_COL;
  const cellOffset = (row: number, col: number, ch: number): number =>
    row * PILEUP_WIDTH * PILEUP_CHANNELS + col * PILEUP_CHANNELS + ch;

  // Reference rows 0..REF_ROWS-1: identical, each cell carries the ref base
  // for its column with default ref-row channel values.
  for (let row = 0; row < REF_ROWS; row++) {
    for (let col = 0; col < PILEUP_WIDTH; col++) {
      const refCol = startCol + col;
      const refBase = reference[refCol];
      if (!refBase) continue; // out-of-reference columns stay 0
      const o = cellOffset(row, col, 0);
      out[o + 0] = baseIntensity(refBase);
      out[o + 1] = REF_ROW_BASE_QUALITY;
      out[o + 2] = REF_ROW_MAPQ;
      out[o + 3] = REF_ROW_STRAND;
      out[o + 4] = REF_ROW_SUPPORTS;
      out[o + 5] = REF_ROW_DIFFERS;
      out[o + 6] = REF_ROW_INSERT_SIZE;
    }
  }

  // Pick which reads land in the image. DV's image is centered on the
  // candidate; reads that don't overlap the [startCol, startCol+221) window
  // are skipped. Sort by row ascending to keep deterministic ordering, then
  // take up to MAX_READ_ROWS.
  const overlapping = reads
    .filter((r) => r.startCol < startCol + PILEUP_WIDTH && r.startCol + r.bases.length > startCol)
    .sort((a, b) => a.row - b.row || a.startCol - b.startCol)
    .slice(0, MAX_READ_ROWS);

  for (let idx = 0; idx < overlapping.length; idx++) {
    const read = overlapping[idx];
    const imageRow = REF_ROWS + idx;
    const supportsRow = readSupportsCandidate(read, position, candidate)
      ? SUPPORTS_VARIANT_YES
      : SUPPORTS_VARIANT_NO;
    const strandValue =
      read.strand === 'forward' ? STRAND_FORWARD : STRAND_REVERSE;
    const mapqValue = encodeMapq(read.mapq);
    const insertSizeValue = encodeInsertSize(read.insertSize);

    for (let col = 0; col < PILEUP_WIDTH; col++) {
      const refCol = startCol + col;
      const cellIdx = refCol - read.startCol;
      if (cellIdx < 0 || cellIdx >= read.bases.length) continue;
      const refBase = reference[refCol];
      if (!refBase) continue;
      const base = read.bases[cellIdx];
      const quality = read.qualities[cellIdx] ?? 0;
      const o = cellOffset(imageRow, col, 0);
      out[o + 0] = baseIntensity(base);
      out[o + 1] = encodeBaseQuality(quality);
      out[o + 2] = mapqValue;
      out[o + 3] = strandValue;
      out[o + 4] = supportsRow;
      out[o + 5] = base === refBase ? DIFFERS_FROM_REF_NO : DIFFERS_FROM_REF_YES;
      out[o + 6] = insertSizeValue;
    }
  }

  return out;
}

function baseIntensity(cell: Cell): number {
  return BASE_INTENSITY[cell] ?? EMPTY_PIXEL;
}

export interface EncodeStats {
  numReads: number;
  numRefRows: number;
  windowStart: number;
  windowEnd: number;
}

export function encodeStats(
  reads: Read[],
  position: number,
): EncodeStats {
  const startCol = position - PREDICT_COL;
  const overlapping = reads.filter(
    (r) => r.startCol < startCol + PILEUP_WIDTH && r.startCol + r.bases.length > startCol,
  );
  return {
    numReads: Math.min(overlapping.length, MAX_READ_ROWS),
    numRefRows: REF_ROWS,
    windowStart: startCol,
    windowEnd: startCol + PILEUP_WIDTH,
  };
}
