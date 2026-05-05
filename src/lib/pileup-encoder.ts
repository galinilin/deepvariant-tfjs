import type { Base } from './palette';
import type { Read } from './reads';
import { readSupportsCandidate, type Candidate } from './candidate';
import {
  BASE_INTENSITY,
  DIFFERS_FROM_REF_NO,
  DIFFERS_FROM_REF_YES,
  IMAGE_MIN_BASE_QUALITY,
  IMAGE_MIN_MAPPING_QUALITY,
  INDEL_ANCHOR_INTENSITY,
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

export interface EncodeResult {
  tensor: Float32Array;
  /** Per-image-row mapping from the encoder layout (rows 0..99) to the
   * read.id that occupies that row, or null for ref-rows (0..4) and
   * empty padding rows. Length is always 100. Used by the bottom canvas
   * to translate pixel hover (row index) → top canvas read highlight. */
  rowToReadId: (string | null)[];
}

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
 * Returns { tensor, rowToReadId }. Tensor is length 100*221*7 = 154,700,
 * ready to feed into `DeepVariantModel.predict()` (after `(x-128)/128`
 * normalization on the consumer side). Returns null if `candidate` is
 * null (no candidate → no prediction makes sense).
 */
export function encodePileup(
  reads: Read[],
  reference: Base[],
  position: number,
  candidate: Candidate,
): EncodeResult | null {
  if (!candidate) return null;

  const out = new Float32Array(SAMPLE_FLOATS);
  const rowToReadId: (string | null)[] = new Array(PILEUP_HEIGHT).fill(null);
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

  // DV applies image-level read filters BEFORE encoding (pileup_image_native.cc:282-288
  // and pileup_channel_lib.cc:147-149):
  //   1) mapq >= IMAGE_MIN_MAPPING_QUALITY (default 10)
  //   2) if the read covers `position` AND the base there is not a deletion,
  //      the base quality at that column must be >= IMAGE_MIN_BASE_QUALITY.
  // Reads that don't cover `position` pass the base-Q gate trivially.
  const overlapping = reads
    .filter((r) => r.startCol < startCol + PILEUP_WIDTH && r.startCol + r.bases.length > startCol)
    .filter((r) => passesImageFilter(r, position))
    .sort((a, b) => a.row - b.row || a.startCol - b.startCol)
    .slice(0, MAX_READ_ROWS);

  for (let idx = 0; idx < overlapping.length; idx++) {
    const read = overlapping[idx];
    const imageRow = REF_ROWS + idx;
    rowToReadId[imageRow] = read.id;
    const supportsRow = readSupportsCandidate(read, position, candidate)
      ? SUPPORTS_VARIANT_YES
      : SUPPORTS_VARIANT_NO;
    const strandValue =
      read.strand === 'forward' ? STRAND_FORWARD : STRAND_REVERSE;
    const mapqValue = encodeMapq(read.mapq);
    const insertSizeValue = encodeInsertSize(read.insertSize);
    const insertionOffsets = new Set(
      (read.insertions ?? []).map((ins) => ins.offset),
    );

    for (let col = 0; col < PILEUP_WIDTH; col++) {
      const refCol = startCol + col;
      const cellIdx = refCol - read.startCol;
      if (cellIdx < 0 || cellIdx >= read.bases.length) continue;
      const refBase = reference[refCol];
      if (!refBase) continue;
      const base = read.bases[cellIdx];

      // Deletion column: leave empty (zero across all channels). DV's CIGAR
      // walker doesn't write deletion columns; only the cell BEFORE a
      // deletion run gets the '*' anchor write (handled below).
      if (base === '-') continue;

      // Indel anchor: the cell carries the '*' encoding instead of the
      // actual base intensity. Two cases:
      //   - deletion anchor: this cell is a real base AND the next cell is '-'
      //     (the deletion run starts at cellIdx+1)
      //   - insertion anchor: this cell offset is in read.insertions
      const isDeletionAnchor =
        cellIdx + 1 < read.bases.length && read.bases[cellIdx + 1] === '-';
      const isInsertionAnchor = insertionOffsets.has(cellIdx);
      const isAnchor = isDeletionAnchor || isInsertionAnchor;

      const quality = read.qualities[cellIdx] ?? 0;
      const o = cellOffset(imageRow, col, 0);
      out[o + 0] = isAnchor ? INDEL_ANCHOR_INTENSITY : baseIntensity(base);
      out[o + 1] = encodeBaseQuality(quality);
      out[o + 2] = mapqValue;
      out[o + 3] = strandValue;
      out[o + 4] = supportsRow;
      // '*' anchor never matches the ref base, so differs_from_ref = YES.
      out[o + 5] = isAnchor
        ? DIFFERS_FROM_REF_YES
        : base === refBase
          ? DIFFERS_FROM_REF_NO
          : DIFFERS_FROM_REF_YES;
      out[o + 6] = insertSizeValue;
    }
  }

  return { tensor: out, rowToReadId };
}

function passesImageFilter(read: Read, position: number): boolean {
  if (read.mapq < IMAGE_MIN_MAPPING_QUALITY) return false;
  const offset = position - read.startCol;
  if (offset < 0 || offset >= read.bases.length) return true; // doesn't cover
  if (read.bases[offset] === '-') return true; // deletion column, no Q check
  const q = read.qualities[offset] ?? 0;
  return q >= IMAGE_MIN_BASE_QUALITY;
}

function baseIntensity(base: Base): number {
  return BASE_INTENSITY[base] ?? 0;
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

/**
 * Per-channel value sets observed across the 5 golden DV pileups (NA12878
 * chr20 hom_alt SNVs). Channels with continuous distributions (base_qual,
 * insert_size) are bounded ranges instead. Use these to sanity-check what
 * `encodePileup` produces — values outside these sets/ranges are red flags.
 */
export const GOLDEN_CHANNEL_RANGES = {
  read_base: { allowed: new Set([0, 30, 100, 180, 250]) },
  base_quality: { min: 0, max: 254 },
  mapping_quality: { min: 0, max: 254 },
  strand: { allowed: new Set([0, 70, 240]) },
  supports_variant: { allowed: new Set([0, 152, 254]) },
  differs_from_ref: { allowed: new Set([0, 50, 254]) },
  insert_size: { min: 0, max: 254 },
};

export interface ValidationReport {
  passed: boolean;
  issues: string[];
  channelHistograms: Array<Record<number, number>>;
}

/**
 * Audit a tensor produced by `encodePileup` against the value sets
 * extracted from golden DV pileups. Discrete channels (read_base, strand,
 * supports_variant, differs_from_ref) must use only allowed values;
 * continuous channels are checked for in-range only. Returns histograms
 * to make discrepancies easy to spot.
 */
export function validateEncodedTensor(tensor: Float32Array): ValidationReport {
  const issues: string[] = [];
  const histograms: Array<Record<number, number>> = Array.from(
    { length: PILEUP_CHANNELS },
    () => ({}),
  );
  const ranges = [
    GOLDEN_CHANNEL_RANGES.read_base,
    GOLDEN_CHANNEL_RANGES.base_quality,
    GOLDEN_CHANNEL_RANGES.mapping_quality,
    GOLDEN_CHANNEL_RANGES.strand,
    GOLDEN_CHANNEL_RANGES.supports_variant,
    GOLDEN_CHANNEL_RANGES.differs_from_ref,
    GOLDEN_CHANNEL_RANGES.insert_size,
  ];
  const channelNames = [
    'read_base',
    'base_quality',
    'mapping_quality',
    'strand',
    'supports_variant',
    'differs_from_ref',
    'insert_size',
  ];

  for (let ch = 0; ch < PILEUP_CHANNELS; ch++) {
    const range = ranges[ch];
    const hist = histograms[ch];
    const violations = new Set<number>();
    for (let i = ch; i < tensor.length; i += PILEUP_CHANNELS) {
      const v = Math.round(tensor[i]);
      hist[v] = (hist[v] ?? 0) + 1;
      if ('allowed' in range && !range.allowed.has(v)) {
        violations.add(v);
      } else if ('min' in range && (v < range.min || v > range.max)) {
        violations.add(v);
      }
    }
    if (violations.size > 0) {
      const list = Array.from(violations).sort((a, b) => a - b).slice(0, 10);
      issues.push(
        `[${ch}] ${channelNames[ch]}: ${violations.size} disallowed value(s) (showing up to 10): ${list.join(', ')}`,
      );
    }
  }
  return { passed: issues.length === 0, issues, channelHistograms: histograms };
}
