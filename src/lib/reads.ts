import type { Base, Cell } from './palette';
import type { Scenario } from './scenarios';
import { pickAltBase } from './scenarios';

export type Strand = 'forward' | 'reverse';

export interface Read {
  id: string;
  startCol: number;
  bases: Cell[];
  qualities: Uint8Array;
  strand: Strand;
  mapq: number;
  insertSize: number;
  row: number;
}

export const DEFAULT_READ_COUNT = 40;
export const READ_MIN_LENGTH = 90;
export const READ_MAX_LENGTH = 130;
export const MAX_PACKED_ROWS = 14;

export function makeRng(seed: number): () => number {
  let x = seed | 0;
  return () => {
    x = (Math.imul(x, 1103515245) + 12345) | 0;
    return ((x >>> 8) & 0xffff) / 0xffff;
  };
}

const SCENARIO_COVERAGE_FLOOR = 4;

export function buildReads(
  reference: Base[],
  scenarios: Scenario[],
  rng: () => number,
  count = DEFAULT_READ_COUNT,
): Read[] {
  const reads: Read[] = [];
  const coverageAt = (col: number): number =>
    reads.reduce(
      (n, r) => (col >= r.startCol && col < r.startCol + r.bases.length ? n + 1 : n),
      0,
    );

  // Pre-seed each scenario position with a few reads guaranteed to cover it,
  // so hint dots always land on something visible.
  for (const sc of scenarios) {
    while (coverageAt(sc.position) < SCENARIO_COVERAGE_FLOOR) {
      const length =
        READ_MIN_LENGTH +
        Math.floor(rng() * (READ_MAX_LENGTH - READ_MIN_LENGTH + 1));
      const offsetIntoRead = Math.floor(rng() * (length - 1));
      const startCol = Math.max(
        0,
        Math.min(reference.length - length, sc.position - offsetIntoRead),
      );
      reads.push(makeRead(`s${reads.length}`, startCol, length, reference, rng));
      if (reads.length >= count) break;
    }
    if (reads.length >= count) break;
  }

  // Fill the rest with uniformly-distributed reads.
  while (reads.length < count) {
    const length =
      READ_MIN_LENGTH +
      Math.floor(rng() * (READ_MAX_LENGTH - READ_MIN_LENGTH + 1));
    const startCol = Math.floor(rng() * (reference.length - length));
    reads.push(makeRead(`r${reads.length}`, startCol, length, reference, rng));
  }

  for (const read of reads) {
    applyScenariosToRead(read, scenarios, reference, rng);
  }

  packReads(reads);
  return reads;
}

function makeRead(
  id: string,
  startCol: number,
  length: number,
  reference: Base[],
  rng: () => number,
): Read {
  const bases: Cell[] = reference.slice(startCol, startCol + length);
  const qualities = qualityProfile(length, 36, rng);
  const strand: Strand = rng() < 0.5 ? 'forward' : 'reverse';
  return {
    id,
    startCol,
    bases,
    qualities,
    strand,
    mapq: generateMapq(rng),
    insertSize: 320 + Math.floor(rng() * 120),
    row: 0,
  };
}

/**
 * Per-base Phred quality profile. Mostly hovers around `baseQ` with mild
 * jitter, then decays toward the 3' end (last ~25% of the read drops by
 * up to ~18 points), simulating Illumina quality fall-off. Capped at Q40.
 */
function qualityProfile(
  length: number,
  baseQ: number,
  rng: () => number,
): Uint8Array {
  const out = new Uint8Array(length);
  const tailStart = Math.floor(length * 0.75);
  for (let i = 0; i < length; i++) {
    let q = baseQ + Math.floor(rng() * 7) - 3; // ±3 jitter
    if (i >= tailStart && tailStart < length) {
      const decayProgress = (i - tailStart) / (length - tailStart);
      q -= Math.round(decayProgress * 18);
    }
    out[i] = Math.max(8, Math.min(40, q));
  }
  return out;
}

/**
 * Realistic mapq distribution: ~85% of reads at high mapq (50–60), ~15%
 * at lower mapq (25–45) — a small "ambiguous mapping" tail.
 */
function generateMapq(rng: () => number): number {
  if (rng() < 0.85) {
    return 50 + Math.floor(rng() * 11); // 50..60
  }
  return 25 + Math.floor(rng() * 21); // 25..45
}

/**
 * IGV-style packing: each read goes in the lowest row where its span doesn't
 * overlap any read already placed in that row. Mutates each read.row.
 */
function packReads(reads: Read[]): void {
  reads.sort((a, b) => a.startCol - b.startCol);
  const rowEnds: number[] = [];
  for (const read of reads) {
    const readEnd = read.startCol + read.bases.length;
    let placed = false;
    for (let r = 0; r < rowEnds.length; r++) {
      if (rowEnds[r] <= read.startCol) {
        read.row = r;
        rowEnds[r] = readEnd;
        placed = true;
        break;
      }
    }
    if (!placed) {
      read.row = rowEnds.length;
      rowEnds.push(readEnd);
    }
  }
}

function applyScenariosToRead(
  read: Read,
  scenarios: Scenario[],
  reference: Base[],
  rng: () => number,
): void {
  for (const sc of scenarios) {
    const offset = sc.position - read.startCol;
    if (offset < 0 || offset >= read.bases.length) continue;

    switch (sc.type) {
      case 'het':
        if (rng() < 0.5 && sc.altBase) {
          read.bases[offset] = sc.altBase;
        }
        break;
      case 'hom_alt':
        if (rng() < 0.95 && sc.altBase) {
          read.bases[offset] = sc.altBase;
        }
        break;
      case 'het_del':
        if (rng() < 0.5) {
          applyDeletion(read, offset, sc.delLength ?? 1);
        }
        break;
      case 'hom_alt_del':
        if (rng() < 0.95) {
          applyDeletion(read, offset, sc.delLength ?? 1);
        }
        break;
      case 'noisy_burst':
        if (rng() < 0.25) {
          const refSafe = safeBase(reference[sc.position]);
          read.bases[offset] = pickAltBase(refSafe, rng);
        }
        break;
      case 'strand_bias_het':
        if (read.strand === 'forward' && rng() < 0.7 && sc.altBase) {
          read.bases[offset] = sc.altBase;
        }
        break;
      case 'tandem_snps':
        if (rng() < 0.5) {
          if (sc.altBase) {
            read.bases[offset] = sc.altBase;
          }
          if (sc.altBase2 && offset + 1 < read.bases.length) {
            read.bases[offset + 1] = sc.altBase2;
          }
        }
        break;
      case 'compound_alt': {
        const r = rng();
        if (r < 0.4 && sc.altBase) {
          read.bases[offset] = sc.altBase;
        } else if (r < 0.8 && sc.altBase2) {
          read.bases[offset] = sc.altBase2;
        }
        break;
      }
      case 'low_quality_alt':
        if (rng() < 0.5 && sc.altBase) {
          read.bases[offset] = sc.altBase;
          read.qualities[offset] = 10 + Math.floor(rng() * 9); // Q10..Q18
        }
        break;
      case 'low_mapq_alt':
        if (rng() < 0.5 && sc.altBase) {
          read.bases[offset] = sc.altBase;
          // Drag this read's mapq down — only reads carrying the alt
          // become the "ambiguously mapped" ones.
          read.mapq = 15 + Math.floor(rng() * 16); // 15..30
        }
        break;
    }
  }
}

function safeBase(b: Base | undefined): Base {
  return b === 'A' || b === 'C' || b === 'G' || b === 'T' ? b : 'A';
}

function applyDeletion(read: Read, offset: number, length: number): void {
  for (let i = 0; i < length; i++) {
    if (offset + i < read.bases.length) {
      read.bases[offset + i] = '-';
    }
  }
}
