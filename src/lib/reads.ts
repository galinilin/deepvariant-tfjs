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
}

export const DEFAULT_READ_COUNT = 40;
export const READ_MIN_LENGTH = 90;
export const READ_MAX_LENGTH = 130;

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

  reads.sort((a, b) => a.startCol - b.startCol);
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
  const qualities = new Uint8Array(length).fill(35);
  const strand: Strand = rng() < 0.5 ? 'forward' : 'reverse';
  return {
    id,
    startCol,
    bases,
    qualities,
    strand,
    mapq: 60,
    insertSize: 350,
  };
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
      case 'hom_ref':
        // No mutation; reads remain reference at this locus.
        break;
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
      case 'error_burst':
        if (rng() < 0.05) {
          const refBase = reference[sc.position] ?? 'N';
          const refSafe: Base =
            refBase === 'A' || refBase === 'C' || refBase === 'G' || refBase === 'T'
              ? refBase
              : 'A';
          read.bases[offset] = pickAltBase(refSafe, rng);
        }
        break;
    }
  }
}

function applyDeletion(read: Read, offset: number, length: number): void {
  for (let i = 0; i < length; i++) {
    if (offset + i < read.bases.length) {
      read.bases[offset + i] = '-';
    }
  }
}
