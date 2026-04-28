import type { Base } from './palette';

export type Strand = 'forward' | 'reverse';

export interface Read {
  id: string;
  startCol: number;
  bases: Base[];
  qualities: Uint8Array;
  strand: Strand;
  mapq: number;
  insertSize: number;
}

export const DEFAULT_READ_COUNT = 40;
export const READ_MIN_LENGTH = 90;
export const READ_MAX_LENGTH = 130;

function makeRng(seed: number): () => number {
  let x = seed;
  return () => {
    x = (Math.imul(x, 1103515245) + 12345) | 0;
    return ((x >>> 8) & 0xffff) / 0xffff;
  };
}

export function buildReads(
  reference: Base[],
  count = DEFAULT_READ_COUNT,
  seed = 31337,
): Read[] {
  const rng = makeRng(seed);
  const reads: Read[] = [];

  for (let i = 0; i < count; i++) {
    const length =
      READ_MIN_LENGTH +
      Math.floor(rng() * (READ_MAX_LENGTH - READ_MIN_LENGTH + 1));
    const startCol = Math.floor(rng() * (reference.length - length));
    const bases: Base[] = reference.slice(startCol, startCol + length);
    const qualities = new Uint8Array(length).fill(35);
    const strand: Strand = rng() < 0.5 ? 'forward' : 'reverse';
    reads.push({
      id: `r${i.toString().padStart(2, '0')}`,
      startCol,
      bases,
      qualities,
      strand,
      mapq: 60,
      insertSize: 350,
    });
  }

  reads.sort((a, b) => a.startCol - b.startCol);
  return reads;
}
