import type { Base } from './palette';

export const DEFAULT_REFERENCE_LENGTH = 600;
export const WINDOW_LENGTH = 221;

const BASES: Base[] = ['A', 'C', 'G', 'T'];

export function buildReference(length = DEFAULT_REFERENCE_LENGTH, seed = 17389): Base[] {
  const out: Base[] = new Array(length);
  let x = seed;
  for (let i = 0; i < length; i++) {
    x = (Math.imul(x, 1103515245) + 12345) | 0;
    out[i] = BASES[(x >>> 4) & 3];
  }
  return out;
}

export function defaultWindowStart(refLength: number): number {
  return Math.max(0, Math.floor((refLength - WINDOW_LENGTH) / 2));
}

export function clampWindowStart(start: number, refLength: number): number {
  const max = Math.max(0, refLength - WINDOW_LENGTH);
  if (start < 0) return 0;
  if (start > max) return max;
  return start;
}
