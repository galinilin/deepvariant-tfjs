import type { Base, Cell } from './palette';
import type { Read } from './reads';

export interface CandidateInfo {
  base: Cell;
  refBase: Base;
  supportingReads: number;
  totalCovering: number;
}

export type Candidate = CandidateInfo | null;

/**
 * Derive the candidate alt at a genomic position by majority vote across reads
 * that cover it. Most-common non-ref base wins; ties broken alphabetically
 * (which puts '-' first, then A < C < G < T). Returns null if no read shows a
 * non-ref base at this position.
 */
export function deriveCandidate(
  reads: Read[],
  reference: Base[],
  position: number,
): Candidate {
  const refBase = reference[position];
  if (!refBase || refBase === 'N') return null;

  const counts = new Map<Cell, number>();
  let totalCovering = 0;

  for (const read of reads) {
    const offset = position - read.startCol;
    if (offset < 0 || offset >= read.bases.length) continue;
    totalCovering++;
    const base = read.bases[offset];
    if (base !== refBase) {
      counts.set(base, (counts.get(base) ?? 0) + 1);
    }
  }

  if (counts.size === 0) return null;

  let best: Cell | null = null;
  let bestCount = -1;
  for (const [base, count] of counts) {
    if (
      count > bestCount ||
      (count === bestCount && (best === null || base < best))
    ) {
      bestCount = count;
      best = base;
    }
  }
  if (best === null) return null;
  return {
    base: best,
    refBase,
    supportingReads: bestCount,
    totalCovering,
  };
}

export function readSupportsCandidate(
  read: Read,
  position: number,
  candidate: Candidate,
): boolean {
  if (!candidate) return false;
  const offset = position - read.startCol;
  if (offset < 0 || offset >= read.bases.length) return false;
  return read.bases[offset] === candidate.base;
}
