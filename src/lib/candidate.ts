import type { Base, Cell } from './palette';
import type { Read } from './reads';

/**
 * Candidate-generation thresholds matching DeepVariant 1.8 defaults
 * (`make_examples` flags). At a given position, a base is counted toward
 * an alt allele only if its read passes mapq>=5 AND (for non-deletion
 * bases) the base quality is >=10. An allele only becomes a candidate
 * if its qualifying count clears both a minimum count and a minimum
 * fraction-of-qualifying-reads.
 *
 * Source: google/deepvariant r1.8 make_examples_options.py
 *   vsc_min_count_snps=2, vsc_min_fraction_snps=0.12
 *   vsc_min_count_indels=2, vsc_min_fraction_indels=0.06
 *   min_base_quality=10, min_mapping_quality=5
 */
export const MIN_MAPPING_QUALITY = 5;
export const MIN_BASE_QUALITY = 10;
export const MIN_COUNT_SNP = 2;
export const MIN_COUNT_INDEL = 2;
export const MIN_FRACTION_SNP = 0.12;
export const MIN_FRACTION_INDEL = 0.06;

export interface CandidateInfo {
  base: Cell;
  refBase: Base;
  supportingReads: number;
  qualifyingReads: number;
}

export type Candidate = CandidateInfo | null;

/**
 * Discriminated outcome of running DV's candidate filter at a position.
 * `accepted` returns the chosen candidate. The four `rejected-*` cases
 * carry enough context for the UI to explain *why* DV would skip the
 * locus.
 */
export type CandidateOutcome =
  | { kind: 'accepted'; info: CandidateInfo }
  | { kind: 'no-coverage' }
  | { kind: 'no-alt-evidence'; qualifyingReads: number }
  | {
      kind: 'below-count';
      bestAlt: Cell;
      count: number;
      qualifyingReads: number;
    }
  | {
      kind: 'below-fraction';
      bestAlt: Cell;
      count: number;
      qualifyingReads: number;
    };

export function deriveCandidateOutcome(
  reads: Read[],
  reference: Base[],
  position: number,
): CandidateOutcome {
  const refBase = reference[position];
  if (!refBase || refBase === 'N') return { kind: 'no-coverage' };

  const counts = new Map<Cell, number>();
  let qualifyingTotal = 0;

  for (const read of reads) {
    const offset = position - read.startCol;
    if (offset < 0 || offset >= read.bases.length) continue;
    if (read.mapq < MIN_MAPPING_QUALITY) continue;

    const base = read.bases[offset];
    if (base !== '-') {
      const q = read.qualities[offset] ?? 0;
      if (q < MIN_BASE_QUALITY) continue;
    }

    qualifyingTotal++;
    if (base !== refBase) {
      counts.set(base, (counts.get(base) ?? 0) + 1);
    }
  }

  if (qualifyingTotal === 0) return { kind: 'no-coverage' };
  if (counts.size === 0) {
    return { kind: 'no-alt-evidence', qualifyingReads: qualifyingTotal };
  }

  // Find the most-common alt (alphabetical tie-break) regardless of pass/fail.
  let bestAlt: Cell | null = null;
  let bestCount = 0;
  for (const [base, count] of counts) {
    if (
      count > bestCount ||
      (count === bestCount && (bestAlt === null || base < bestAlt))
    ) {
      bestCount = count;
      bestAlt = base;
    }
  }
  if (bestAlt === null) {
    return { kind: 'no-alt-evidence', qualifyingReads: qualifyingTotal };
  }

  const isIndel = bestAlt === '-';
  const minCount = isIndel ? MIN_COUNT_INDEL : MIN_COUNT_SNP;
  const minFraction = isIndel ? MIN_FRACTION_INDEL : MIN_FRACTION_SNP;

  if (bestCount < minCount) {
    return {
      kind: 'below-count',
      bestAlt,
      count: bestCount,
      qualifyingReads: qualifyingTotal,
    };
  }
  if (bestCount / qualifyingTotal < minFraction) {
    return {
      kind: 'below-fraction',
      bestAlt,
      count: bestCount,
      qualifyingReads: qualifyingTotal,
    };
  }

  return {
    kind: 'accepted',
    info: {
      base: bestAlt,
      refBase,
      supportingReads: bestCount,
      qualifyingReads: qualifyingTotal,
    },
  };
}

export function deriveCandidate(
  reads: Read[],
  reference: Base[],
  position: number,
): Candidate {
  const outcome = deriveCandidateOutcome(reads, reference, position);
  return outcome.kind === 'accepted' ? outcome.info : null;
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

export function formatAlt(c: Cell): string {
  return c === '-' ? 'del' : c;
}
