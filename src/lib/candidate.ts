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
const MIN_MAPPING_QUALITY = 5;
const MIN_BASE_QUALITY = 10;
const MIN_COUNT_SNP = 2;
const MIN_COUNT_INDEL = 2;
const MIN_FRACTION_SNP = 0.12;
const MIN_FRACTION_INDEL = 0.06;

export interface CandidateInfo {
  base: Cell;
  refBase: Base;
  supportingReads: number;
  qualifyingReads: number;
}

export type Candidate = CandidateInfo | null;

/**
 * Derive the candidate alt at a genomic position by majority vote across
 * reads that cover it, gated by DV's candidate-generation thresholds.
 *
 *  1. Reads with mapq < MIN_MAPPING_QUALITY don't count.
 *  2. For non-deletion bases, base quality < MIN_BASE_QUALITY don't count.
 *  3. An allele only becomes a candidate if its qualifying count meets
 *     both the minimum count (2) and the minimum fraction (12% for SNPs,
 *     6% for indels) of all qualifying reads at the position.
 *  4. If multiple alleles qualify, the highest-count one wins; ties break
 *     alphabetically (so '-' < 'A' < 'C' < 'G' < 'T').
 *
 * Returns null if no allele qualifies — meaning DV's `make_examples` would
 * not have generated a pileup image here, and no model evaluation happens.
 */
export function deriveCandidate(
  reads: Read[],
  reference: Base[],
  position: number,
): Candidate {
  const refBase = reference[position];
  if (!refBase || refBase === 'N') return null;

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

  if (qualifyingTotal === 0 || counts.size === 0) return null;

  let best: Cell | null = null;
  let bestCount = 0;
  for (const [base, count] of counts) {
    const isIndel = base === '-';
    const minCount = isIndel ? MIN_COUNT_INDEL : MIN_COUNT_SNP;
    const minFraction = isIndel ? MIN_FRACTION_INDEL : MIN_FRACTION_SNP;
    if (count < minCount) continue;
    if (count / qualifyingTotal < minFraction) continue;

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
    qualifyingReads: qualifyingTotal,
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
