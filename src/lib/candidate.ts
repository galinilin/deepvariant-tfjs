import type { Base } from './palette';
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

export type CandidateInfo =
  | {
      kind: 'snv';
      refBase: Base;
      altBase: Base;
      supportingReads: number;
      qualifyingReads: number;
    }
  | {
      kind: 'del';
      refBase: Base;
      supportingReads: number;
      qualifyingReads: number;
    }
  | {
      kind: 'ins';
      refBase: Base;
      altSequence: Base[];
      supportingReads: number;
      qualifyingReads: number;
    };

export type Candidate = CandidateInfo | null;

/** Lightweight alt descriptor used by rejection outcomes and label formatting. */
export type AltDescriptor =
  | { kind: 'snv'; altBase: Base }
  | { kind: 'del' }
  | { kind: 'ins'; altSequence: Base[] };

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
      alt: AltDescriptor;
      count: number;
      qualifyingReads: number;
    }
  | {
      kind: 'below-fraction';
      alt: AltDescriptor;
      count: number;
      qualifyingReads: number;
    };

type BestAlt =
  | { kind: 'snv'; altBase: Base; count: number }
  | { kind: 'del'; count: number }
  | { kind: 'ins'; altSequence: Base[]; count: number };

export function deriveCandidateOutcome(
  reads: Read[],
  reference: Base[],
  position: number,
): CandidateOutcome {
  const refBase = reference[position];
  if (!refBase || refBase === 'N') return { kind: 'no-coverage' };

  const snvCounts = new Map<Base, number>();
  let delCount = 0;
  const insCounts = new Map<string, number>();
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
    if (base === '-') {
      delCount++;
    } else if (base !== refBase) {
      snvCounts.set(base, (snvCounts.get(base) ?? 0) + 1);
    }

    if (read.insertions) {
      for (const ins of read.insertions) {
        if (ins.offset === offset) {
          const key = ins.bases.join('');
          insCounts.set(key, (insCounts.get(key) ?? 0) + 1);
        }
      }
    }
  }

  if (qualifyingTotal === 0) return { kind: 'no-coverage' };

  // Collect every alt candidate, then sort: highest count wins; SNV>del>ins
  // on count ties; alphabetical within SNV; shortest sequence within ins.
  const candidates: BestAlt[] = [];
  for (const [b, c] of snvCounts) {
    candidates.push({ kind: 'snv', altBase: b, count: c });
  }
  if (delCount > 0) {
    candidates.push({ kind: 'del', count: delCount });
  }
  for (const [seqKey, c] of insCounts) {
    candidates.push({
      kind: 'ins',
      altSequence: seqKey.split('') as Base[],
      count: c,
    });
  }

  if (candidates.length === 0) {
    return { kind: 'no-alt-evidence', qualifyingReads: qualifyingTotal };
  }

  const kindRank = (k: BestAlt['kind']): number =>
    k === 'snv' ? 0 : k === 'del' ? 1 : 2;

  candidates.sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    if (a.kind !== b.kind) return kindRank(a.kind) - kindRank(b.kind);
    if (a.kind === 'snv' && b.kind === 'snv') {
      return a.altBase < b.altBase ? -1 : a.altBase > b.altBase ? 1 : 0;
    }
    if (a.kind === 'ins' && b.kind === 'ins') {
      if (a.altSequence.length !== b.altSequence.length) {
        return a.altSequence.length - b.altSequence.length;
      }
      const aSeq = a.altSequence.join('');
      const bSeq = b.altSequence.join('');
      return aSeq < bSeq ? -1 : aSeq > bSeq ? 1 : 0;
    }
    return 0;
  });

  const best = candidates[0];
  const isIndel = best.kind !== 'snv';
  const minCount = isIndel ? MIN_COUNT_INDEL : MIN_COUNT_SNP;
  const minFraction = isIndel ? MIN_FRACTION_INDEL : MIN_FRACTION_SNP;

  const alt: AltDescriptor =
    best.kind === 'snv'
      ? { kind: 'snv', altBase: best.altBase }
      : best.kind === 'del'
        ? { kind: 'del' }
        : { kind: 'ins', altSequence: best.altSequence };

  if (best.count < minCount) {
    return {
      kind: 'below-count',
      alt,
      count: best.count,
      qualifyingReads: qualifyingTotal,
    };
  }
  if (best.count / qualifyingTotal < minFraction) {
    return {
      kind: 'below-fraction',
      alt,
      count: best.count,
      qualifyingReads: qualifyingTotal,
    };
  }

  const info: CandidateInfo =
    best.kind === 'snv'
      ? {
          kind: 'snv',
          refBase,
          altBase: best.altBase,
          supportingReads: best.count,
          qualifyingReads: qualifyingTotal,
        }
      : best.kind === 'del'
        ? {
            kind: 'del',
            refBase,
            supportingReads: best.count,
            qualifyingReads: qualifyingTotal,
          }
        : {
            kind: 'ins',
            refBase,
            altSequence: best.altSequence,
            supportingReads: best.count,
            qualifyingReads: qualifyingTotal,
          };

  return { kind: 'accepted', info };
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
  switch (candidate.kind) {
    case 'snv':
      return read.bases[offset] === candidate.altBase;
    case 'del':
      return read.bases[offset] === '-';
    case 'ins': {
      if (!read.insertions) return false;
      const target = candidate.altSequence.join('');
      for (const ins of read.insertions) {
        if (ins.offset === offset && ins.bases.join('') === target) return true;
      }
      return false;
    }
  }
}

export function formatAlt(alt: AltDescriptor | CandidateInfo): string {
  switch (alt.kind) {
    case 'snv':
      return alt.altBase;
    case 'del':
      return 'del';
    case 'ins':
      return '+' + alt.altSequence.join('');
  }
}
