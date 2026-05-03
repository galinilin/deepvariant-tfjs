import type { Base, Cell } from './palette';
import type { Read, Insertion } from './reads';
import { buildReads, makeRng } from './reads';
import { buildReference } from './reference';
import { placeScenarios, type Scenario, type ScenarioType } from './scenarios';

export type WorldKind = 'synthetic' | 'real-bam';

export interface World {
  kind: WorldKind;
  reference: Base[];
  /**
   * For 'synthetic' worlds these are the generated `het` / `het_del` / etc.
   * scenarios.
   *
   * For 'real-bam' worlds we re-purpose the same shape so the rest of the UI
   * (scrubber hint dots, snap-to-scenario, predict-label) keeps working
   * unchanged. `Scenario.type` carries the kind from DV's emitted candidate
   * (snv/del/ins → 'het'/'het_del'/'het_ins' as a placeholder; the actual
   * candidate at the predict column is derived live from reads either way).
   */
  scenarios: Scenario[];
  reads: Read[];
  /** Region label for the UI: e.g. "chr20:10000001–10005000". */
  label: string;
}

export interface SyntheticOpts {
  kind: 'synthetic';
  seed: number;
}

export interface RealBamOpts {
  kind: 'real-bam';
  fixtureBaseUrl?: string;
}

export type WorldOpts = SyntheticOpts | RealBamOpts;

interface RealBamMeta {
  chrom: string;
  region_start: number;
  region_end: number;
  region_length: number;
  bam: string;
  fasta: string;
  n_reads: number;
  n_candidates: number;
}

interface ReadJson {
  id: string;
  startCol: number;
  bases: string[];
  qualities: number[];
  strand: 'forward' | 'reverse';
  mapq: number;
  insertSize: number;
  row: number;
  insertions: Array<{
    offset: number;
    bases: string[];
    qualities: number[];
  }> | null;
}

interface CandidateJson {
  position: number;
  position_genomic: number;
  ref_alleles: string;
  alt_alleles: string[];
  alt_indices: number[];
  emitted_alts: string[];
  primary_kind: 'snv' | 'del' | 'ins' | 'complex' | 'unknown';
}

export async function buildWorld(opts: WorldOpts): Promise<World> {
  if (opts.kind === 'synthetic') {
    const rng = makeRng(opts.seed);
    const reference = buildReference(undefined, opts.seed);
    const scenarios = placeScenarios(reference, rng);
    const reads = buildReads(reference, scenarios, rng);
    return {
      kind: 'synthetic',
      reference,
      scenarios,
      reads,
      label: 'synthetic',
    };
  }

  const base = (opts.fixtureBaseUrl ?? `${import.meta.env.BASE_URL}fixtures/browser-region/`).replace(
    /\/?$/,
    '/',
  );

  const [metaResp, refResp, candResp, readsBuf] = await Promise.all([
    fetch(`${base}meta.json`).then((r) => r.json() as Promise<RealBamMeta>),
    fetch(`${base}reference.txt`).then((r) => r.text()),
    fetch(`${base}candidates.json`).then((r) => r.json() as Promise<CandidateJson[]>),
    fetch(`${base}reads.json.gz`).then((r) => r.arrayBuffer()),
  ]);

  // Decompress gzipped reads via the platform's native DecompressionStream.
  const ds = new DecompressionStream('gzip');
  const readsBlob = new Blob([readsBuf]);
  const decompressed = await new Response(readsBlob.stream().pipeThrough(ds)).text();
  const readsJson = JSON.parse(decompressed) as ReadJson[];

  const reference: Base[] = refResp.split('').map((c) => c.toUpperCase() as Base);

  const reads: Read[] = readsJson.map((r, i) => ({
    id: r.id,
    startCol: r.startCol,
    bases: r.bases as Cell[],
    qualities: new Uint8Array(r.qualities),
    strand: r.strand,
    mapq: r.mapq,
    insertSize: r.insertSize,
    row: i, // initial; packReads is unused here — we set rows below
    insertions: r.insertions
      ? (r.insertions.map((ins) => ({
          offset: ins.offset,
          bases: ins.bases as Base[],
          qualities: new Uint8Array(ins.qualities),
        })) as Insertion[])
      : undefined,
  }));

  // IGV-style packing of real reads: sort by startCol; assign each to the
  // lowest row whose last placed read ends before this one starts. (Mirrors
  // the synthetic path's packReads().)
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

  // Re-purpose the synthetic Scenario shape so scrubber/predict-label keep
  // working without modification. Map DV's primary_kind → our existing
  // ScenarioType (we use the 'het' family — 'hom_alt' family would also be
  // valid; 'het' is the visually softer choice for hint dots).
  const kindMap: Record<CandidateJson['primary_kind'], ScenarioType> = {
    snv: 'het',
    del: 'het_del',
    ins: 'het_ins',
    complex: 'het',
    unknown: 'het',
  };
  const scenarios: Scenario[] = candResp.map((c) => ({
    position: c.position,
    type: kindMap[c.primary_kind] ?? 'het',
    altBase:
      c.primary_kind === 'snv' && c.emitted_alts.length === 1
        ? (c.emitted_alts[0] as Base)
        : undefined,
  }));

  const oneBased = metaResp.region_start + 1;
  return {
    kind: 'real-bam',
    reference,
    scenarios,
    reads,
    label: `${metaResp.chrom}:${oneBased.toLocaleString()}–${metaResp.region_end.toLocaleString()}`,
  };
}
