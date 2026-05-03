import { mountTopSketch, type HoverInfo } from './sketch/top';
import { mountBottomSketch } from './sketch/bottom';
import { formatAlt, type CandidateOutcome } from './lib/candidate';
import { encodePileup, validateEncodedTensor } from './lib/pileup-encoder';
import { sandboxState } from './lib/sandbox-state';

// Debug helpers exposed on window for ad-hoc inspection from the browser
// console. Run e.g. `dvDebug.predictColumnStats()` to see what the
// encoder is producing at the candidate column for the current state.
(globalThis as unknown as { dvDebug: unknown }).dvDebug = {
  state: () => ({
    candidate: sandboxState.candidate,
    nReads: sandboxState.reads?.length ?? 0,
    refLen: sandboxState.reference?.length ?? 0,
    predictPos: sandboxState.predictPos,
    generation: sandboxState.readsGeneration,
  }),
  encode: () => {
    if (!sandboxState.reads || !sandboxState.reference || sandboxState.predictPos === null || !sandboxState.candidate) {
      return null;
    }
    return encodePileup(
      sandboxState.reads,
      sandboxState.reference,
      sandboxState.predictPos,
      sandboxState.candidate,
    );
  },
  validate: () => {
    const t = (globalThis as unknown as { dvDebug: { encode: () => Float32Array | null } }).dvDebug.encode();
    return t ? validateEncodedTensor(t) : null;
  },
  predictColumnStats: () => {
    const t = (globalThis as unknown as { dvDebug: { encode: () => Float32Array | null } }).dvDebug.encode();
    if (!t) return null;
    const PREDICT_COL = 110;
    const W = 221;
    const C = 7;
    const names = ['read_base', 'base_quality', 'mapping_quality', 'strand', 'supports_variant', 'differs_from_ref', 'insert_size'];
    const stats: { ch: string; counts: [number, number][] }[] = [];
    for (let ch = 0; ch < C; ch++) {
      const counts = new Map<number, number>();
      for (let row = 0; row < 100; row++) {
        const v = Math.round(t[row * W * C + PREDICT_COL * C + ch]);
        counts.set(v, (counts.get(v) ?? 0) + 1);
      }
      stats.push({
        ch: names[ch],
        counts: Array.from(counts.entries()).sort((a, b) => b[1] - a[1]),
      });
    }
    return stats;
  },
  rowSummary: () => {
    const t = (globalThis as unknown as { dvDebug: { encode: () => Float32Array | null } }).dvDebug.encode();
    if (!t) return null;
    const W = 221;
    const C = 7;
    let yes = 0, no = 0, empty = 0;
    for (let row = 0; row < 100; row++) {
      // Look at supports_variant (ch 4) at the predict column for this row
      const v = Math.round(t[row * W * C + 110 * C + 4]);
      if (v === 254) yes++;
      else if (v === 152) no++;
      else if (v === 0) empty++;
    }
    return { yes, no, empty };
  },
};

const topEl = document.getElementById('sketch-top');
const bottomEl = document.getElementById('sketch-bottom');
if (!topEl || !bottomEl) throw new Error('missing sketch containers');

const top = mountTopSketch(topEl);
mountBottomSketch(bottomEl);

const resetBtn = document.getElementById('reset-view');
resetBtn?.addEventListener('click', () => top.resetView());

const randomizeBtn = document.getElementById('randomize');
randomizeBtn?.addEventListener('click', () => top.randomize());

const tooltip = document.getElementById('read-tooltip');
let isDragging = false;
if (tooltip) {
  topEl.addEventListener('mousedown', () => {
    isDragging = true;
    tooltip.style.display = 'none';
  });
  window.addEventListener('mouseup', () => {
    isDragging = false;
  });
  topEl.addEventListener('mousemove', (ev) => {
    if (isDragging) return;
    const rect = topEl.getBoundingClientRect();
    const sx = ev.clientX - rect.left;
    const sy = ev.clientY - rect.top;
    const info = top.hoverInfo(sx, sy);
    if (!info) {
      tooltip.style.display = 'none';
      return;
    }
    tooltip.innerHTML = formatTooltip(info);
    const padding = 14;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let leftPx = ev.clientX + padding;
    let topPx = ev.clientY + padding;
    if (leftPx + tooltip.offsetWidth + 8 > vw) {
      leftPx = ev.clientX - tooltip.offsetWidth - padding;
    }
    if (topPx + tooltip.offsetHeight + 8 > vh) {
      topPx = ev.clientY - tooltip.offsetHeight - padding;
    }
    tooltip.style.left = `${leftPx}px`;
    tooltip.style.top = `${topPx}px`;
    tooltip.style.display = 'block';
  });
  topEl.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });
}

function formatTooltip(info: HoverInfo): string {
  const supportsLine = formatSupportsLine(info);
  const readSection = [
    `<div class="section">`,
    `<div class="section-label">Read</div>`,
    `<div class="row-id">${esc(info.readId)} <span class="muted">·</span> ` +
      `${info.startCol + 1}–${info.endCol + 1} <span class="muted">·</span> ` +
      `${info.strand}</div>`,
    `<div class="muted">mapq=${info.mapq}</div>`,
    `</div>`,
  ].join('');

  if (info.kind === 'insertion') {
    const seqLabel = '+' + info.sequence.join('');
    const qLabel = Array.from(info.qualities).map((q) => `Q${q}`).join(' ');
    return [
      readSection,
      `<div class="section">`,
      `<div class="section-label">Insertion after locus ${info.absCol + 1}</div>`,
      `<div><b class="ins-seq">${esc(seqLabel)}</b> <span class="muted">· ${esc(qLabel)}</span></div>`,
      supportsLine,
      `</div>`,
    ].join('');
  }

  const baseLabel = info.base === '-' ? 'del' : info.base;
  return [
    readSection,
    `<div class="section">`,
    `<div class="section-label">Locus ${info.absCol + 1}</div>`,
    `<div>base=<b>${baseLabel}</b>${info.base !== '-' ? ` <span class="muted">·</span> Q=${info.quality}` : ''}</div>`,
    supportsLine,
    `</div>`,
  ].join('');
}

function formatSupportsLine(info: HoverInfo): string {
  if (!info.isPredictColumn) {
    return `<div class="muted">supports candidate: n/a</div>`;
  }
  if (info.candidate && info.supportsCandidate !== null) {
    const supports = info.supportsCandidate ? 'yes' : 'no';
    const cls = info.supportsCandidate ? 'supports-yes' : 'supports-no';
    const counts = `${info.candidate.supportingReads}/${info.candidate.qualifyingReads}`;
    return `<div class="muted">supports candidate: <span class="${cls}">${supports}</span> (${counts})</div>`;
  }
  return `<div class="muted">${formatRejection(info.outcome)}</div>`;
}

function formatRejection(outcome: CandidateOutcome): string {
  switch (outcome.kind) {
    case 'accepted':
      return '';
    case 'no-coverage':
      return 'no candidate · no qualifying reads';
    case 'no-alt-evidence':
      return `no candidate · 0 alt of ${outcome.qualifyingReads}`;
    case 'below-count':
      return `no candidate · ${outcome.count}× ${esc(formatAlt(outcome.alt))} < 2 required`;
    case 'below-fraction': {
      const pct = ((outcome.count / outcome.qualifyingReads) * 100).toFixed(1);
      const minPct = outcome.alt.kind === 'snv' ? 12 : 6;
      return `no candidate · ${pct}% < ${minPct}% required`;
    }
  }
}

function esc(s: string): string {
  return s.replace(/[&<>]/g, (c) =>
    c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;',
  );
}
