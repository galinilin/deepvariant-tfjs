import { mountTopSketch, type HoverInfo } from './sketch/top';
import { mountBottomSketch } from './sketch/bottom';
import type { CandidateOutcome } from './lib/candidate';

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
  const baseLabel = info.base === '-' ? 'del' : info.base;
  const supportsLine = formatSupportsLine(info);
  return [
    `<div class="section">`,
    `<div class="section-label">Read</div>`,
    `<div class="row-id">${esc(info.readId)} <span class="muted">·</span> ` +
      `${info.startCol + 1}–${info.endCol + 1} <span class="muted">·</span> ` +
      `${info.strand}</div>`,
    `<div class="muted">mapq=${info.mapq}</div>`,
    `</div>`,
    `<div class="section">`,
    `<div class="section-label">Locus ${info.absCol + 1}</div>`,
    `<div>base=<b>${baseLabel}</b> <span class="muted">·</span> Q=${info.quality}</div>`,
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
      return `no candidate · ${outcome.count}× ${esc(outcome.bestAlt === '-' ? 'del' : outcome.bestAlt)} < 2 required`;
    case 'below-fraction': {
      const pct = ((outcome.count / outcome.qualifyingReads) * 100).toFixed(1);
      const minPct = outcome.bestAlt === '-' ? 6 : 12;
      return `no candidate · ${pct}% < ${minPct}% required`;
    }
  }
}

function esc(s: string): string {
  return s.replace(/[&<>]/g, (c) =>
    c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;',
  );
}
