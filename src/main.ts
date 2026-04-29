import { mountTopSketch, type HoverInfo } from './sketch/top';
import { mountBottomSketch } from './sketch/bottom';

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
if (tooltip) {
  topEl.addEventListener('mousemove', (ev) => {
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
    // Flip to the other side if we'd overflow the viewport.
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
  const supportsClass =
    info.supportsCandidate === true
      ? 'supports-yes'
      : info.supportsCandidate === false
        ? 'supports-no'
        : 'muted';
  const supportsText =
    info.supportsCandidate === true
      ? 'yes'
      : info.supportsCandidate === false
        ? 'no'
        : 'n/a (not predict column)';
  const baseLabel = info.base === '-' ? 'del' : info.base;
  return [
    `<div class="row-id">${esc(info.readId)} <span class="muted">·</span> ` +
      `${info.startCol + 1}–${info.endCol + 1} <span class="muted">·</span> ` +
      `${info.strand}</div>`,
    `<div class="muted">mapq=${info.mapq}  insert=${info.insertSize}</div>`,
    `<div>pos ${info.absCol + 1} <span class="muted">·</span> ` +
      `base=<b>${baseLabel}</b> Q=${info.quality}</div>`,
    `<div class="muted">supports candidate: <span class="${supportsClass}">${supportsText}</span></div>`,
  ].join('');
}

function esc(s: string): string {
  return s.replace(/[&<>]/g, (c) =>
    c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;',
  );
}
