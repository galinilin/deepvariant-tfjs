import { mountTopSketch, type HoverInfo } from './sketch/top';
import { mountBottomSketch } from './sketch/bottom';
import { formatAlt, type CandidateOutcome } from './lib/candidate';
import {
  verifyGoldenParity,
  inspectGoldenChannels,
  compareEncoderAgainstGolden,
} from './lib/parity';
import { sandboxState } from './lib/sandbox-state';

const topEl = document.getElementById('sketch-top');
const bottomEl = document.getElementById('sketch-bottom');
if (!topEl || !bottomEl) throw new Error('missing sketch containers');

const top = mountTopSketch(topEl);
mountBottomSketch(bottomEl);

const resetBtn = document.getElementById('reset-view');
resetBtn?.addEventListener('click', () => top.resetView());

const randomizeBtn = document.getElementById('randomize');
randomizeBtn?.addEventListener('click', () => top.randomize());

const verifyBtn = document.getElementById('verify-golden') as HTMLButtonElement | null;
verifyBtn?.addEventListener('click', async () => {
  const original = verifyBtn.textContent ?? 'Verify golden';
  verifyBtn.disabled = true;
  try {
    const result = await verifyGoldenParity({
      onProgress: (s) => {
        verifyBtn.textContent = s;
      },
    });
    const headline = result.passed
      ? '✓ exact'
      : result.argmaxMatches
        ? `~ argmax ✓ (uint8 quant Δ${result.maxDelta.toExponential(1)})`
        : `✗ argmax mismatch (Δ${result.maxDelta.toExponential(1)})`;
    console.group(`Golden parity ${headline}`);
    for (const line of result.details) console.log(line);
    console.groupEnd();
    verifyBtn.textContent = result.passed
      ? 'Golden ✓'
      : result.argmaxMatches
        ? 'Golden ✓ (quant)'
        : `Golden ✗ Δ${result.maxDelta.toExponential(1)}`;
  } catch (err) {
    console.error('verify-golden failed:', err);
    verifyBtn.textContent = 'Verify failed (see console)';
  } finally {
    setTimeout(() => {
      verifyBtn.textContent = original;
      verifyBtn.disabled = false;
    }, 6000);
  }
});

const debugBtn = document.getElementById('debug-force') as HTMLButtonElement | null;
const renderDebugLabel = () => {
  if (!debugBtn) return;
  debugBtn.textContent = sandboxState.forcePredict ? 'Debug: on' : 'Debug: off';
  debugBtn.style.color = sandboxState.forcePredict ? '#ffdc6e' : '';
};
renderDebugLabel();
debugBtn?.addEventListener('click', () => {
  sandboxState.forcePredict = !sandboxState.forcePredict;
  sandboxState.debugLogs = sandboxState.forcePredict;
  // Invalidate the current encoded tensor so the next frame re-encodes
  // with the new mode. Setting pileupPosition to a sentinel forces re-encode.
  sandboxState.pileupPosition = -1;
  sandboxState.prediction = null;
  renderDebugLabel();
});

const encoderBtn = document.getElementById('encoder-vs-golden') as HTMLButtonElement | null;
encoderBtn?.addEventListener('click', async () => {
  const original = encoderBtn.textContent ?? 'Encoder vs golden';
  encoderBtn.disabled = true;
  try {
    const { DeepVariantModel } = await import('./lib/DeepVariantModel');
    const base = (import.meta.env.BASE_URL || '/');
    encoderBtn.textContent = 'Loading model…';
    const model = await DeepVariantModel.load({ modelBaseUrl: `${base}models/` });
    const summary: string[] = [];
    let allMatch = true;
    for (let idx = 0; idx < 5; idx++) {
      encoderBtn.textContent = `Sample ${idx}/5…`;
      const result = await compareEncoderAgainstGolden({
        sampleIndex: idx,
        model,
        onProgress: (s) => {
          encoderBtn.textContent = `[${idx}] ${s}`;
        },
      });
      console.group(
        `[${idx}] ${result.variant}: ours=${result.ourArgmax} golden=${result.goldenArgmax} ${result.argmaxMatch ? '✓' : '✗'}`,
      );
      console.log(
        `  our probs:    [${result.ourProbs.map((v) => v.toFixed(4)).join(', ')}]`,
      );
      console.log(
        `  golden probs: [${result.goldenProbs.map((v) => v.toFixed(4)).join(', ')}]`,
      );
      for (const ch of result.channelStats) {
        const extra = ch.extraInOurs.length ? ` extraInOurs=[${ch.extraInOurs.join(',')}]` : '';
        const missing = ch.missingFromOurs.length
          ? ` missingFromOurs=[${ch.missingFromOurs.join(',')}]`
          : '';
        const flag = ch.extraInOurs.length || ch.missingFromOurs.length ? '⚠' : '✓';
        console.log(`  ${flag} ${ch.channel}:${extra}${missing}`);
      }
      console.groupEnd();
      summary.push(
        `[${idx}] ${result.argmaxMatch ? '✓' : '✗'} ours=${result.ourArgmax} golden=${result.goldenArgmax}`,
      );
      if (!result.argmaxMatch) allMatch = false;
    }
    model.dispose();
    console.log('---');
    summary.forEach((s) => console.log(s));
    encoderBtn.textContent = allMatch ? 'Encoder ✓ (5/5)' : `Encoder ✗ (see console)`;
  } catch (err) {
    console.error('encoder-vs-golden failed:', err);
    encoderBtn.textContent = 'Failed (see console)';
  } finally {
    setTimeout(() => {
      encoderBtn.textContent = original;
      encoderBtn.disabled = false;
    }, 8000);
  }
});

const inspectBtn = document.getElementById('inspect-golden') as HTMLButtonElement | null;
inspectBtn?.addEventListener('click', async () => {
  const original = inspectBtn.textContent ?? 'Inspect golden';
  inspectBtn.disabled = true;
  inspectBtn.textContent = 'Loading…';
  try {
    await inspectGoldenChannels();
    inspectBtn.textContent = 'Logged to console';
  } catch (err) {
    console.error('inspect-golden failed:', err);
    inspectBtn.textContent = 'Failed (see console)';
  } finally {
    setTimeout(() => {
      inspectBtn.textContent = original;
      inspectBtn.disabled = false;
    }, 4000);
  }
});

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
