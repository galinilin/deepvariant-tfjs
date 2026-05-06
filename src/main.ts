import { mountTopSketch, type HoverInfo, type SketchHandle } from './sketch/top';
import { mountBottomSketch, type BottomHandle } from './sketch/bottom';
import { formatAlt, type CandidateOutcome } from './lib/candidate';
import { encodePileup, validateEncodedTensor } from './lib/pileup-encoder';
import { sandboxState } from './lib/sandbox-state';
import { buildWorld } from './lib/world-builder';
import { mountDebugModal } from './sketch/debug-modal';

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
    const enc = encodePileup(
      sandboxState.reads,
      sandboxState.reference,
      sandboxState.predictPos,
      sandboxState.candidate,
    );
    return enc?.tensor ?? null;
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

const topElMaybe = document.getElementById('sketch-top');
const bottomElMaybe = document.getElementById('sketch-bottom');
if (!topElMaybe || !bottomElMaybe) throw new Error('missing sketch containers');
const topEl: HTMLElement = topElMaybe;
const bottomEl: HTMLElement = bottomElMaybe;

// v5.3: welcome screen + explicit Synthetic / Real picker. Model loads
// in parallel with the user reading the welcome text. Query-param
// shortcuts (?world=synthetic|real-bam) skip the picker for direct
// deep-linking.
const params = new URLSearchParams(window.location.search);
const queryWorld =
  params.get('world') === 'real-bam' || params.get('world') === 'real'
    ? 'real-bam'
    : params.get('world') === 'synthetic'
      ? 'synthetic'
      : null;

let top: SketchHandle | null = null;
let bottom: BottomHandle | null = null;
let chosenWorldKind: 'synthetic' | 'real-bam' = 'synthetic';

const welcomeEl = document.getElementById('welcome-overlay');
const welcomeStatus = document.getElementById('welcome-status');
const synthBtn = document.getElementById('start-synthetic') as HTMLButtonElement | null;
const realBtn = document.getElementById('start-real') as HTMLButtonElement | null;
const explainBtn = document.getElementById('show-explain');
const explainEl = document.getElementById('welcome-explain');

// v5.3: lazy model load. The 22 MB checkpoint download only starts when
// the user commits to a mode, not on page open. Visitors who never click
// pay nothing. Buttons are enabled immediately; clicking either kicks off
// the load + warmup + world build pipeline with live status.
let modelLoadInFlight:
  | Promise<import('./lib/DeepVariantModel').DeepVariantModel>
  | null = null;

async function loadModel(): Promise<
  import('./lib/DeepVariantModel').DeepVariantModel
> {
  if (modelLoadInFlight) return modelLoadInFlight;
  const { DeepVariantModel } = await import('./lib/DeepVariantModel');
  let lastPercent = -1;
  modelLoadInFlight = DeepVariantModel.load({
    onProgress: (frac) => {
      const pct = Math.floor(frac * 100);
      if (pct !== lastPercent && welcomeStatus) {
        welcomeStatus.textContent = `Loading model… ${pct}%`;
        lastPercent = pct;
      }
    },
    onStage: (stage) => {
      if (!welcomeStatus) return;
      if (stage === 'warming') welcomeStatus.textContent = 'Compiling shaders…';
      else if (stage === 'ready') welcomeStatus.textContent = 'Building world…';
    },
  });
  // If the load rejects, clear the cache so a retry click can try again.
  modelLoadInFlight.catch(() => {
    modelLoadInFlight = null;
  });
  return modelLoadInFlight;
}

explainBtn?.addEventListener('click', () => {
  if (!explainEl) return;
  const showing = explainEl.style.display !== 'none';
  explainEl.style.display = showing ? 'none' : 'block';
});

// Buttons enabled immediately — the load happens on click, not on open.
synthBtn?.addEventListener('click', () => void startWorld('synthetic'));
realBtn?.addEventListener('click', () => void startWorld('real-bam'));

// Query-param shortcut still works: triggers the same startWorld path,
// which loads the model on demand.
if (queryWorld) {
  void startWorld(queryWorld);
}

async function startWorld(kind: 'synthetic' | 'real-bam'): Promise<void> {
  chosenWorldKind = kind;
  if (synthBtn) synthBtn.disabled = true;
  if (realBtn) realBtn.disabled = true;
  if (welcomeStatus) welcomeStatus.classList.remove('error');

  try {
    // Load model + build world in parallel — both are async, both
    // depend on the click but not on each other. Model load callbacks
    // own the welcome status text during the longer (~22 MB) download.
    const buildingWorld = buildWorld(
      kind === 'synthetic'
        ? { kind: 'synthetic', seed: 42 }
        : { kind: 'real-bam' },
    );
    const [model, world] = await Promise.all([loadModel(), buildingWorld]);

    // Reveal sandbox containers before mounting so canvas size measurement
    // sees the right viewport dimensions. Use .sandbox-revealed (instead
    // of just removing .hidden-until-ready) so the containers fade in
    // over 0.4s instead of hard-cutting in.
    document.querySelectorAll('.hidden-until-ready').forEach((el) => {
      (el as HTMLElement).classList.add('sandbox-revealed');
    });

    top = mountTopSketch(topEl, world);
    bottom = mountBottomSketch(bottomEl, model);
    attachUiHandlers(top);

    // Fade out the welcome overlay then remove it.
    if (welcomeEl) {
      welcomeEl.classList.add('fade-out');
      setTimeout(() => welcomeEl.remove(), 320);
    }
  } catch (err) {
    if (welcomeStatus) {
      welcomeStatus.textContent = `Failed to start: ${err instanceof Error ? err.message : 'unknown'}`;
      welcomeStatus.classList.add('error');
    }
    if (synthBtn) synthBtn.disabled = false;
    if (realBtn) realBtn.disabled = false;
  }
}

mountDebugModal();

// v6.2 — vertical splitter between top and bottom canvases. Drag the
// thin bar to resize. The CSS exposes the top height via --top-h on
// the root; we set it in pixels during drag (clamped to keep both
// panels usable). After each move we manually trigger each sketch's
// resize() so p5 re-fits its canvas to the new container size.
{
  const splitter = document.getElementById('splitter');
  if (splitter) {
    let dragging = false;
    splitter.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      dragging = true;
      splitter.classList.add('dragging');
      document.body.style.cursor = 'row-resize';
    });
    window.addEventListener('mousemove', (ev) => {
      if (!dragging) return;
      const totalH = window.innerHeight;
      const splitterH = 6;
      const minTop = 220;
      const minBottom = 220;
      let topH = ev.clientY;
      topH = Math.max(minTop, Math.min(totalH - splitterH - minBottom, topH));
      document.documentElement.style.setProperty('--top-h', `${topH}px`);
      // Resize the canvases on each move so the drag feels live.
      top?.resize();
      bottom?.resize();
    });
    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      splitter.classList.remove('dragging');
      document.body.style.cursor = '';
      // One final resize after release in case the last move was clamped.
      top?.resize();
      bottom?.resize();
    });
  }
}

/**
 * Randomize semantics:
 *   - synthetic: regenerate the whole world with a fresh seed (new ref,
 *     new scenarios, new reads). The reference length stays at the
 *     synthetic default so the p5 ref-cache size remains valid.
 *   - real-bam: pick a random already-loaded candidate to snap the
 *     window to (the underlying world is fixed for the session).
 */
async function rerollWorld(handle: SketchHandle): Promise<void> {
  if (chosenWorldKind === 'synthetic') {
    const seed = Math.floor(Math.random() * 0x7fffffff) || 1;
    const fresh = await buildWorld({ kind: 'synthetic', seed });
    handle.setWorld(fresh);
  } else {
    handle.randomize();
  }
}

function attachUiHandlers(handle: SketchHandle): void {
  const resetBtn = document.getElementById('reset-view');
  resetBtn?.addEventListener('click', () => handle.resetView());

  const randomizeBtn = document.getElementById('randomize');
  randomizeBtn?.addEventListener('click', () => void rerollWorld(handle));

  const prevCandBtn = document.getElementById('prev-cand');
  prevCandBtn?.addEventListener('click', () => handle.prevCandidate());

  const nextCandBtn = document.getElementById('next-cand');
  nextCandBtn?.addEventListener('click', () => handle.nextCandidate());

  // v6.1: Auto-focus and Row sort toggles moved into the bottom canvas
  // (drawn beneath the active channel image, hit-tested in
  // bottom.ts mousePressed). HTML buttons removed.

  // Keyboard navigation: ←/→ for candidate stepping, 'r' for re-roll.
  // Skip when the user is typing in an input.
  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) {
      return;
    }
    if (ev.key === 'ArrowLeft') handle.prevCandidate();
    else if (ev.key === 'ArrowRight') handle.nextCandidate();
    else if (ev.key === 'r' || ev.key === 'R') void rerollWorld(handle);
  });
}

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
    if (!top) return;
    const info = top.hoverInfo(sx, sy);
    if (!info) {
      tooltip.style.display = 'none';
      // Clear any top-source hover so the bottom canvas crosshair fades.
      if (sandboxState.hover?.source === 'top') sandboxState.hover = null;
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

    // Mirror the hover into shared state so the bottom canvas can light
    // up the corresponding (col, row) crosshair on its active channel.
    // We only write if the hovered read is actually present in the
    // current encoder image (passed mapq + base-Q filters); otherwise
    // the linkage doesn't exist on the bottom side.
    const map = sandboxState.latestRowToReadId;
    if (map) {
      const imageRow = map.indexOf(info.readId);
      if (imageRow >= 0) {
        sandboxState.hover = {
          source: 'top',
          genomicPos: info.absCol,
          imageRow,
          readId: info.readId,
          cellValue: 0, // not sampled from this side
          channel: -1, // top doesn't track the active channel
        };
      } else if (sandboxState.hover?.source === 'top') {
        // The read is filtered out of the encoder image — clear stale
        // top-source hover.
        sandboxState.hover = null;
      }
    }
  });
  topEl.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
    if (sandboxState.hover?.source === 'top') sandboxState.hover = null;
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
