import p5 from 'p5';
import { Camera } from '../world/camera';
import { TieredCache } from '../world/cache';
import {
  drawRefFrame,
  paintRefBases,
  REF_COUNT,
  CELL_W,
  CELL_H,
} from '../world/regions/ref';
import {
  SCRUBBER_HEIGHT,
  drawScrubber,
  isInsideScrubber,
  windowStartFromWorldX,
} from '../world/regions/scrubber';
import { RULER_HEIGHT, drawRuler } from '../world/regions/ruler';
import {
  READ_ROW_H,
  drawReadsFrame,
  paintReadsBases,
} from '../world/regions/reads';
import {
  DEFAULT_REFERENCE_LENGTH,
  WINDOW_LENGTH,
  buildReference,
  clampWindowStart,
  defaultWindowStart,
} from '../lib/reference';
import { buildReads, makeRng, MAX_PACKED_ROWS, type Read } from '../lib/reads';
import { placeScenarios, type Scenario } from '../lib/scenarios';
import {
  deriveCandidateOutcome,
  formatAlt,
  readSupportsCandidate,
  type Candidate,
  type CandidateOutcome,
} from '../lib/candidate';
import { sandboxState } from '../lib/sandbox-state';
import { hitTestReads } from '../world/hit-test';
import type { Base, Cell } from '../lib/palette';
import type { Strand } from '../lib/reads';

interface HoverInfoBase {
  readId: string;
  startCol: number;
  endCol: number;
  strand: Strand;
  mapq: number;
  insertSize: number;
  absCol: number;
  isPredictColumn: boolean;
  supportsCandidate: boolean | null;
  candidate: Candidate;
  outcome: CandidateOutcome;
}

export type HoverInfo =
  | (HoverInfoBase & { kind: 'cell'; base: Cell; quality: number })
  | (HoverInfoBase & {
      kind: 'insertion';
      sequence: Base[];
      qualities: Uint8Array;
    });

export interface SketchHandle {
  resetView: () => void;
  randomize: () => void;
  hoverInfo: (sx: number, sy: number) => HoverInfo | null;
  destroy: () => void;
}

const REF_GAP = 80;
const READS_GAP = 18;
const INITIAL_SEED = 42;

export function mountTopSketch(container: HTMLElement): SketchHandle {
  let resetFn: () => void = () => {};
  let randomizeFn: () => void = () => {};
  let camRef: Camera | null = null;
  let readsRef: Read[] = [];
  let referenceRef: Base[] = [];
  const readsOriginRef = { x: 0, y: 0 };
  const windowStartRef = { value: 0 };

  const size = () => ({
    w: container.clientWidth,
    h: container.clientHeight,
  });

  const instance = new p5((p: p5) => {
    const cam = new Camera();
    camRef = cam;

    let reference: Base[] = [];
    let scenarios: Scenario[] = [];
    let reads: Read[] = [];
    let windowStart = 0;

    const generateWorld = (seed: number) => {
      const rng = makeRng(seed);
      reference = buildReference(DEFAULT_REFERENCE_LENGTH, seed);
      scenarios = placeScenarios(reference, rng);
      reads = buildReads(reference, scenarios, rng);
      referenceRef = reference;
      readsRef = reads;
    };

    generateWorld(INITIAL_SEED);
    windowStart = defaultWindowStart(reference.length);
    windowStartRef.value = windowStart;

    const refWidth = REF_COUNT * CELL_W;
    const fullPileupWidth = reference.length * CELL_W;
    const readsHeight = MAX_PACKED_ROWS * READ_ROW_H;
    const scrubberWidth = refWidth * 0.8;

    const scrubberOrigin = { x: (refWidth - scrubberWidth) / 2, y: 0 };
    const refOrigin = { x: 0, y: SCRUBBER_HEIGHT + REF_GAP };
    const rulerOrigin = { x: 0, y: refOrigin.y + CELL_H };
    const readsOrigin = { x: 0, y: rulerOrigin.y + RULER_HEIGHT + READS_GAP };
    readsOriginRef.x = readsOrigin.x;
    readsOriginRef.y = readsOrigin.y;
    const totalHeight =
      SCRUBBER_HEIGHT + REF_GAP + CELL_H + RULER_HEIGHT + READS_GAP + readsHeight;

    let dragMode: 'pan' | 'scrubber' = 'pan';
    let refCache: TieredCache | null = null;
    let readsCache: p5.Graphics | null = null;

    const scrubberState = () => ({
      reference,
      windowStart,
      origin: scrubberOrigin,
      width: scrubberWidth,
      scenarios,
    });

    const setWindow = (next: number) => {
      const clamped = clampWindowStart(next, reference.length);
      if (clamped === windowStart) return;
      windowStart = clamped;
      windowStartRef.value = windowStart;
    };

    const snapWindowToScenarios = (candidateStart: number): number => {
      if (scenarios.length === 0) return candidateStart;
      const candidateCenter = candidateStart + WINDOW_LENGTH / 2;
      const snapThresholdBp =
        (8 / cam.zoom) * (reference.length / scrubberWidth);
      let bestStart = candidateStart;
      let bestDistance = snapThresholdBp;
      for (const sc of scenarios) {
        const distance = Math.abs(sc.position - candidateCenter);
        if (distance <= bestDistance) {
          bestDistance = distance;
          bestStart = sc.position - Math.floor(WINDOW_LENGTH / 2);
        }
      }
      return bestStart;
    };

    const handleScrubberWorldX = (wx: number) => {
      const candidate = windowStartFromWorldX(scrubberState(), wx);
      setWindow(snapWindowToScenarios(candidate));
    };

    const initializeView = () => {
      const { w, h } = size();
      const fitZoomW = (w - 220) / refWidth;
      const fitZoomH = (h - 80) / totalHeight;
      const fitZoom = Math.min(1, Math.min(fitZoomW, fitZoomH));
      cam.zoom = Math.max(cam.minZoom, fitZoom);
      const cw = refWidth * cam.zoom;
      const ch = totalHeight * cam.zoom;
      cam.x = (w - cw) / 2;
      cam.y = (h - ch) / 2;
      windowStart = defaultWindowStart(reference.length);
      windowStartRef.value = windowStart;
    };

    resetFn = () => initializeView();

    const buildReadsCache = () => {
      if (!readsCache) {
        readsCache = p.createGraphics(fullPileupWidth, readsHeight);
        readsCache.pixelDensity(2);
        readsCache.textFont('Inconsolata');
      }
      readsCache.clear();
      paintReadsBases(readsCache, reads, reference, CELL_W);
    };

    const randomize = () => {
      const seed = Math.floor(Math.random() * 0x7fffffff) || 1;
      generateWorld(seed);

      if (scenarios.length > 0) {
        const pick = scenarios[Math.floor(Math.random() * scenarios.length)];
        windowStart = clampWindowStart(
          pick.position - Math.floor(WINDOW_LENGTH / 2),
          reference.length,
        );
      } else {
        windowStart = defaultWindowStart(reference.length);
      }
      windowStartRef.value = windowStart;

      refCache?.invalidate();
      buildReadsCache();

      // Bump the generation counter so the bottom canvas knows to
      // invalidate any cached prediction tied to the previous reads set.
      sandboxState.readsGeneration += 1;
    };

    randomizeFn = randomize;

    const drawWindowFunnel = () => {
      const winLeft =
        scrubberOrigin.x + (windowStart / reference.length) * scrubberWidth;
      const winRight =
        scrubberOrigin.x +
        ((windowStart + WINDOW_LENGTH) / reference.length) * scrubberWidth;
      const top = scrubberOrigin.y + SCRUBBER_HEIGHT;
      const bottom = refOrigin.y;
      const refLeft = refOrigin.x;
      const refRight = refOrigin.x + REF_COUNT * CELL_W;

      p.noFill();
      p.stroke(220, 220, 220, 180);
      p.strokeWeight(1.4 / cam.zoom);
      p.line(winLeft, top, refLeft, bottom);
      p.line(winRight, top, refRight, bottom);
    };

    const drawPredictMarker = () => {
      const colIdx = Math.floor(WINDOW_LENGTH / 2);
      const x = refOrigin.x + colIdx * CELL_W + CELL_W / 2;
      const margin = 6 / cam.zoom;
      const top = refOrigin.y - margin;
      const bottom = readsOrigin.y + readsHeight + margin;

      // Connector from window center on scrubber down to the predict line top
      const winCenterX =
        scrubberOrigin.x +
        ((windowStart + WINDOW_LENGTH / 2) / reference.length) * scrubberWidth;
      const winBottom = scrubberOrigin.y + SCRUBBER_HEIGHT;

      p.stroke(255, 220, 110, 120);
      p.strokeWeight(1 / cam.zoom);
      p.line(winCenterX, winBottom, x, top);

      // Vertical predict line through Ref + Reads
      p.line(x, top, x, bottom);

      // Top-down arrow at the line's top
      const triHalfW = 5 / cam.zoom;
      const triH = 7 / cam.zoom;
      p.noStroke();
      p.fill(255, 220, 110, 200);
      p.triangle(
        x - triHalfW,
        top - triH,
        x + triHalfW,
        top - triH,
        x,
        top,
      );
    };

    const drawSupportsMarkers = (candidate: Candidate, predictPos: number) => {
      if (!candidate) return;
      const dotX = refOrigin.x + Math.floor(WINDOW_LENGTH / 2) * CELL_W + CELL_W / 2;
      const dotDiameter = 3 / cam.zoom;
      p.noStroke();
      p.fill(255, 220, 110, 235);
      for (const read of reads) {
        if (read.row >= MAX_PACKED_ROWS) continue;
        if (!readSupportsCandidate(read, predictPos, candidate)) continue;
        const dotY =
          readsOrigin.y + read.row * READ_ROW_H + READ_ROW_H - 3 / cam.zoom;
        p.circle(dotX, dotY, dotDiameter);
      }
    };

    const drawPredictLabel = (predictX: number, outcome: CandidateOutcome) => {
      const margin = 6 / cam.zoom;
      const triH = 7 / cam.zoom;
      const labelY = refOrigin.y - margin - triH - 6 / cam.zoom;

      let text: string;
      let r = 255;
      let g = 220;
      let b = 110;
      let alpha = 230;

      switch (outcome.kind) {
        case 'accepted': {
          const altLabel = formatAlt(outcome.info);
          text = `${outcome.info.refBase}\u2192${altLabel}  ${outcome.info.supportingReads}/${outcome.info.qualifyingReads}`;
          break;
        }
        case 'no-coverage':
          text = 'no reads';
          r = 140; g = 140; b = 140; alpha = 200;
          break;
        case 'no-alt-evidence':
          text = `all ref (${outcome.qualifyingReads})`;
          r = 140; g = 140; b = 140; alpha = 200;
          break;
        case 'below-count': {
          const altLabel = formatAlt(outcome.alt);
          text = `${altLabel} ${outcome.count}\u00d7 (need 2)`;
          r = 200; g = 180; b = 110; alpha = 200;
          break;
        }
        case 'below-fraction': {
          const altLabel = formatAlt(outcome.alt);
          const pct = ((outcome.count / outcome.qualifyingReads) * 100).toFixed(1);
          const minPct = outcome.alt.kind === 'snv' ? 12 : 6;
          text = `${altLabel} ${pct}% (need ${minPct}%)`;
          r = 200; g = 180; b = 110; alpha = 200;
          break;
        }
      }

      p.noStroke();
      p.fill(r, g, b, alpha);
      p.textFont('Inconsolata');
      p.textStyle(p.BOLD);
      p.textSize(14 / cam.zoom);
      p.textAlign(p.CENTER, p.BOTTOM);
      p.text(text, predictX, labelY);
    };

    p.setup = () => {
      const { w, h } = size();
      p.createCanvas(w, h);
      p.pixelDensity(p.displayDensity());
      p.textFont('Inconsolata');
      refCache = new TieredCache(p, fullPileupWidth, CELL_H, (target) =>
        paintRefBases(target, reference),
      );
      buildReadsCache();
      initializeView();

      void document.fonts.ready.then(() => {
        refCache?.invalidate();
        buildReadsCache();
      });
    };

    p.windowResized = () => {
      const { w, h } = size();
      p.resizeCanvas(w, h);
    };

    p.draw = () => {
      const predictPos = windowStart + Math.floor(WINDOW_LENGTH / 2);
      const outcome = deriveCandidateOutcome(reads, reference, predictPos);
      const candidate: Candidate =
        outcome.kind === 'accepted' ? outcome.info : null;
      // Publish the full state the bottom canvas needs to encode + predict.
      sandboxState.candidate = candidate;
      sandboxState.reads = reads;
      sandboxState.reference = reference;
      sandboxState.predictPos = predictPos;
      const predictX =
        refOrigin.x + Math.floor(WINDOW_LENGTH / 2) * CELL_W + CELL_W / 2;

      p.background(0);
      p.push();
      cam.apply(p);

      drawScrubber(p, scrubberState(), cam.zoom);
      drawWindowFunnel();

      drawRefFrame(p, { origin: refOrigin });
      if (refCache) {
        p.image(
          refCache.get(cam.zoom),
          refOrigin.x,
          refOrigin.y,
          refWidth,
          CELL_H,
          windowStart * CELL_W,
          0,
          WINDOW_LENGTH * CELL_W,
          CELL_H,
        );
      }

      drawRuler(p, {
        windowStart,
        origin: rulerOrigin,
        cellWidth: CELL_W,
        zoom: cam.zoom,
      });

      drawReadsFrame(p, {
        origin: readsOrigin,
        readsCount: MAX_PACKED_ROWS,
        width: refWidth,
      });
      if (readsCache) {
        p.image(
          readsCache,
          readsOrigin.x,
          readsOrigin.y,
          refWidth,
          readsHeight,
          windowStart * CELL_W,
          0,
          WINDOW_LENGTH * CELL_W,
          readsHeight,
        );
      }

      drawSupportsMarkers(candidate, predictPos);
      drawPredictMarker();
      drawPredictLabel(predictX, outcome);

      p.pop();
    };

    p.mousePressed = () => {
      const wp = cam.screenToWorld(p.mouseX, p.mouseY);
      if (isInsideScrubber(scrubberState(), wp.x, wp.y)) {
        dragMode = 'scrubber';
        handleScrubberWorldX(wp.x);
      } else {
        dragMode = 'pan';
      }
    };

    p.mouseDragged = () => {
      if (dragMode === 'scrubber') {
        const wp = cam.screenToWorld(p.mouseX, p.mouseY);
        handleScrubberWorldX(wp.x);
      } else {
        cam.pan(p.movedX, p.movedY);
      }
    };

    p.mouseWheel = (event?: WheelEvent) => {
      if (!event) return;
      event.preventDefault();
      const factor = event.deltaY > 0 ? 0.9 : 1.1;
      cam.zoomAt(p.mouseX, p.mouseY, factor);
    };
  }, container);

  const hoverInfo = (sx: number, sy: number): HoverInfo | null => {
    if (!camRef || !readsRef) return null;
    const wp = camRef.screenToWorld(sx, sy);
    const hit = hitTestReads(
      wp.x,
      wp.y,
      readsRef,
      readsOriginRef,
      WINDOW_LENGTH,
      CELL_W,
      READ_ROW_H,
      MAX_PACKED_ROWS,
      windowStartRef.value,
    );
    if (!hit) return null;
    const predictPos = windowStartRef.value + Math.floor(WINDOW_LENGTH / 2);
    const outcome = deriveCandidateOutcome(readsRef, referenceRef, predictPos);
    const candidate: Candidate =
      outcome.kind === 'accepted' ? outcome.info : null;
    // For an insertion hit, "predict column" means the candidate is an
    // insertion anchored at the same offset (right after this base).
    const isPredictColumn = hit.absCol === predictPos;
    const supportsCandidate =
      isPredictColumn && candidate
        ? readSupportsCandidate(hit.read, predictPos, candidate)
        : null;
    const base: HoverInfoBase = {
      readId: hit.read.id,
      startCol: hit.read.startCol,
      endCol: hit.read.startCol + hit.read.bases.length - 1,
      strand: hit.read.strand,
      mapq: hit.read.mapq,
      insertSize: hit.read.insertSize,
      absCol: hit.absCol,
      isPredictColumn,
      supportsCandidate,
      candidate,
      outcome,
    };
    if (hit.kind === 'insertion') {
      return {
        ...base,
        kind: 'insertion',
        sequence: hit.sequence,
        qualities: hit.qualities,
      };
    }
    return {
      ...base,
      kind: 'cell',
      base: hit.base,
      quality: hit.quality,
    };
  };

  return {
    resetView: () => resetFn(),
    randomize: () => randomizeFn(),
    hoverInfo,
    destroy: () => instance.remove(),
  };
}
