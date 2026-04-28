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
  WINDOW_LENGTH,
  buildReference,
  clampWindowStart,
  defaultWindowStart,
} from '../lib/reference';
import { buildReads } from '../lib/reads';

export interface SketchHandle {
  resetView: () => void;
  destroy: () => void;
}

const REF_GAP = 80;
const READS_GAP = 18;

export function mountTopSketch(container: HTMLElement): SketchHandle {
  let resetFn: () => void = () => {};

  const size = () => ({
    w: container.clientWidth,
    h: container.clientHeight,
  });

  const instance = new p5((p: p5) => {
    const cam = new Camera();
    const reference = buildReference();
    const reads = buildReads(reference);
    let windowStart = defaultWindowStart(reference.length);

    const refWidth = REF_COUNT * CELL_W;
    const fullPileupWidth = reference.length * CELL_W;
    const readsHeight = reads.length * READ_ROW_H;
    const scrubberWidth = refWidth * 0.8;

    const scrubberOrigin = { x: (refWidth - scrubberWidth) / 2, y: 0 };
    const refOrigin = { x: 0, y: SCRUBBER_HEIGHT + REF_GAP };
    const rulerOrigin = { x: 0, y: refOrigin.y + CELL_H };
    const readsOrigin = { x: 0, y: rulerOrigin.y + RULER_HEIGHT + READS_GAP };
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
    });

    const setWindow = (next: number) => {
      const clamped = clampWindowStart(next, reference.length);
      if (clamped === windowStart) return;
      windowStart = clamped;
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
    };

    resetFn = () => initializeView();

    const buildReadsCache = () => {
      if (!readsCache) {
        readsCache = p.createGraphics(fullPileupWidth, readsHeight);
        readsCache.pixelDensity(2);
        readsCache.textFont('Inconsolata');
      }
      readsCache.clear();
      paintReadsBases(readsCache, reads, CELL_W);
    };

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
      const colIdx = Math.floor(WINDOW_LENGTH / 2); // 110
      const x = refOrigin.x + colIdx * CELL_W + CELL_W / 2;
      const margin = 6 / cam.zoom;
      const top = refOrigin.y - margin;
      const bottom = readsOrigin.y + readsHeight + margin;

      // Vertical guide line
      p.stroke(255, 220, 110, 120);
      p.strokeWeight(1 / cam.zoom);
      p.line(x, top, x, bottom);

      // Top-down arrow: tip at the top of the line, base above
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

      // If Inconsolata wasn't loaded when caches were first built, rebuild
      // them once the font is ready so letters render in Inconsolata.
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
      p.background(0);
      p.push();
      cam.apply(p);

      drawScrubber(p, scrubberState());
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
        readsCount: reads.length,
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

      drawPredictMarker();

      p.pop();
    };

    p.mousePressed = () => {
      const wp = cam.screenToWorld(p.mouseX, p.mouseY);
      if (isInsideScrubber(scrubberState(), wp.x, wp.y)) {
        dragMode = 'scrubber';
        setWindow(windowStartFromWorldX(scrubberState(), wp.x));
      } else {
        dragMode = 'pan';
      }
    };

    p.mouseDragged = () => {
      if (dragMode === 'scrubber') {
        const wp = cam.screenToWorld(p.mouseX, p.mouseY);
        setWindow(windowStartFromWorldX(scrubberState(), wp.x));
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

  return {
    resetView: () => resetFn(),
    destroy: () => instance.remove(),
  };
}
