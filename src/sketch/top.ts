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
  WINDOW_LENGTH,
  buildReference,
  clampWindowStart,
  defaultWindowStart,
} from '../lib/reference';

export interface SketchHandle {
  resetView: () => void;
  destroy: () => void;
}

const REF_GAP = 80;

export function mountTopSketch(container: HTMLElement): SketchHandle {
  let resetFn: () => void = () => {};

  const size = () => ({
    w: container.clientWidth,
    h: container.clientHeight,
  });

  const instance = new p5((p: p5) => {
    const cam = new Camera();
    const reference = buildReference();
    let windowStart = defaultWindowStart(reference.length);

    const refWidth = REF_COUNT * CELL_W;
    const scrubberWidth = refWidth * 0.8;
    const scrubberOrigin = { x: (refWidth - scrubberWidth) / 2, y: 0 };
    const refOrigin = { x: 0, y: SCRUBBER_HEIGHT + REF_GAP };
    const rulerOrigin = { x: 0, y: refOrigin.y + CELL_H };
    const totalHeight = SCRUBBER_HEIGHT + REF_GAP + CELL_H + RULER_HEIGHT;

    let dragMode: 'pan' | 'scrubber' = 'pan';

    let refCache: TieredCache | null = null;

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
      refCache?.invalidate();
    };

    const initializeView = () => {
      const { w, h } = size();
      const fitZoom = Math.min(1, (w - 220) / refWidth);
      cam.zoom = Math.max(cam.minZoom, fitZoom);
      const cw = refWidth * cam.zoom;
      const ch = totalHeight * cam.zoom;
      cam.x = (w - cw) / 2;
      cam.y = (h - ch) / 2;
      windowStart = defaultWindowStart(reference.length);
      refCache?.invalidate();
    };

    resetFn = () => initializeView();

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

    p.setup = () => {
      const { w, h } = size();
      p.createCanvas(w, h);
      p.pixelDensity(p.displayDensity());
      p.textFont('Inconsolata');
      refCache = new TieredCache(p, refWidth, CELL_H, (target) =>
        paintRefBases(target, reference, windowStart),
      );
      initializeView();
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
        p.image(refCache.get(cam.zoom), refOrigin.x, refOrigin.y, refWidth, CELL_H);
      }
      drawRuler(p, {
        windowStart,
        origin: rulerOrigin,
        cellWidth: CELL_W,
        zoom: cam.zoom,
      });
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
