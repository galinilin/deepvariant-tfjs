import p5 from 'p5';
import { Camera } from '../world/camera';
import { drawRef, REF_COUNT, CELL_W, CELL_H } from '../world/regions/ref';
import {
  SCRUBBER_HEIGHT,
  drawScrubber,
  isInsideScrubber,
  windowStartFromWorldX,
} from '../world/regions/scrubber';
import {
  buildReference,
  clampWindowStart,
  defaultWindowStart,
} from '../lib/reference';

export interface SketchHandle {
  resetView: () => void;
  destroy: () => void;
}

const REF_GAP = 28;

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

    const contentWidth = REF_COUNT * CELL_W;
    const scrubberOrigin = { x: 0, y: 0 };
    const refOrigin = { x: 0, y: SCRUBBER_HEIGHT + REF_GAP };
    const totalHeight = SCRUBBER_HEIGHT + REF_GAP + CELL_H;

    let dragMode: 'pan' | 'scrubber' = 'pan';

    const scrubberState = () => ({
      reference,
      windowStart,
      origin: scrubberOrigin,
      width: contentWidth,
    });

    const setWindow = (next: number) => {
      windowStart = clampWindowStart(next, reference.length);
    };

    const initializeView = () => {
      const { w, h } = size();
      const fitZoom = Math.min(1, (w - 80) / contentWidth);
      cam.zoom = Math.max(cam.minZoom, fitZoom);
      const cw = contentWidth * cam.zoom;
      const ch = totalHeight * cam.zoom;
      cam.x = (w - cw) / 2;
      cam.y = (h - ch) / 2;
      windowStart = defaultWindowStart(reference.length);
    };

    resetFn = initializeView;

    p.setup = () => {
      const { w, h } = size();
      p.createCanvas(w, h);
      p.pixelDensity(p.displayDensity());
      p.textFont('Inconsolata');
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
      drawRef(p, { reference, windowStart, origin: refOrigin });
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
