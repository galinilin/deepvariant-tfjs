import p5 from 'p5';
import { sandboxState } from '../lib/sandbox-state';

export interface BottomHandle {
  destroy: () => void;
}

const LABEL_COLOR: [number, number, number] = [210, 210, 210];
const BORDER_COLOR: [number, number, number] = [48, 48, 48];
const PLACEHOLDER_COLOR: [number, number, number] = [90, 90, 90];
const NO_CANDIDATE_COLOR: [number, number, number] = [120, 120, 120];

const MARGIN = 40;
const GAP = 40;
const HEADER_GAP = 8;

export function mountBottomSketch(container: HTMLElement): BottomHandle {
  const size = () => ({
    w: container.clientWidth,
    h: container.clientHeight,
  });

  const instance = new p5((p: p5) => {
    p.setup = () => {
      const { w, h } = size();
      p.createCanvas(w, h);
      p.pixelDensity(p.displayDensity());
      p.textFont('Inconsolata');
    };

    p.windowResized = () => {
      const { w, h } = size();
      p.resizeCanvas(w, h);
    };

    p.draw = () => {
      p.background(0);
      drawPlaceholders(p);
    };
  }, container);

  return {
    destroy: () => instance.remove(),
  };
}

function drawPlaceholders(p: p5): void {
  const innerW = p.width - MARGIN * 2 - GAP;
  const innerH = p.height - MARGIN * 2;
  const leftW = Math.round(innerW * 0.74);
  const rightW = innerW - leftW;

  drawPanel(p, MARGIN, MARGIN, leftW, innerH, 'Pileup Representation');
  drawPanel(p, MARGIN + leftW + GAP, MARGIN, rightW, innerH, 'Prediction');
}

function drawPanel(
  p: p5,
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
): void {
  p.noStroke();
  p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
  p.textSize(14);
  p.textAlign(p.LEFT, p.BOTTOM);
  p.text(label, x, y - HEADER_GAP);

  p.noFill();
  p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
  p.strokeWeight(0.5);
  p.rect(x, y, w, h);

  const noCandidate = sandboxState.candidate === null;
  const fill = noCandidate ? NO_CANDIDATE_COLOR : PLACEHOLDER_COLOR;
  const text = noCandidate ? 'No candidate' : 'placeholder';

  p.noStroke();
  p.fill(fill[0], fill[1], fill[2]);
  p.textSize(noCandidate ? 13 : 11);
  p.textAlign(p.CENTER, p.CENTER);
  p.text(text, x + w / 2, y + h / 2);
}
