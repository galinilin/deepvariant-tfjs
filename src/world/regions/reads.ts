import type p5 from 'p5';
import { igvColor, type Base } from '../../lib/palette';
import type { Read } from '../../lib/reads';

export const READ_ROW_H = 18;

const LABEL_GAP = 12;
const LABEL_SIZE = 14;
const BASE_SIZE = 14;
const READ_BODY_HEIGHT = 7;

const LABEL_COLOR: [number, number, number] = [210, 210, 210];
const BORDER_COLOR: [number, number, number] = [48, 48, 48];
const PANEL_COLOR: [number, number, number] = [10, 10, 12];
const FORWARD_STRAND_COLOR: [number, number, number] = [105, 112, 130];
const REVERSE_STRAND_COLOR: [number, number, number] = [130, 112, 105];
const DELETION_COLOR: [number, number, number] = [105, 105, 105];

export interface ReadsState {
  origin: { x: number; y: number };
  readsCount: number;
  width: number;
}

export function drawReadsFrame(p: p5, state: ReadsState): void {
  const { x, y } = state.origin;
  const w = state.width;
  const h = state.readsCount * READ_ROW_H;

  // Subtle panel background for visual anchoring
  p.noStroke();
  p.fill(PANEL_COLOR[0], PANEL_COLOR[1], PANEL_COLOR[2]);
  p.rect(x, y, w, h);

  // Label
  p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
  p.textFont('Inconsolata');
  p.textStyle(p.NORMAL);
  p.textSize(LABEL_SIZE);
  p.textAlign(p.RIGHT, p.TOP);
  p.text('Reads', x - LABEL_GAP, y);

  // Border
  p.noFill();
  p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
  p.strokeWeight(0.5);
  p.rect(x, y, w, h);
}

/**
 * Paints all reads at their absolute genomic positions onto the target,
 * IGV-style: matches render as a thin gray "read body" bar, mismatches
 * render the colored letter on top, deletions render a dim '-'.
 *
 * The target buffer is expected to be wide enough to hold the full
 * reference length (i.e., reference.length * cellW).
 *
 * Uses the underlying CanvasRenderingContext2D directly because p5 v2's
 * textAlign() does not apply reliably to p5.Graphics buffers.
 */
export function paintReadsBases(
  target: p5 | p5.Graphics,
  reads: Read[],
  reference: Base[],
  cellW: number,
): void {
  const ctx = target.drawingContext as CanvasRenderingContext2D;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const boldFont = `600 ${BASE_SIZE}px Inconsolata, ui-monospace, Menlo, monospace`;
  const regularFont = `${BASE_SIZE}px Inconsolata, ui-monospace, Menlo, monospace`;
  const delFill = `rgb(${DELETION_COLOR[0]},${DELETION_COLOR[1]},${DELETION_COLOR[2]})`;
  const radius = READ_BODY_HEIGHT / 2;

  for (let r = 0; r < reads.length; r++) {
    const read = reads[r];
    const rowYTop = r * READ_ROW_H;
    const rowYCenter = rowYTop + READ_ROW_H / 2;
    const bodyY = rowYCenter - READ_BODY_HEIGHT / 2;
    const bodyX = read.startCol * cellW;
    const bodyW = read.bases.length * cellW;

    // One rounded "capsule" per read, strand-tinted, semi-transparent so
    // mismatch letters dominate the visual hierarchy.
    const strandColor =
      read.strand === 'forward' ? FORWARD_STRAND_COLOR : REVERSE_STRAND_COLOR;
    ctx.fillStyle = `rgba(${strandColor[0]},${strandColor[1]},${strandColor[2]},0.45)`;
    ctx.beginPath();
    ctx.roundRect(bodyX, bodyY, bodyW, READ_BODY_HEIGHT, radius);
    ctx.fill();

    // Per-cell overlays: mismatches as bold colored letters, deletions as '-'.
    for (let i = 0; i < read.bases.length; i++) {
      const absCol = read.startCol + i;
      const cellCenterX = absCol * cellW + cellW / 2;
      const base = read.bases[i];

      if (base === '-') {
        ctx.font = regularFont;
        ctx.fillStyle = delFill;
        ctx.fillText('-', cellCenterX, rowYCenter);
        continue;
      }

      const refBase = reference[absCol] ?? 'N';
      if (base !== refBase) {
        const c = igvColor(base);
        ctx.font = boldFont;
        ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
        ctx.fillText(base, cellCenterX, rowYCenter);
      }
    }
  }
}
