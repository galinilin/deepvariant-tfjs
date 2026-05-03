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
const DELETION_COLOR: [number, number, number] = [165, 165, 165];
const DELETION_HEIGHT = 2;
const INSERTION_COLOR_RGB: [number, number, number] = [155, 110, 200];
const INSERTION_LABEL_RGB: [number, number, number] = [195, 145, 230];
const INSERTION_TICK_WIDTH = 3;
const INSERTION_TICK_OVERHANG = 2;
const INSERTION_LABEL_FONT = '600 8px Inconsolata, ui-monospace, Menlo, monospace';
const INSERTION_LABEL_GAP = 2;

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
  const delFill = `rgb(${DELETION_COLOR[0]},${DELETION_COLOR[1]},${DELETION_COLOR[2]})`;
  const radius = READ_BODY_HEIGHT / 2;

  for (let r = 0; r < reads.length; r++) {
    const read = reads[r];
    const rowYTop = read.row * READ_ROW_H;
    const rowYCenter = rowYTop + READ_ROW_H / 2;
    const bodyY = rowYCenter - READ_BODY_HEIGHT / 2;
    const bodyX = read.startCol * cellW;
    const bodyW = read.bases.length * cellW;

    // Body alpha encodes mapping quality: high mapq stays at ~0.45 (current),
    // low mapq fades the read body so the eye discounts its evidence.
    const mapqClamped = Math.min(Math.max(read.mapq, 0), 60);
    const bodyAlpha = 0.15 + 0.30 * (mapqClamped / 60);

    const strandColor =
      read.strand === 'forward' ? FORWARD_STRAND_COLOR : REVERSE_STRAND_COLOR;
    ctx.fillStyle = `rgba(${strandColor[0]},${strandColor[1]},${strandColor[2]},${bodyAlpha.toFixed(3)})`;
    ctx.beginPath();
    ctx.roundRect(bodyX, bodyY, bodyW, READ_BODY_HEIGHT, radius);
    ctx.fill();

    // Per-cell overlays: mismatches as bold colored letters dimmed by base
    // quality, deletions as a solid horizontal bar replacing the body.
    for (let i = 0; i < read.bases.length; i++) {
      const absCol = read.startCol + i;
      const cellX = absCol * cellW;
      const cellCenterX = cellX + cellW / 2;
      const base = read.bases[i];

      if (base === '-') {
        const delY = rowYCenter - DELETION_HEIGHT / 2;
        ctx.fillStyle = delFill;
        ctx.fillRect(cellX, delY, cellW, DELETION_HEIGHT);
        continue;
      }

      const refBase = reference[absCol] ?? 'N';
      if (base !== refBase) {
        const c = igvColor(base);
        const q = Math.min(Math.max(read.qualities[i] ?? 0, 0), 40);
        const letterAlpha = 0.40 + 0.60 * (q / 40);
        ctx.font = boldFont;
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${letterAlpha.toFixed(3)})`;
        ctx.fillText(base, cellCenterX, rowYCenter);
      }
    }

    // Insertion ticks: drawn at the boundary between bases[offset] and
    // bases[offset+1], straddling the column edge. Alpha tracks mapq so the
    // visual weight matches the read body. A small "+N" length label sits
    // just above the tick (IGV convention) so length is visible at-a-glance.
    const insertions = read.insertions;
    if (insertions) {
      const tickAlpha = Math.min(1, bodyAlpha + 0.45);
      const insertionFill = `rgba(${INSERTION_COLOR_RGB[0]},${INSERTION_COLOR_RGB[1]},${INSERTION_COLOR_RGB[2]},${tickAlpha.toFixed(3)})`;
      const tickY = bodyY - INSERTION_TICK_OVERHANG;
      const tickH = READ_BODY_HEIGHT + INSERTION_TICK_OVERHANG * 2;
      ctx.fillStyle = insertionFill;
      for (const ins of insertions) {
        const tickCenterX = (read.startCol + ins.offset + 1) * cellW;
        ctx.fillRect(
          tickCenterX - INSERTION_TICK_WIDTH / 2,
          tickY,
          INSERTION_TICK_WIDTH,
          tickH,
        );
      }
      ctx.font = INSERTION_LABEL_FONT;
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgba(${INSERTION_LABEL_RGB[0]},${INSERTION_LABEL_RGB[1]},${INSERTION_LABEL_RGB[2]},${tickAlpha.toFixed(3)})`;
      const labelX0Offset =
        INSERTION_TICK_WIDTH / 2 + INSERTION_LABEL_GAP;
      for (const ins of insertions) {
        const tickCenterX = (read.startCol + ins.offset + 1) * cellW;
        ctx.fillText(`+${ins.bases.length}`, tickCenterX + labelX0Offset, rowYCenter);
      }
      ctx.textAlign = 'center';
    }
  }
}
