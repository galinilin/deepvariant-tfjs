import type p5 from 'p5';

export interface Point {
  x: number;
  y: number;
}

export class Camera {
  x = 0;
  y = 0;
  zoom = 1;
  // Tightened from 0.3..3 → 0.6..1.6. Anything outside this range tends
  // to either render text far smaller than legible or zoom past where the
  // pileup is meaningful.
  minZoom = 0.6;
  maxZoom = 1.6;

  apply(p: p5): void {
    p.translate(this.x, this.y);
    p.scale(this.zoom);
  }

  screenToWorld(sx: number, sy: number): Point {
    return {
      x: (sx - this.x) / this.zoom,
      y: (sy - this.y) / this.zoom,
    };
  }

  pan(dx: number, dy: number): void {
    this.x += dx;
    this.y += dy;
  }

  zoomAt(sx: number, sy: number, factor: number): void {
    const next = this.zoom * factor;
    if (next < this.minZoom || next > this.maxZoom) return;
    this.x = sx - (sx - this.x) * factor;
    this.y = sy - (sy - this.y) * factor;
    this.zoom = next;
  }
}
