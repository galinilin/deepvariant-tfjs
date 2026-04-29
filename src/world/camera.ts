import type p5 from 'p5';

export interface Point {
  x: number;
  y: number;
}

export class Camera {
  x = 0;
  y = 0;
  zoom = 1;
  minZoom = 0.3;
  maxZoom = 3;

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
