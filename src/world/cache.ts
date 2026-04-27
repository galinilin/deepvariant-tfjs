import type p5 from 'p5';

const TIER_DENSITIES = [1, 2, 4];
const TIER_UPPER_BOUNDS = [1, 2]; // < bounds[i] uses tier i; else next tier

export type Painter = (target: p5.Graphics) => void;

interface Tier {
  buffer: p5.Graphics;
  dirty: boolean;
}

/**
 * Multi-resolution offscreen cache. Stores up to 3 versions of the same
 * artwork at increasing pixel densities; picks the cheapest one that stays
 * crisp at the current camera zoom.
 *
 *   - Tiers are built lazily on first use.
 *   - invalidate() marks every existing tier dirty; the next get() rebuilds
 *     whichever tier is requested.
 *   - The painter closure receives a p5.Graphics target and renders the
 *     content at native (camera-zoom-independent) coordinates.
 */
export class TieredCache {
  private tiers: (Tier | null)[] = [null, null, null];

  constructor(
    private readonly p: p5,
    private readonly width: number,
    private readonly height: number,
    private painter: Painter,
  ) {}

  setPainter(painter: Painter): void {
    this.painter = painter;
    this.invalidate();
  }

  invalidate(): void {
    for (const t of this.tiers) {
      if (t) t.dirty = true;
    }
  }

  get(zoom: number): p5.Graphics {
    const idx = chooseTier(zoom);
    let tier = this.tiers[idx];
    if (!tier) {
      const buffer = this.p.createGraphics(this.width, this.height);
      buffer.pixelDensity(TIER_DENSITIES[idx]);
      tier = { buffer, dirty: true };
      this.tiers[idx] = tier;
    }
    if (tier.dirty) {
      tier.buffer.clear();
      this.painter(tier.buffer);
      tier.dirty = false;
    }
    return tier.buffer;
  }

  destroy(): void {
    for (const t of this.tiers) {
      if (t) t.buffer.remove();
    }
    this.tiers = [null, null, null];
  }
}

function chooseTier(zoom: number): number {
  for (let i = 0; i < TIER_UPPER_BOUNDS.length; i++) {
    if (zoom < TIER_UPPER_BOUNDS[i]) return i;
  }
  return TIER_UPPER_BOUNDS.length;
}
