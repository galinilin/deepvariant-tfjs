import * as tf from '@tensorflow/tfjs';

export const DV_INPUT_SHAPE = [100, 221, 7] as const;
export const DV_CLASSES = ['hom_ref', 'het', 'hom_alt'] as const;
export type Genotype = (typeof DV_CLASSES)[number];

const SAMPLES_PER_PILEUP = DV_INPUT_SHAPE[0] * DV_INPUT_SHAPE[1] * DV_INPUT_SHAPE[2];

export interface PredictionResult {
  probs: Record<Genotype, number>;
  argmax: Genotype;
  confidence: number;
}

export type LoadStage = 'fetching' | 'warming' | 'ready';

export interface LoadOptions {
  precision?: 'float32' | 'uint8';
  modelBaseUrl?: string;
  /** Download progress (0..1) — fired during weight-shard fetches. */
  onProgress?: (fraction: number) => void;
  /** Phase transitions: 'fetching' → 'warming' → 'ready'. The warming
   * phase compiles ~70 WebGL conv shaders by running one zero-tensor
   * predict; without this the first user-triggered predict eats
   * 200–1000 ms of shader compilation. */
  onStage?: (stage: LoadStage) => void;
}

export interface BatchInput {
  data: Float32Array;
  batch: number;
}

export class DeepVariantModel {
  static readonly INPUT_SHAPE = DV_INPUT_SHAPE;
  static readonly CLASSES = DV_CLASSES;

  private constructor(
    private readonly model: tf.LayersModel,
    readonly precision: 'float32' | 'uint8',
    readonly backend: string,
  ) {}

  static async load(opts: LoadOptions = {}): Promise<DeepVariantModel> {
    const precision = opts.precision ?? 'uint8';
    const base = (opts.modelBaseUrl ?? '/models/').replace(/\/?$/, '/');
    const dir = precision === 'uint8' ? 'tfjs_dv_wgs_uint8' : 'tfjs_dv_wgs';
    await tf.ready();
    opts.onStage?.('fetching');
    const model = await tf.loadLayersModel(`${base}${dir}/model.json`, {
      onProgress: opts.onProgress,
    });

    // Yield to the event loop so the welcome status update from the last
    // onProgress callback gets a chance to repaint before the warmup
    // freezes the main thread again.
    opts.onStage?.('warming');
    await new Promise((r) => setTimeout(r, 16));

    // Warmup: run one inference on a zero tensor. Forces every WebGL
    // shader for InceptionV3's ~70 conv ops to compile NOW (still on the
    // welcome screen, before any user-visible interaction). After this
    // the first user-triggered predict is ~10× faster.
    const warmInput = tf.zeros([1, ...DV_INPUT_SHAPE]);
    const warmOutput = model.predict(warmInput) as tf.Tensor;
    await warmOutput.data();
    warmInput.dispose();
    warmOutput.dispose();

    opts.onStage?.('ready');
    return new DeepVariantModel(model, precision, tf.getBackend());
  }

  async predict(pileup: Float32Array | tf.Tensor3D): Promise<PredictionResult> {
    const [r] = await this.predictBatch(this.asBatch1(pileup));
    return r;
  }

  async predictBatch(pileups: BatchInput | tf.Tensor4D): Promise<PredictionResult[]> {
    const isTensor = pileups instanceof tf.Tensor;
    const x = isTensor
      ? (pileups as tf.Tensor4D)
      : tf.tensor4d(
          (pileups as BatchInput).data,
          [(pileups as BatchInput).batch, ...DV_INPUT_SHAPE],
        );
    try {
      const y = this.model.predict(x) as tf.Tensor;
      const flat = await y.data();
      y.dispose();
      const out: PredictionResult[] = [];
      for (let i = 0; i < flat.length; i += 3) {
        const probs: Record<Genotype, number> = {
          hom_ref: flat[i],
          het: flat[i + 1],
          hom_alt: flat[i + 2],
        };
        let argmaxIdx: 0 | 1 | 2 = 0;
        if (probs.het > probs.hom_ref) argmaxIdx = 1;
        const currentMax = argmaxIdx === 0 ? probs.hom_ref : probs.het;
        if (probs.hom_alt > currentMax) argmaxIdx = 2;
        out.push({
          probs,
          argmax: DV_CLASSES[argmaxIdx],
          confidence: flat[i + argmaxIdx],
        });
      }
      return out;
    } finally {
      if (!isTensor) x.dispose();
    }
  }

  countParams(): number {
    return this.model.countParams();
  }

  dispose(): void {
    this.model.dispose();
  }

  private asBatch1(pileup: Float32Array | tf.Tensor3D): BatchInput | tf.Tensor4D {
    if (pileup instanceof tf.Tensor) return pileup.expandDims(0) as tf.Tensor4D;
    if (pileup.length !== SAMPLES_PER_PILEUP) {
      throw new Error(
        `pileup length ${pileup.length} != expected ${SAMPLES_PER_PILEUP} (100*221*7)`,
      );
    }
    return { data: pileup, batch: 1 };
  }
}
