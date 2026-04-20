# deepvariant-tfjs

Vue 3 + Vite + TypeScript app shell around a single
`DeepVariantModel` class that runs Google's DeepVariant 1.8 WGS
InceptionV3 model client-side via TensorFlow.js.

The conversion + cross-runtime parity verification lives in the sibling
[`dv-tfjs`](https://github.com/galinilin/dv-tfjs) repo. This project
just consumes its outputs.

Live demo: <https://galinilin.github.io/deepvariant-tfjs/>

## Layout

```
src/lib/DeepVariantModel.ts    The class. Single export, depends only on @tensorflow/tfjs.
src/lib/npy.ts                 Tiny .npy v1/v2 reader (used by the smoke test).
src/App.vue                    Placeholder page with a "Run parity smoke test" button.
public/models/                 Sync'd from dv-tfjs (gitignored).
public/fixtures/               Sync'd .npy parity fixtures (gitignored).
scripts/sync-models.sh         Hydrate public/ from /root/dv-tfjs/out/.
```

## Use the class

```ts
import { DeepVariantModel } from './lib/DeepVariantModel';

const dv = await DeepVariantModel.load({ precision: 'uint8' });
const probs = await dv.predict(pileupFloat32Array);   // pileup is length 100*221*7
console.log(probs.argmax, probs.confidence);          // e.g. "het" 0.94
dv.dispose();
```

The class only does inference. Pileup construction (BAM → tensor),
visualization, and any UI are out of scope — build them on top.

## Run

```sh
# 1. Pull both model artifacts (float32 + uint8) from sibling dv-tfjs build
npm run sync-models             # full sync, both precisions, ~110 MB

# 2. Install + dev
npm install
npm run dev                     # http://127.0.0.1:5173
# click "Run parity smoke test" to confirm the class produces the same
# numbers the dv-tfjs verification suite produced.

# 3. Build
npm run build && npm run preview
```

### Choosing what to sync

`sync-models.sh` takes one positional argument:

| Mode         | What it pulls                                | Use for          |
| ------------ | -------------------------------------------- | ---------------- |
| `full`       | float32 (88 MB) + uint8 (22 MB) + fixtures   | Local dev (default) |
| `uint8-only` | uint8 (22 MB) + fixtures, removes float32    | Pages deploy     |

```sh
bash scripts/sync-models.sh full          # both precisions
bash scripts/sync-models.sh uint8-only    # uint8 only, deletes any float32 dir
```

`npm run predeploy` automatically runs `uint8-only` so the published
Pages site stays lean.

## Deploy to GitHub Pages

The site is a pure SPA — `dist/` after `vite build` is everything Pages
needs. The deploy script ships `dist/` to a `gh-pages` branch on the
configured remote.

```sh
# One-time: create the GitHub repo (public) and link it
git init
git remote add origin git@github.com:<user>/deepvariant-tfjs.git
git add . && git commit -m "init"
git push -u origin master      # source repo

# Anytime: build (uint8-only) + push to gh-pages
npm run deploy
# Then in repo Settings -> Pages: Source = "Deploy from a branch",
# branch = gh-pages, dir = /
# Site goes live at https://<user>.github.io/deepvariant-tfjs/
```

`npm run deploy` runs `predeploy` first, which calls
`sync-models.sh uint8-only` and then `vite build`. The built `dist/` is
~24 MB (uint8 model 22 MB + fixtures + JS/CSS). Float32 is intentionally
excluded from the Pages artifact for bandwidth reasons — see the
"Float32 backwards compatibility" section below.

### If your repo is named something other than `deepvariant-tfjs`

`vite.config.ts` sets `base: '/deepvariant-tfjs/'` for production
builds. Override per-build for a different repo name:

```sh
VITE_BASE=/foo/ npm run deploy
```

(Or edit the default in `vite.config.ts`.)

### Float32 backwards compatibility

The `DeepVariantModel` class permanently supports
`precision: 'float32'`. The Pages deployment ships uint8 only as a
size optimization, not a deprecation. Three ways to keep float32
available:

1. **Local dev / self-hosted deploy.** Run `sync-models.sh full` and
   build without the `predeploy` indirection — the float32 dir lives
   alongside uint8 under `public/models/tfjs_dv_wgs/` and the class
   loads it via `DeepVariantModel.load({ precision: 'float32' })`.
2. **External hosting.** Upload `public/models/tfjs_dv_wgs/` to a CDN
   or to a GitHub Release on this repo, then construct the URL at the
   call site:
   ```ts
   await DeepVariantModel.load({
     precision: 'float32',
     modelBaseUrl: 'https://github.com/<user>/deepvariant-tfjs/releases/download/v1/',
   });
   ```
3. **Reproduce from upstream.** The conversion pipeline in the sibling
   `dv-tfjs` repo regenerates float32 (and uint8) weights bit-identical
   to the published DeepVariant 1.8 WGS checkpoint. Run that, then
   `npm run sync-models full`.

### Caveats

- **GH Pages on private repos requires GitHub Pro/Team/Enterprise.** This
  repo is intended to be public, so this isn't an issue here — flagging
  it for future forks.
- **Bandwidth.** GH Pages soft-limit is 100 GB/month. uint8 at 22 MB
  per first-time visitor leaves plenty of headroom.
- **Per-file limit is 100 MB**, every `.bin` shard is 4 MB. Fine.
- **`.nojekyll` is auto-included** (lives in `public/`, copied into
  `dist/` by Vite).

## Pinning

`@tensorflow/tfjs@4.22.0` matches the version proven in dv-tfjs's
browser CPU + WebGL parity check. Bumping it requires re-running
parity.
