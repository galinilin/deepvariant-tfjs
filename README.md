# deepvariant-tfjs

Vue 3 + Vite + TypeScript app shell around a single
`DeepVariantModel` class that runs Google's DeepVariant 1.8 WGS
InceptionV3 model client-side via TensorFlow.js.

Live demo: <https://galinilin.github.io/deepvariant-tfjs/>

## What even is this?

This project runs Google's DeepVariant v1.8 WGS model end-to-end in your browser. The candidate engine, the seven-channel pileup encoder, and the backbone InceptionV3 network itself are all here, in TypeScript + tfjs. No server.

I quantized the released checkpoint down to uint8 to shrink it to ~22 MB, ran parity tests against upstream, and the predictions match within rounding noise. Inference runs on your device through tfjs's WebGL backend.

### Why does this exist?

Over the last year or so I have been exploring where deep learning meets biology, and DeepVariant felt like the cleanest thing to start from: a CNN doing real, useful work, used in real clinical pipelines, with a paper trail you can sit down and read. While digging into how variant-calling gets reframed as image classification, I stumbled into the official blog post [Looking Through DeepVariant's Eyes](https://google.github.io/deepvariant/posts/2020-02-20-looking-through-deepvariants-eyes/) and got the urge to build a live version of it.

I wanted to take it apart and make it easily traceable, instead of just calling `.predict()` on a black box. So I rebuilt the encoder and wrapped a visualization layer around it. That turned into this sandbox.

The data within the sandbox is generated synthetically. Each time you click Randomize I regenerate a ~1500 bp reference, drop in a handful of scenarios (het / hom_alt SNVs, small indels, low-quality cases), and simulate reads that cover them, so the model is real, the data feeding it is fabricated, and the placed scenarios act as ground truth.

Some years ago I built a similar in-browser sandbox called [artificialneural.network](https://github.com/galinilin/artificialneural.network) — a toy neural net you can configure layer by layer, compile against a sample dataset, train step by step, and click any neuron to inspect its weights. That one was a way to visualize and teach how an MLP actually moves while it learns. This is what happens when you point the same instinct: interactive, in-browser, p5 + tfjs, every step legible, at a real production model.

[GitHub](https://github.com/galinilin) · [X](https://x.com/galinilin) · [LinkedIn](https://www.linkedin.com/in/galiun)
