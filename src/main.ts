import { mountTopSketch } from './sketch/top';
import { mountBottomSketch } from './sketch/bottom';

const topEl = document.getElementById('sketch-top');
const bottomEl = document.getElementById('sketch-bottom');
if (!topEl || !bottomEl) throw new Error('missing sketch containers');

const top = mountTopSketch(topEl);
mountBottomSketch(bottomEl);

const resetBtn = document.getElementById('reset-view');
resetBtn?.addEventListener('click', () => top.resetView());

const randomizeBtn = document.getElementById('randomize');
randomizeBtn?.addEventListener('click', () => top.randomize());
