import type { Candidate } from './candidate';

/**
 * Tiny shared state object so the top and bottom p5 sketches can talk
 * without a heavier event bus. Top writes; bottom reads.
 *
 * Keep this minimal. Add fields here only when the bottom sketch (or
 * future sibling sketches) needs read-only access to top-side state.
 */
export const sandboxState: { candidate: Candidate } = {
  candidate: null,
};
