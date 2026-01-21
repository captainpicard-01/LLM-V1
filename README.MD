# Tiny Transformer in JavaScript

This project contains a minimal transformer-style language model implemented
entirely in JavaScript — no dependencies, no frameworks.

## Files
- `tiny_transformer.js` — the full transformer implementation
- `index.html` — runs the model in the browser console

## How to Run
1. Clone or download this repository.
2. Open `index.html` in any browser.
3. Open the browser console to see generated text.

## Notes
- This is a forward-pass-only transformer (no training).
- Includes token embeddings, positional embeddings, self-attention,
  feed-forward layers, and sampling-based generation.
