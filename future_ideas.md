# Future ideas
This file is for longer-term roadmap items that are intentionally out-of-scope for the minimal “working end-to-end” pipeline.

## Opponent mixing / league training
Goal: avoid overspecializing to Stockfish by mixing opponents.

Ideas:
- Play a fraction of games against older checkpoints (self-league), not just the current “latest” model.
- Maintain a small pool of opponents (e.g. last K champions, plus a few diversity picks) and sample opponents by a schedule.
- Add simple promotion rules for opponents (e.g. keep models that are distinct and/or that beat some baseline).
- Track generalization canaries (puzzle suite, fixed-strength Stockfish eval set) and adjust mixing ratio if they regress.
