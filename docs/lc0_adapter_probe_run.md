# lc0-net-in-our-search probe

Frozen output of, on 2026-08-06 (CPU only, 32 threads, no GPU):

```
PYTHONPATH=. python3 scripts/lc0_adapter_probe.py \
  --onnx /home/josh/projects/chess/data/lc0/onnx/BT4-it332-vanilla-winner.onnx \
  --matched <scratchpad>/lc0data/matched60.npz \
  --out runs/lc0_adapter_probe.md --root-dump runs/lc0_adapter_roots.npz
```

`runs/` is gitignored, so this copy is the branch-visible record. Diagnostic
only: nothing here changes a production path, and the KL column is a
distribution-SHAPE comparison against a foreign net's 800-visit noisy-root
target, NOT an argument for retuning `gumbel_c_scale`.

The matched set is tracked at `tests/data/lc0_matched60.npz` (the `--matched`
default); the scratchpad path in the command above was where it lived when this
run was made.

**Moved goalpost, recorded.** The startpos gate was originally written as
"mainstream mass (e4/d4/Nf3/c4) > 0.75". BT4 realized 0.519, so the bar was
LOWERED to > 0.5 plus a new requirement that the top-1 move be one of the four.
The realized 0.519 clears the new bar by 3.8%, and the new bar was chosen after
seeing the number. The reasoning — a modern Leela opening book is genuinely
broad, so a tighter bar tests lc0's taste rather than this adapter — is in the
gate's own comment, but it is a post-hoc threshold and should be read as one.
The mate-in-1 and index-correspondence gates were NOT touched.

**RESOLVED — the mass threshold is gone, the number is not.** A post-hoc bar
cleared by 3.8% has no discriminating power left, so `gate_net_sanity` no longer
scores mainstream mass at all: it PRINTS the realized value alongside the
original >0.75 bar and this loosening, with no PASS/FAIL attached. The startpos
verdict now rests only on the clauses that can still fail — top-1 among
e4/d4/Nf3/c4, and |W-L| < 0.15. Nothing changes about the exit code, which has
always been decided by round-trip and index-correspondence alone and never read
`ok_net`. Any earlier "net sanity PASS" in this file therefore included a term
that could not fail; read it accordingly.

- net: `/home/josh/projects/chess/data/lc0/onnx/BT4-it332-vanilla-winner.onnx`
- net sha256: `1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0`
- matched set: `/tmp/claude-1000/-home-josh-projects-chess/f2207141-e3db-40fc-82c8-a50dd9d223f1/scratchpad/lc0data/matched60.npz`
- search: 32 sims, gumbel_scale 0.75, seed 20260806 (re-seeded per position, so configs are paired)

- ONNX session up in 1.8s (CPU provider)

## Gates
- GATE round-trip encoder(decoder(planes)) == planes: all 112 planes exact on 59/60; the 96 piece planes exact on 60/60 — PASS
    record 9 (training.1388605638.gz:121) differs on planes [64]
- GATE index-correspondence: 55586 legal moves over 1848 positions, 0 mismatches — PASS
    move kinds exercised: {'normal': 55186, 'castle': 68, 'promo_q': 83, 'promo_r': 83, 'promo_b': 83, 'promo_n': 83}
- GATE startpos: top = [('d4', 0.163), ('Nf3', 0.157), ('c4', 0.13), ('g3', 0.091), ('e3', 0.08), ('e4', 0.069), ('b3', 0.058), ('Nc3', 0.047)]; mass on e4/d4/Nf3/c4 = 0.519 (want >0.5, top-1 among them); W-L = +0.025 (W/D/L 0.190/0.645/0.165) — PASS
- GATE mate-in-1 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1: top1 = Ra8# p=0.912 (want Ra8#); W-L = +1.000 — PASS
    top: [('Ra8#', 0.912), ('Kf2', 0.018), ('Kg2', 0.011), ('Kh2', 0.006), ('Ra5', 0.006), ('Ra6', 0.005)]
- GATE mate-in-1 r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4: top1 = Qxf7# p=0.817 (want Qxf7#); W-L = +0.999 — PASS
    top: [('Qxf7#', 0.817), ('Bxf7+', 0.134), ('Qf5', 0.001), ('Nh3', 0.001), ('Qh5', 0.001), ('Qf6', 0.001)]
- midgame r1bqkb1r/pp2pppp/2n2n2/2pp4/3P4/2N1PN2/PPP2PPP/R1BQKB1R b KQkq - 0 5
    top: [('a6', 0.531), ('Bg4', 0.269), ('e6', 0.047), ('Bf5', 0.043), ('h6', 0.019)]
- midgame r2q1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R b KQ - 0 10
    top: [('Na5', 0.099), ('Rc8', 0.093), ('Qa5', 0.073), ('Qc7', 0.067), ('e6', 0.058)]
- midgame 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1
    top: [('Rxf4+', 0.813), ('Rc4', 0.122), ('e3', 0.023), ('Ka6', 0.015), ('Ka4', 0.004)]
- midgame r3k2r/pbpnqppp/1p2pn2/3p4/2PP4/1PN1PN2/PBQ2PPP/R3KB1R w KQkq - 0 10
    top: [('cxd5', 0.111), ('Bd3', 0.106), ('a4', 0.09), ('Be2', 0.064), ('h4', 0.064)]

- gate summary: round-trip PASS, index-correspondence PASS, net sanity PASS
- positions kept 60/60 (dropped 0 on legal-set disagreement with the stored target)

## Eval memoization
- warm pass: topk=all, 64 sims, all 60 positions in one batched search, 157.5s, 3819 net evals cached (83 ONNX calls)
- sweep: 9x4 configs x 60 positions in 184.2s; cache hits 68144, misses 1828 after warm-up
- misses per config (worst 4): [('c=0.0/k=8', 875), ('c=0.05/k=8', 201), ('c=0.1/k=8', 175), ('c=0.25/k=8', 135)]
- topk=8 vs topk=16: bit-identical search output on 116/540 (position, c_scale) pairs [candidates are capped at ceil(sims/2)=16]
- topk=32 vs topk=16: bit-identical search output on 540/540 (position, c_scale) pairs [candidates are capped at ceil(sims/2)=16]
- topk=218 vs topk=16: bit-identical search output on 540/540 (position, c_scale) pairs [candidates are capped at ceil(sims/2)=16]
- root dump (production config) → `runs/lc0_adapter_roots.npz` (59 of 60 positions: per-action legal id / prior / visits / completed-Q / lc0 target, plus per-position root Q, phase, FEN, provenance). Positions the search short-circuits at the root (single legal move, or a root tactic that returns a fixed policy) have no tree to dump.

## lc0's own stored targets (reference)

| phase | n | median entropy (nats) | median support (p>1e-3) | median max-prob | median n_legal | median moves with p>0 |
|---|---|---|---|---|---|---|
| opening | 20 | 1.719 | 31.0 | 0.445 | 31.0 | 31.0 |
| middlegame | 20 | 1.393 | 34.0 | 0.601 | 35.5 | 35.5 |
| endgame | 20 | 1.888 | 20.0 | 0.449 | 20.0 | 20.0 |

Supports below are all counted at p>1e-3 so the two sides are comparable; lc0's targets additionally carry a nonzero tail on essentially every legal move (root Dirichlet noise in training games).

## Our Gumbel search target, per (config, phase)

| c_scale | topk | phase | median H (nats) | median support | median max-p | median KL(lc0‖ours) | top1 agree |
|---|---|---|---|---|---|---|---|
| 0.0 | 8 | opening | 1.773 | 30.5 | 0.459 | 0.055 | 0.80 |
| 0.0 | 8 | middlegame | 1.567 | 34.0 | 0.505 | 0.132 | 0.65 |
| 0.0 | 8 | endgame | 1.985 | 18.5 | 0.380 | 0.033 | 0.65 |
| 0.0 | 16 | opening | 1.773 | 30.5 | 0.459 | 0.055 | 0.80 |
| 0.0 | 16 | middlegame | 1.567 | 34.0 | 0.505 | 0.132 | 0.65 |
| 0.0 | 16 | endgame | 1.985 | 18.5 | 0.380 | 0.033 | 0.65 |
| 0.0 | 32 | opening | 1.773 | 30.5 | 0.459 | 0.055 | 0.80 |
| 0.0 | 32 | middlegame | 1.567 | 34.0 | 0.505 | 0.132 | 0.65 |
| 0.0 | 32 | endgame | 1.985 | 18.5 | 0.380 | 0.033 | 0.65 |
| 0.0 | 218 | opening | 1.773 | 30.5 | 0.459 | 0.055 | 0.80 |
| 0.0 | 218 | middlegame | 1.567 | 34.0 | 0.505 | 0.132 | 0.65 |
| 0.0 | 218 | endgame | 1.985 | 18.5 | 0.380 | 0.033 | 0.65 |
| 0.01 | 8 | opening | 1.744 | 30.0 | 0.472 | 0.050 | 0.80 |
| 0.01 | 8 | middlegame | 1.483 | 34.0 | 0.558 | 0.146 | 0.70 |
| 0.01 | 8 | endgame | 1.936 | 18.5 | 0.393 | 0.058 | 0.60 |
| 0.01 | 16 | opening | 1.721 | 30.0 | 0.469 | 0.050 | 0.80 |
| 0.01 | 16 | middlegame | 1.502 | 33.5 | 0.556 | 0.164 | 0.70 |
| 0.01 | 16 | endgame | 1.931 | 18.5 | 0.391 | 0.058 | 0.60 |
| 0.01 | 32 | opening | 1.721 | 30.0 | 0.469 | 0.050 | 0.80 |
| 0.01 | 32 | middlegame | 1.502 | 33.5 | 0.556 | 0.164 | 0.70 |
| 0.01 | 32 | endgame | 1.931 | 18.5 | 0.391 | 0.058 | 0.60 |
| 0.01 | 218 | opening | 1.721 | 30.0 | 0.469 | 0.050 | 0.80 |
| 0.01 | 218 | middlegame | 1.502 | 33.5 | 0.556 | 0.164 | 0.70 |
| 0.01 | 218 | endgame | 1.931 | 18.5 | 0.391 | 0.058 | 0.60 |
| 0.025 | 8 | opening | 1.678 | 29.5 | 0.490 | 0.055 | 0.80 |
| 0.025 | 8 | middlegame | 1.400 | 32.5 | 0.555 | 0.160 | 0.65 |
| 0.025 | 8 | endgame | 1.842 | 18.5 | 0.438 | 0.116 | 0.60 |
| 0.025 | 16 | opening | 1.663 | 28.0 | 0.481 | 0.046 | 0.80 |
| 0.025 | 16 | middlegame | 1.414 | 30.0 | 0.558 | 0.150 | 0.70 |
| 0.025 | 16 | endgame | 1.857 | 16.0 | 0.406 | 0.094 | 0.60 |
| 0.025 | 32 | opening | 1.663 | 28.0 | 0.481 | 0.046 | 0.80 |
| 0.025 | 32 | middlegame | 1.414 | 30.0 | 0.558 | 0.150 | 0.70 |
| 0.025 | 32 | endgame | 1.857 | 16.0 | 0.406 | 0.094 | 0.60 |
| 0.025 | 218 | opening | 1.663 | 28.0 | 0.481 | 0.046 | 0.80 |
| 0.025 | 218 | middlegame | 1.414 | 30.0 | 0.558 | 0.150 | 0.70 |
| 0.025 | 218 | endgame | 1.857 | 16.0 | 0.406 | 0.094 | 0.60 |
| 0.05 | 8 | opening | 1.595 | 28.5 | 0.517 | 0.070 | 0.80 |
| 0.05 | 8 | middlegame | 1.275 | 32.5 | 0.629 | 0.128 | 0.70 |
| 0.05 | 8 | endgame | 1.649 | 18.5 | 0.489 | 0.224 | 0.55 |
| 0.05 | 16 | opening | 1.583 | 26.0 | 0.506 | 0.064 | 0.80 |
| 0.05 | 16 | middlegame | 1.279 | 27.0 | 0.592 | 0.137 | 0.70 |
| 0.05 | 16 | endgame | 1.793 | 13.5 | 0.416 | 0.132 | 0.60 |
| 0.05 | 32 | opening | 1.583 | 26.0 | 0.506 | 0.064 | 0.80 |
| 0.05 | 32 | middlegame | 1.279 | 27.0 | 0.592 | 0.137 | 0.70 |
| 0.05 | 32 | endgame | 1.793 | 13.5 | 0.416 | 0.132 | 0.60 |
| 0.05 | 218 | opening | 1.583 | 26.0 | 0.506 | 0.064 | 0.80 |
| 0.05 | 218 | middlegame | 1.279 | 27.0 | 0.592 | 0.137 | 0.70 |
| 0.05 | 218 | endgame | 1.793 | 13.5 | 0.416 | 0.132 | 0.60 |
| 0.1 | 8 | opening | 1.362 | 26.5 | 0.601 | 0.081 | 0.75 |
| 0.1 | 8 | middlegame | 1.117 | 29.0 | 0.742 | 0.176 | 0.65 |
| 0.1 | 8 | endgame | 1.315 | 15.5 | 0.546 | 0.550 | 0.60 |
| 0.1 | 16 | opening | 1.455 | 24.0 | 0.539 | 0.108 | 0.85 |
| 0.1 | 16 | middlegame | 1.066 | 27.0 | 0.755 | 0.194 | 0.70 |
| 0.1 | 16 | endgame | 1.672 | 13.5 | 0.431 | 0.222 | 0.60 |
| 0.1 | 32 | opening | 1.455 | 24.0 | 0.539 | 0.108 | 0.85 |
| 0.1 | 32 | middlegame | 1.066 | 27.0 | 0.755 | 0.194 | 0.70 |
| 0.1 | 32 | endgame | 1.672 | 13.5 | 0.431 | 0.222 | 0.60 |
| 0.1 | 218 | opening | 1.455 | 24.0 | 0.539 | 0.108 | 0.85 |
| 0.1 | 218 | middlegame | 1.066 | 27.0 | 0.755 | 0.194 | 0.70 |
| 0.1 | 218 | endgame | 1.672 | 13.5 | 0.431 | 0.222 | 0.60 |
| 0.25 | 8 | opening | 1.096 | 23.5 | 0.741 | 0.211 | 0.75 |
| 0.25 | 8 | middlegame | 0.569 | 17.5 | 0.869 | 0.433 | 0.75 |
| 0.25 | 8 | endgame | 0.219 | 2.5 | 0.955 | 2.588 | 0.50 |
| 0.25 | 16 | opening | 1.232 | 20.5 | 0.597 | 0.262 | 0.80 |
| 0.25 | 16 | middlegame | 0.609 | 14.5 | 0.872 | 0.483 | 0.70 |
| 0.25 | 16 | endgame | 1.284 | 11.5 | 0.500 | 0.426 | 0.55 |
| 0.25 | 32 | opening | 1.232 | 20.5 | 0.597 | 0.262 | 0.80 |
| 0.25 | 32 | middlegame | 0.609 | 14.5 | 0.872 | 0.483 | 0.70 |
| 0.25 | 32 | endgame | 1.284 | 11.5 | 0.500 | 0.426 | 0.55 |
| 0.25 | 218 | opening | 1.232 | 20.5 | 0.597 | 0.262 | 0.80 |
| 0.25 | 218 | middlegame | 0.609 | 14.5 | 0.872 | 0.483 | 0.70 |
| 0.25 | 218 | endgame | 1.284 | 11.5 | 0.500 | 0.426 | 0.55 |
| 0.5 | 8 | opening | 0.659 | 20.5 | 0.865 | 0.391 | 0.70 |
| 0.5 | 8 | middlegame | 0.160 | 3.0 | 0.978 | 1.236 | 0.75 |
| 0.5 | 8 | endgame | 0.028 | 2.0 | 0.996 | 5.932 | 0.50 |
| 0.5 | 16 | opening | 0.955 | 15.5 | 0.703 | 0.576 | 0.75 |
| 0.5 | 16 | middlegame | 0.255 | 3.5 | 0.954 | 1.046 | 0.75 |
| 0.5 | 16 | endgame | 0.691 | 5.5 | 0.729 | 0.640 | 0.55 |
| 0.5 | 32 | opening | 0.955 | 15.5 | 0.703 | 0.576 | 0.75 |
| 0.5 | 32 | middlegame | 0.255 | 3.5 | 0.954 | 1.046 | 0.75 |
| 0.5 | 32 | endgame | 0.691 | 5.5 | 0.729 | 0.640 | 0.55 |
| 0.5 | 218 | opening | 0.955 | 15.5 | 0.703 | 0.576 | 0.75 |
| 0.5 | 218 | middlegame | 0.255 | 3.5 | 0.954 | 1.046 | 0.75 |
| 0.5 | 218 | endgame | 0.691 | 5.5 | 0.729 | 0.640 | 0.55 |
| 1.0 | 8 | opening | 0.269 | 8.0 | 0.951 | 0.953 | 0.70 |
| 1.0 | 8 | middlegame | 0.106 | 1.5 | 0.986 | 2.295 | 0.75 |
| 1.0 | 8 | endgame | 0.000 | 1.0 | 1.000 | 8.310 | 0.45 |
| 1.0 | 16 | opening | 0.623 | 5.0 | 0.841 | 1.180 | 0.80 |
| 1.0 | 16 | middlegame | 0.057 | 1.5 | 0.992 | 1.670 | 0.75 |
| 1.0 | 16 | endgame | 0.366 | 3.0 | 0.885 | 1.528 | 0.55 |
| 1.0 | 32 | opening | 0.623 | 5.0 | 0.841 | 1.180 | 0.80 |
| 1.0 | 32 | middlegame | 0.057 | 1.5 | 0.992 | 1.670 | 0.75 |
| 1.0 | 32 | endgame | 0.366 | 3.0 | 0.885 | 1.528 | 0.55 |
| 1.0 | 218 | opening | 0.623 | 5.0 | 0.841 | 1.180 | 0.80 |
| 1.0 | 218 | middlegame | 0.057 | 1.5 | 0.992 | 1.670 | 0.75 |
| 1.0 | 218 | endgame | 0.366 | 3.0 | 0.885 | 1.528 | 0.55 |
| 2.0 | 8 | opening | 0.101 | 2.0 | 0.985 | 1.938 | 0.70 |
| 2.0 | 8 | middlegame | 0.008 | 1.0 | 0.999 | 5.407 | 0.75 |
| 2.0 | 8 | endgame | 0.000 | 1.0 | 1.000 | 8.641 | 0.40 |
| 2.0 | 16 | opening | 0.149 | 2.5 | 0.972 | 2.204 | 0.75 |
| 2.0 | 16 | middlegame | 0.002 | 1.0 | 1.000 | 3.590 | 0.75 |
| 2.0 | 16 | endgame | 0.089 | 1.5 | 0.984 | 3.583 | 0.50 |
| 2.0 | 32 | opening | 0.149 | 2.5 | 0.972 | 2.204 | 0.75 |
| 2.0 | 32 | middlegame | 0.002 | 1.0 | 1.000 | 3.590 | 0.75 |
| 2.0 | 32 | endgame | 0.089 | 1.5 | 0.984 | 3.583 | 0.50 |
| 2.0 | 218 | opening | 0.149 | 2.5 | 0.972 | 2.204 | 0.75 |
| 2.0 | 218 | middlegame | 0.002 | 1.0 | 1.000 | 3.590 | 0.75 |
| 2.0 | 218 | endgame | 0.089 | 1.5 | 0.984 | 3.583 | 0.50 |

- **argmin-KL opening**: c_scale=0.025, topk=16 (median KL 0.046)
- **argmin-KL middlegame**: c_scale=0.05, topk=8 (median KL 0.128)
- **argmin-KL endgame**: c_scale=0.0, topk=8 (median KL 0.033)
