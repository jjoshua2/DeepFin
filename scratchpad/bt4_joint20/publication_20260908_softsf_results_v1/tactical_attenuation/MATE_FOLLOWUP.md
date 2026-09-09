# Mate handling follow-up, before training

The first diagnostic excluded every row with any mate-encoded move. This is too
broad for a tactical-filter candidate: a quiet position can contain a single move
that permits forced mate. Excluding that whole position removes precisely useful
SF rejection evidence. The exclusion was a diagnostic limitation, not a proposed
permanent training rule.

The qualified sample collector identifies mate bands with abs(effective_cp)>32000.
The shared stockfish/wdl.py mapping places mates outside the clamped raw-cp band.
Do not feed distances between mate encodings into an ordinary centipawn temperature.

Proposed categorical treatment for qualification:

- If a forced winning mate is present, preserve BT4 ratios among all winning-mate
  moves; multiply other moves by the same 0.1 attenuation floor.
- Otherwise, apply the raw-cp filter among nonmate moves and multiply forced-loss
  moves by 0.1. This includes positions whose only mate observation is a bad move.
- If all moves are forced losses, retain BT4 unchanged for this first test. Do not
  convert mate-distance preferences into cp differences or select a delay heuristic.

This does not assert mate findings are infallible. Search depth, score bounds and
source provenance remain material. The retained 0.1 multiplier avoids a hard veto.
Before selection, report how much of the excluded 27.6% falls in each category and
verify the actual raw source/consumer mapping. Keep prior diagnostic files immutable.

The full candidate must preserve legal support before storage, remain finite and
normalized, keep BT4 ratios within equally weighted groups and leave value targets
unchanged. Final float16 effects and full-source admission require implementation
qualification. This note does not authorize a new training launch by itself.

## Saved-bank category count

A subsequent read-only pass verified all 64 NPZ SHA256 pins against the bank
completion, observation SHA256, source row/game/ply joins and exact equivalence of
the stored mate flag to any legal abs(effective_cp)>32000. It completed with exit 0.

| Category | Sample rows | Weighted fraction |
| --- | ---: | ---: |
| No mate scores | 2967 | 0.7243869099177133 |
| Losing-mate alternatives only | 604 | 0.14745290168904288 |
| Winning mate available | 398 | 0.09716056734517108 |
| All moves forced losses | 127 | 0.03099962104807276 |

Counts sum to 4096. These are descriptive inherited sample weights, not population
confidence intervals or independently verified chess outcomes. The first excluded
category is particularly relevant to tactical rejection; its presence does not
establish that BT4 places substantial mass on those losing moves.
