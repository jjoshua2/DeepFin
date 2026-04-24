from __future__ import annotations


def _result_to_wdl(result: str, *, pov_white: bool) -> int:
  # 0=W,1=D,2=L from side-to-move perspective at that position.
  # python-chess returns "*" when a game is truncated before a terminal result
  # (e.g. max_plies reached). Treat this as a draw target so we don't inject
  # systematic "loss" labels into unfinished games.
    if result in {"1/2-1/2", "*"}:
        return 1
    white_won = result == "1-0"
    if pov_white:
        return 0 if white_won else 2
    return 0 if (not white_won) else 2
