from ._lc0_ext import CBoard

MAX_DEPTH: int
SLIDER_BACKEND: str

def perft(board: CBoard, depth: int, /) -> int: ...
def perft_divide(board: CBoard, depth: int, /) -> dict[str, int]: ...
