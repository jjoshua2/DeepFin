from __future__ import annotations
import numpy as np, chess, onnxruntime as ort
from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import fill_lc0_history_repeat
from chess_anti_engine.moves.leela_index import leela_index_for_move
so = ort.SessionOptions(); so.intra_op_num_threads = 6
s = ort.InferenceSession("data/lc0/onnx/BT4-it332-vanilla-winner.onnx", so,
                         providers=["CPUExecutionProvider"])
inn = s.get_inputs()[0].name
CASES = [
  ("startpos", chess.STARTING_FEN),
  ("white_up_queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
  ("black_up_queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
  ("black_to_move_up_q", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"),
  ("mate_in_1_white", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"),
]
for name, fen in CASES:
    b = chess.Board(fen)
    x = fill_lc0_history_repeat(encode_position(b, add_features=False,
        input_history_encoding="lc0_root")).astype(np.float32)[None]
    out = [np.asarray(o) for o in s.run(None, {inn: x})]
    w = [a.shape[-1] for a in out]
    pol = out[int(np.argmax(w))][0]; wdl = out[next(i for i,q in enumerate(w) if q==3)][0]
    legal = list(b.legal_moves)
    li = np.array([leela_index_for_move(b, m) for m in legal])
    lg = np.where(li>=0, pol[li.clip(0)], -1e9).astype(np.float64)
    p = np.exp(lg-lg.max()); p/=p.sum()
    o = np.argsort(-p)[:4]
    print(f"{name:22s} WDL={np.round(wdl,3)}  top: " +
          " ".join(f"{legal[int(k)].uci()}:{p[int(k)]:.3f}" for k in o))
