from __future__ import annotations
import time, numpy as np, chess
import onnxruntime as ort
from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import fill_lc0_history_repeat

NET = "data/lc0/onnx/BT4-it332-vanilla-winner.onnx"
t0 = time.time()
so = ort.SessionOptions(); so.intra_op_num_threads = 8
sess = ort.InferenceSession(NET, so, providers=["CPUExecutionProvider"])
print("init", time.time()-t0, sess.get_providers())
print("inputs", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs", [(o.name, o.shape) for o in sess.get_outputs()])
b = chess.Board()
x = fill_lc0_history_repeat(encode_position(b, add_features=False, input_history_encoding="lc0_root")).astype(np.float32)
for bs in (8, 32):
    feats = np.repeat(x[None], bs, axis=0)
    sess.run(None, {sess.get_inputs()[0].name: feats})
    t = time.time()
    for _ in range(2):
        out = sess.run(None, {sess.get_inputs()[0].name: feats})
    dt = (time.time()-t)/2
    print(f"bs={bs} {dt:.3f}s  {dt/bs*1000:.1f} ms/pos")
print([np.asarray(o).shape for o in out])
