"""Enumerate a Leela .pb.gz net's real architecture straight from the wire format.

No protoc, no lczero-training import: field numbers come from lc0's own
`net.proto` and the payload is walked as raw protobuf. That matters because the
lczero-training checkout available here predates BT4's multi-head value tower --
reading the SHIPPED WEIGHTS is the only way to learn what BT4 actually contains
rather than what some version of the trainer could emit.

Layer.params is fp16 (Format.LINEAR16), so element count = len(params) // 2.
"""
from __future__ import annotations

import gzip
import sys
from typing import Iterator

# --- net.proto field numbers (lc0 libs/lczero-common/proto/net.proto) ---
NET_WEIGHTS, NET_MINVER, NET_FORMAT, NET_TRAINPARAMS = 10, 3, 4, 5
W_ENCODER, W_HEADCOUNT, W_VALUE_HEADS, W_POLICY_HEADS = 27, 28, 44, 45
W_IP_EMB_W, W_SMOLGEN_W = 25, 35
W_IP_MOV_W, W_IP1_MOV_W, W_IP2_MOV_W, W_IP2_MOV_B = 31, 13, 15, 16
LAYER_PARAMS = 3

VALUE_HEAD_SLOTS = {1: "winner", 2: "q", 3: "st"}
VH = {1: "ip_val_w", 2: "ip_val_b", 3: "ip1_val_w", 4: "ip1_val_b",
      5: "ip2_val_w", 6: "ip2_val_b", 7: "ip_val_err_w", 8: "ip_val_err_b",
      9: "ip_val_cat_w", 10: "ip_val_cat_b"}
POLICY_HEAD_SLOTS = {3: "vanilla", 4: "optimistic_st", 5: "soft", 6: "opponent"}
PH = {1: "ip_pol_w", 2: "ip_pol_b", 3: "ip2_pol_w(wq)", 4: "ip2_pol_b",
      5: "ip3_pol_w(wk)", 6: "ip3_pol_b", 7: "ip4_pol_w(ppo)", 9: "pol_headcount"}
MHA = {1: "q_w", 3: "k_w", 5: "v_w", 7: "dense_w", 9: "smolgen",
       10: "rpe_q", 11: "rpe_k", 12: "rpe_v"}
FFN = {1: "dense1_w", 2: "dense1_b", 3: "dense2_w", 4: "dense2_b"}


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s; s += 7
        if not x & 0x80:
            return r, i


def fields(b: bytes) -> Iterator[tuple[int, int, object]]:
    i, n = 0, len(b)
    while i < n:
        k, i = _varint(b, i)
        fn, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, wt, v
        elif wt == 2:
            ln, i = _varint(b, i); yield fn, wt, b[i:i + ln]; i += ln
        elif wt == 5:
            yield fn, wt, b[i:i + 4]; i += 4
        elif wt == 1:
            yield fn, wt, b[i:i + 8]; i += 8
        else:
            raise ValueError(f"unsupported wire type {wt} at {i}")


def one(msg: bytes, want: int):
    for fn, _, v in fields(msg):
        if fn == want:
            return v
    return None


def nparams(layer: bytes | None) -> int:
    """Element count of a Layer (fp16 payload)."""
    if layer is None:
        return 0
    p = one(layer, LAYER_PARAMS)
    return len(p) // 2 if p else 0


def submap(msg: bytes, names: dict[int, str]) -> dict[str, int]:
    return {names.get(fn, str(fn)): nparams(v)
            for fn, wt, v in fields(msg) if wt == 2 and fn in names}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/lc0/BT4-it332.pb.gz"
    raw = gzip.open(path, "rb").read()
    print(f"# {path}\nuncompressed: {len(raw)/1e6:.1f} MB")

    mv = one(raw, NET_MINVER)
    if mv:
        d = {fn: v for fn, _, v in fields(mv)}
        print(f"min lc0 version: {d.get(1,0)}.{d.get(2,0)}.{d.get(3,0)}")
    fmt = one(raw, NET_FORMAT)
    if fmt:
        nf = one(fmt, 2)
        if nf:
            print("NetworkFormat:", {fn: v for fn, _, v in fields(nf)})
    tp = one(raw, NET_TRAINPARAMS)
    if tp:
        print("TrainingParams:", {fn: v for fn, _, v in fields(tp) if not isinstance(v, bytes)})

    W = one(raw, NET_WEIGHTS)
    assert W is not None, "no Weights message"

    enc = [v for fn, _, v in fields(W) if fn == W_ENCODER]
    hc = one(W, W_HEADCOUNT)
    print(f"\n## trunk\nencoder layers: {len(enc)}   headcount: {hc}")
    print(f"input embedding ip_emb_w: {nparams(one(W, W_IP_EMB_W)):,}")
    print(f"global smolgen_w        : {nparams(one(W, W_SMOLGEN_W)):,}")
    if enc:
        e0 = enc[0]
        mha = one(e0, 1); ffn = one(e0, 4)
        print("encoder[0] MHA :", {k: f"{v:,}" for k, v in submap(mha, MHA).items()} if mha else None)
        if mha and one(mha, 9):
            print("encoder[0] smolgen:", {k: f"{v:,}" for k, v in
                  submap(one(mha, 9), {1:"compress",2:"dense1_w",6:"dense2_w"}).items()})
        print("encoder[0] FFN :", {k: f"{v:,}" for k, v in submap(ffn, FFN).items()} if ffn else None)

    print("\n## value heads (Weights.value_heads, field 44)")
    vhs = one(W, W_VALUE_HEADS)
    if vhs:
        for fn, wt, v in fields(vhs):
            if wt != 2:
                continue
            name = VALUE_HEAD_SLOTS.get(fn, f"value_head_map[{fn}]")
            d = submap(v, VH)
            h = d.get("ip1_val_b", 0)
            print(f"\n  [{name}]  {len(v)/1e6:.2f} MB")
            for k in ("ip_val_w","ip1_val_w","ip2_val_w","ip_val_err_w","ip_val_cat_w"):
                if k in d:
                    print(f"    {k:<16}{d[k]:>10,}   out={d.get(k[:-1]+'b',0)}")
            print(f"    -> hidden width {h}; categorical present: {'ip_val_cat_w' in d}"
                  f"; value-error present: {'ip_val_err_w' in d}")

    print("\n## policy heads (Weights.policy_heads, field 45)")
    phs = one(W, W_POLICY_HEADS)
    if phs:
        shared_w = nparams(one(phs, 1))
        print(f"  SHARED ip_pol_w: {shared_w:,}  (one embedding for all policy heads)")
        for fn, wt, v in fields(phs):
            if wt != 2 or fn not in POLICY_HEAD_SLOTS:
                continue
            d = submap(v, PH)
            print(f"  [{POLICY_HEAD_SLOTS[fn]}] {len(v)/1e6:.2f} MB  " +
                  ", ".join(f"{k}={n:,}" for k, n in d.items() if n))

    print("\n## moves-left head")
    print(f"  ip_mov_w={nparams(one(W, W_IP_MOV_W)):,}  ip1={nparams(one(W, W_IP1_MOV_W)):,}"
          f"  ip2={nparams(one(W, W_IP2_MOV_W)):,} -> {nparams(one(W, W_IP2_MOV_B))} out")

    def is_layer(msg: bytes) -> bool:
        """A Layer is structurally {1:fixed32, 2:fixed32, 3:bytes} and nothing else.

        ⚑ Do NOT test "has a field 3 of bytes" -- `PolicyHeads.vanilla` is field 3
        and is a PolicyHead, so that heuristic silently mis-counts it as weights and
        skips its children. It undercounted BT4 by 6x (32.2M vs the true 191.3M).
        """
        try:
            fs = list(fields(msg))
        except Exception:
            return False
        if not fs:
            return False
        shape_ok = all((fn in (1, 2) and wt == 5) or (fn == 3 and wt == 2)
                       for fn, wt, _ in fs)
        return shape_ok and any(fn == LAYER_PARAMS for fn, _, _ in fs)

    def walk(b: bytes) -> int:
        t = 0
        for fn, wt, v in fields(b):
            if wt != 2:
                continue
            if is_layer(v):
                t += nparams(v)
            else:
                try:
                    t += walk(v)
                except Exception:
                    pass
        return t
    total = walk(W)
    print(f"\n## TOTAL parameters (fp16 elements across all Layers): {total:,}  (~{total/1e6:.1f}M)")


if __name__ == "__main__":
    main()
