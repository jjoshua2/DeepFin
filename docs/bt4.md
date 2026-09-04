# BT4 — architecture of the reference Leela net, read off the shipped weights

`data/lc0/BT4-it332.pb.gz` · 382.6 MB uncompressed · **191,303,916 parameters (191.3M)**
· min lc0 version 0.30.0 · `training_steps` 6,315,000

**Why this file exists.** BT4 is our comparison net on the frozen deep-SF audit set and the
candidate teacher for a supervised bootstrap (task #199). Several architectural claims we have
made about "what lc0 does" were taken from `lczero-training` — and **that checkout is stale**:
its `tfprocess.py` builds a *single* value head and its `example.yaml` has no categorical/value-error
entry at all. BT4 ships **three** value towers. Everything below is parsed from the weights
themselves, so it describes what BT4 *is*, not what some version of the trainer could emit.

Reproduce with `scratchpad/bt4/inspect_bt4.py` (no protoc needed — walks the protobuf wire format
using field numbers from lc0's `libs/lczero-common/proto/net.proto`).

---

## 1. Format flags

| field | value | meaning |
|---|---|---|
| `input` | 1 | `INPUT_CLASSICAL_112_PLANE` |
| `output` | 2 | `OUTPUT_WDL` |
| `network` | 6 | `NETWORK_ATTENTIONBODY_WITH_HEADFORMAT` |
| `policy` | 3 | `POLICY_ATTENTION` |
| `value` | 2 | `VALUE_WDL` |
| `moves_left` | 1 | `MOVES_LEFT_V1` |
| `default_activation` | 1 | `DEFAULT_ACTIVATION_MISH` |
| `smolgen_activation` | 7 | `ACTIVATION_SWISH` |
| `ffn_activation` | 0 | `ACTIVATION_DEFAULT` (⇒ Mish) |
| `input_embedding` | — | not set |

⚑ `network = 6` (`..._WITH_HEADFORMAT`), **not** `7` (`..._WITH_MULTIHEADFORMAT`), even though the
net populates `value_heads`/`policy_heads`. Read the head maps directly; do not gate on the enum.

⚑ `input = 1` (`CLASSICAL_112_PLANE`) — i.e. **no canonicalization**, which is surprising for a net
of this generation. Re-verified by decoding the enum by name, so it is what the net declares. Any
consumer that assumes canonicalized input for BT4 is wrong; cross-check against our own lc0 adapter
before trusting either side, since that adapter has already carried one castling-map defect.

## 2. Trunk

- **15 encoder layers**, **32 attention heads**, embedding width **1024** (`ip_emb_w` 638,976 = 624×1024)
- **global smolgen** `smolgen_w` 1,048,576

Per encoder layer:

| block | params |
|---|---|
| MHA `q_w` / `k_w` / `v_w` / `dense_w` | 1,048,576 each (1024×1024) |
| smolgen `compress` | 32,768 (1024×32) |
| smolgen `dense1_w` | 524,288 |
| smolgen `dense2_w` | 2,097,152 |
| FFN `dense1_w` | 1,572,864 (1024×1536) |
| FFN `dense2_w` | 1,572,864 |

FFN multiplier is **1.5×** (1024 → 1536 → 1024). Smolgen is per-layer and is the single largest
per-layer cost (~2.65M of ~8.9M).

## 3. Value heads — **three towers, split by TARGET**

`Weights.value_heads` (field 44). Each is a full independent tower; there is **no** shared value
embedding across them.

| head | `ip_val_w` (embed) | `ip1_val_w` (dense1) | WDL out | **value-error** | **value-categorical** |
|---|---|---|---|---|---|
| `winner` | 131,072 → 128 | 1,048,576 → 128 | 384 → 3 | — | — |
| **`q`** | 131,072 → 128 | 1,048,576 → 128 | 384 → 3 | **128 → 1** | **4,096 → 32 bins** |
| **`st`** | 131,072 → 128 | 1,048,576 → 128 | 384 → 3 | **128 → 1** | **4,096 → 32 bins** |

Per-tower shape: `1024 → 128` per square → flatten `64×128 = 8192` → dense1 `→128` → outputs.

**Two facts that matter and were previously assumed wrong:**

1. **The categorical and value-error outputs are COUPLED** — they are fields *inside* the same
   `ValueHead` message as the embedding and dense1, reading that tower's own **128-d hidden**. The
   categorical head is **4,096 + 32 = 4,128 params**. It is not, and never was, a separate tower.
2. **They are attached to `q` and `st`, NOT to `winner`.** The distributional aux and the
   uncertainty output ride the search-Q and short-term heads; the game-result head gets neither.

lc0's proto reinforces the design: `ValueErrorLossConfig` carries an explicit
`bool propagate_value_gradients`, while `ValueCategoricalLossConfig` has no such flag — the
categorical loss **always** propagates into the value representation.

## 4. Policy heads — **four heads sharing ONE embedding**

`Weights.policy_heads` (field 45). `ip_pol_w` = **1,048,576**, shared by every head below.

| head | `ip2_pol_w` (wq) | `ip3_pol_w` (wk) | `ip4_pol_w` (ppo) |
|---|---|---|---|
| `vanilla` | 1,048,576 | 1,048,576 | 4,096 |
| `optimistic_st` | 1,048,576 | 1,048,576 | 4,096 |
| `soft` | 262,144 | 262,144 | 1,024 |
| `opponent` | 262,144 | 262,144 | 1,024 |

`soft` and `opponent` are **quarter-width** (256 vs 1024 projection). Note the ONNX exporter emits
one value head per file, which is why `data/lc0/onnx/` holds six variants
(`vanilla-q`, `vanilla-st`, `vanilla-winner`, `optimistic-winner`, `soft-winner`, `opponent-winner`)
— **an ONNX export shows only the selected head and cannot be used to enumerate the architecture.**

## 5. Moves-left head

`ip_mov_w` 32,768 (1024→32) → `ip1` 262,144 → `ip2` 128 → **1** output.

---

## 6. Comparison with ours (`configs/pbt2_small.yaml`, 63,084,128 params)

| | BT4 | ours |
|---|---|---|
| trunk width × layers × heads | 1024 × 15 × 32 | 512 × 16 × 16 |
| parameters | 191.3M | 63.1M |
| FFN multiplier | 1.5 flat | 1.5 → ~1.9, per-layer non-uniform |
| smolgen | per-layer + global | per-layer (26.7M, 42.3%), weight-tied `gen_weight` |
| **value tower shape** | 1024→128/sq, flat 8192, hidden 128 | **512→128/sq, flat 8192, hidden 128** |
| **value towers** | **3, split by TARGET** (winner / q / st) | **3, split by OUTPUT TYPE** (wdl / sf_eval / categorical) |
| **categorical head** | **4,128-param branch off the value hidden** | **1,118,496-param independent `ValueHead`** |
| categorical bins | 32 | 32 |
| value-error head | present on `q` and `st` | **absent** |
| policy heads | 4, **one shared embedding** | 4 `AttentionPolicyHead`s |
| policy : value loss weight | 1 : 1 (lc0 `example.yaml`) | **2.01 : 1** (`w_policy` 1.0 + `w_soft` 1.0 vs `w_wdl` 1.0) |

### The one structural inversion

Our value tower already matches BT4's **shape exactly** — 128 per-square projection, 8192 flat,
128 hidden. There is nothing to import there. The divergence is *how the three towers are split*:

- **BT4 splits by TARGET and shares across OUTPUT TYPE.** One tower per value target; within a
  tower, WDL + error + categorical all read the same 128-d hidden.
- **We split by OUTPUT TYPE.** `value_wdl`, `value_sf_eval` and `value_categorical` are three
  private towers, so a categorical loss reaches `value_wdl`'s representation **not at all** —
  measured: exactly zero gradient on `value_wdl.net[0].weight`. It supervises only the shared trunk.

`categorical_head_coupled` (PR #397) implements BT4's topology exactly:
`nn.Linear(value_wdl.hidden_dim=128, CATEGORICAL_HEAD_BINS=32)` = **4,128 params**, matching BT4's
4,096 + 32 parameter for parameter, at **271× less** than our standalone head. **The flag is
currently off** (`categorical_head_coupled` default; `w_categorical: 0.0` in the live yaml).

By BT4's precedent the correct attachment point is a **q-like** head, not a game-result head. Our
`value_wdl` trains on a blend that is 0% game outcome (`sf_wdl_frac` 0.69 / `search_wdl_frac` 0.31),
so it is q-like and is the right target.

## 7. What this file does NOT establish

- **lc0's categorical/value-error LOSS WEIGHTS.** A `.pb.gz` carries weights, not training config.
  The ledger's claim that `w_categorical 0.1` "matches lc0 CF-240M's lambda" is **unverified** —
  it is not sourceable from BT4 and the local `lczero-training` predates both heads. Treat the aux
  weight as an open dose, not a constant to copy.
- **Training targets.** `ValueType` in lc0's training proto is `{RESULT, BEST, PLAYED, ORIG, ROOT, ST}`;
  which one feeds `winner`/`q`/`st` is a config choice not recorded in the net.
- **Whether any of this buys Elo here.** Structural fidelity to BT4 is not evidence of strength on
  our data; the target-shaping direction is where our nulls cluster (Tier-3, Tier-4, Tier-13,
  categorical target repair).
