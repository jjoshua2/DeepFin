# Chess AI Training Framework: Implementation Specification

## Project Overview

Build a chess neural network training framework in Python/PyTorch that trains a transformer-based network primarily by playing against Stockfish. The network receives classical chess features (attack maps, pins, pawn structure) as additional input planes beyond standard piece positions, and uses auxiliary heads to model Stockfish's behavior. The goal is to learn to exploit Stockfish's known weaknesses (fortress blindness, horizon effects, closed-position overconfidence) rather than maximizing general playing strength.

This is designed as a community-runnable distributed training project (like LC0/KataGo), targeting consumer GPUs (RTX 3090/4090/5090 class, 24-32GB VRAM).

---

## Phase 1: Build in Python First

Use Python for everything initially. The bottleneck is GPU inference and CPU Stockfish computation, not tree traversal. A Python MCTS that batches network calls efficiently runs at 90%+ the speed of C++ for this use case. Once the architecture stabilizes, hot paths can be ported to C++ with pybind11 if needed.

### Core Dependencies
- `python-chess` (>= 1.10): Board logic, legal move generation, bitboard feature extraction
- `PyTorch` (>= 2.0): Network definition, training, mixed precision (BF16)
- `stockfish` pip package OR direct UCI subprocess communication
- `numpy`: Feature plane construction, replay buffer
- `tensorboard`: Training monitoring

---

## Neural Network Architecture

### Base: BT3-scale Transformer

Based on LC0's Chessformer architecture (Monroe et al., arXiv:2409.12272).

```
Encoder-only transformer:
  - 15 encoder layers
  - 768 embedding dimension
  - 24 attention heads (head depth 32)
  - FFN expansion ratio 1x-2x (NOT standard 4x — LC0 found minimal gain from higher ratios)
  - Post-LayerNorm with DeepNet initialization
  - Mish activation in FFN layers
  - 64 tokens (one per square)
  - Smolgen module for positional attention bias
  
Total parameters: ~105M (fits comfortably in 24GB VRAM in BF16)
For 32GB VRAM: can scale to BT4 size (1024 embedding, ~190M params) with gradient checkpointing
```

### Smolgen Module

Solves chess topology problem (diagonal/knight distances ≠ Euclidean distance). Generates a 64×64 attention bias matrix per layer:
- Static component: learned 64×64×h bias added to attention logits (acts as ~50% effective size increase with negligible throughput cost)
- Dynamic component: position-dependent attention logits (another ~50% effective size, ~10% throughput reduction)
- Implementation: small MLP that takes compressed board state and generates per-head attention bias matrices

Reference implementation: LC0's lczero-training repo `tf/tfprocess.py` (search for smolgen). Also CeresTrain (github.com/dje-dev/CeresTrain) has a PyTorch implementation.

### Input Representation

#### Standard LC0 planes (112 planes):
- 8 history steps × 12 planes (6 piece types × 2 colors) = 96 planes
- Castling rights (4 planes)
- En passant (1 plane)  
- Color to move (1 plane)
- Rule50 counter (1 plane)
- Repetition markers (8 planes, 1 per history step)
- All-ones plane (1 plane)

Board is always oriented from side-to-move perspective (flip for black).

#### Additional Stockfish-derived feature planes (~34 planes):

These are classical chess features computed via python-chess bitboard operations. They are NOT Stockfish's internal features (modern NNUE doesn't compute these). They're the same features the old hand-crafted classical eval used before SF16 deleted it.

**King safety attack maps (10 planes):**
- For each color (2): attack map on king zone by knights, bishops, rooks, queens (4 piece types × 2 colors = 8 planes)
- King zone squares for each color (2 planes)
- King zone = 3×3 area around king + squares in front of king

Computing with python-chess:
```python
king_sq = board.king(color)
king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
# For each piece type, compute attacks that overlap king zone
for sq in board.pieces(piece_type, not color):
    attacks = board.attacks_mask(sq)
    king_attack_plane |= (attacks & king_zone.mask)
```

**Pin and x-ray geometry (6 planes):**
- Absolute pin masks per color (2 planes): squares containing pinned pieces
- Pin ray masks per color (2 planes): the line between pinner and king through pinned piece  
- Discovered attack potential per color (2 planes): pieces that would give check/attack if blocker moved

Computing with python-chess:
```python
# Pin detection
for sq in chess.SQUARES:
    pin = board.pin_mask(color, sq)
    if pin != chess.BB_ALL:  # piece is pinned
        pinned_pieces_plane |= chess.BB_SQUARES[sq]
        pin_ray_plane |= pin
```

**Pawn structure (8 planes):**
- Passed pawns per color (2 planes)
- Isolated pawns per color (2 planes)
- Backward pawns per color (2 planes)  
- Connected pawns per color (2 planes)

These require manual bitboard computation. Example for passed pawns:
```python
# White passed pawn: no enemy pawns on same file or adjacent files ahead
for sq in board.pieces(chess.PAWN, chess.WHITE):
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    # Check files file-1, file, file+1 for enemy pawns on ranks > rank
    blocking_mask = 0
    for f in range(max(0, file-1), min(7, file+1)+1):
        for r in range(rank+1, 8):
            blocking_mask |= chess.BB_SQUARES[chess.square(f, r)]
    if not (blocking_mask & board.pieces_mask(chess.PAWN, chess.BLACK)):
        passed_pawn_plane |= chess.BB_SQUARES[sq]
```

**Mobility maps (6 planes):**
- Per piece type (knight, bishop, rook, queen, king, pawn): count of safe destination squares
- "Safe" = not defended by enemy pawns (for minor pieces) or not attacked by lower-value piece
- Encoded as: popcount at piece's square normalized to [0,1], broadcast to that square on the plane

**Outpost and space (4 planes):**
- Outpost squares per color (2 planes): squares that can't be attacked by enemy pawns and are supported by own pawns
- Space control per color (2 planes): squares behind own pawn chain

**Total input: 112 + 34 = 146 planes**

#### Feature dropout during training:
Randomly zero out 30% of the additional feature planes during training. This prevents the network from becoming fully dependent on explicit features and forces it to develop redundant internal representations. At inference, all features are present. If playing against a non-Stockfish opponent, features can be computed normally (they're position facts, not SF-specific).

### Output Heads (10 total)

All heads share the transformer trunk. Each is a small MLP (1-2 FC layers) branching from the final encoder output.

**Important training convention ("network-turn only"):**
We only store training samples on the network-to-move positions (e.g. White in our current selfplay loop). Stockfish targets (reply policy + eval) are attached to the *preceding* network-turn sample, representing Stockfish's response after the network plays its move.

#### Policy-type heads:

**1. Own policy (primary):**
- Architecture: Attention-based policy head (LC0/Chessformer style)
  - Generate query vectors (from-square) and key vectors (to-square) via linear projection from trunk output
  - Logits = scaled dot product → 64×64 matrix of from→to move scores
  - Promotion handled separately with scalar offsets
- Target: MCTS visit distribution (softmax of visit counts) from network's own turns
- Loss: Cross-entropy
- Weight: 1.0
- Active on: Network's turns only

**2. Soft policy:**
- Architecture: Identical to own policy head (separate weights)
- Target: MCTS visit distribution with temperature 2.0 applied before normalization
- Loss: KL divergence
- Weight: 0.5
- Active on: Network's turns only
- Purpose: Regularizer — prevents policy from becoming too sharp, maintains awareness of alternative moves

**3. SF reply policy imitation (MultiPV soft target):**
- Architecture: Same attention-based policy head (separate weights)
- Target: A *soft* distribution over Stockfish's **reply** moves (one ply after the network plays), derived from Stockfish MultiPV output:
  - After the network chooses and plays its move, query Stockfish on the resulting position (Stockfish to move) with `MultiPV=K`.
  - For each PV candidate move with available WDL, compute a winrate-like scalar score:
    - `score = W + 0.5 * D` (from Stockfish's reported WDL probabilities for that PV).
  - Convert candidate scores to a distribution via softmax with temperature `sf_policy_temp` (default 0.25).
  - Apply label smoothing `sf_policy_label_smooth` (default 0.05) by mixing with uniform probability over legal moves.
  - If MultiPV/WDL parsing fails, fall back to a one-hot target on Stockfish's chosen move.
- Loss: Cross-entropy against the soft target distribution
- Weight: 0.15
- Active on: Network-turn samples where an SF reply exists (game did not end after the network move)
- Purpose: Opponent modeling — forces the trunk to represent how Stockfish thinks (not just its single best move).

**4. Own future move (t+2 plies):**
- Architecture: Same attention-based policy head (separate weights)
- Target: MCTS visit distribution from the network's next move after Stockfish replies (i.e. 2 plies later).
- Loss: Cross-entropy  
- Weight: 0.15
- Active on: Positions where the next network-turn sample exists (not near game end)
- Purpose: Encourages internal look-ahead. Jenner et al. (arXiv:2406.00877) showed LC0 already develops internal representations of future board states — this head directly supervises that capability.

#### Value-type heads:

**5. WDL value (primary):**
- Architecture: Global average pooling → FC(768, 128) → FC(128, 3) → softmax
- Target: Game outcome as (win%, draw%, loss%) — 1.0/0.0/0.0 for win, 0.0/1.0/0.0 for draw, etc.
- Loss: Cross-entropy
- Weight: 1.0
- Active on: All positions

**6. SF eval prediction (after SF reply):**
- Architecture: Same as WDL head (separate weights) → 3-way softmax
- Target: Stockfish's search WDL from the *reply search* (the same search used to choose Stockfish's reply move), attached to the preceding network-turn sample.
- Loss: Cross-entropy
- Weight: 0.15
- Active on: Network-turn samples where an SF reply exists
- Purpose: Opponent value modeling. Divergence between this head and primary WDL head identifies positions where network disagrees with SF's evaluation — fortress candidates, horizon-effect positions.

**7. Categorical value:**
- Architecture: Global average pooling → FC(768, 128) → FC(128, 32) → softmax over 32 bins
- Target: Game outcome bucketed into 32 bins spanning [-1, 1], with HL-Gauss label smoothing (Gaussian centered on true value, σ=0.04)
- Loss: Cross-entropy
- Weight: 0.10
- Active on: All positions
- Purpose: Distributional RL — captures uncertainty in value estimates. Network learns that some positions have bimodal value distributions (decisive vs drawn).

**8. Network volatility:**
- Architecture: Global average pooling → FC(768, 64) → FC(64, 3) → ReLU (outputs non-negative)
- Target: `(E[|ΔW|], E[|ΔD|], E[|ΔL|])` — expected absolute change in each WDL component over next 6 plies
  - Computed from game record: `|WDL_{t+6} - WDL_t|` per component
  - Default: `WDL_t` is the network's **raw** WDL head output at time t (pre-search).
  - Optional ablation: use a **search-adjusted** WDL distribution by taking the raw draw probability and replacing (W-L) with the root search value estimate.
- Loss: Huber loss (δ=0.1)
- Weight: 0.05
- Active on: Positions where t+6 data exists
- Purpose: Predicts whether the network's own evaluation will shift dramatically.

**9. Stockfish volatility:**
- Architecture: same as Network volatility (separate weights)
- Target: `|SF_WDL_{t+6} - SF_WDL_t|` per component, where `SF_WDL_t` is Stockfish's WDL output from the SF reply search attached to network-turn sample t.
- Loss: Huber loss (δ=0.1)
- Weight: 0.05
- Active on: Positions where t+6 data exists AND both SF evals exist
- Purpose: Predicts whether Stockfish's evaluation is about to shift dramatically.

#### Other:

**10. Moves-left:**
- Architecture: Global average pooling → FC(768, 32) → FC(32, 1)
- Target: Remaining game length (normalized)
- Loss: Huber loss
- Weight: 0.02
- Active on: All positions
- Purpose: Prevents "trolling" (needlessly prolonging won games). Helps search allocate time.

### Loss Function

```python
total_loss = (
    1.0  * policy_loss              # network-turn samples only
  + 0.5  * soft_policy_loss         # network-turn samples only
  + 0.15 * sf_move_pred_loss        # network-turn samples where SF reply exists
  + 0.15 * future_move_loss         # where t+2 plies exists
  + 1.0  * wdl_value_loss           # all samples
  + 0.15 * sf_eval_pred_loss        # network-turn samples where SF reply exists
  + 0.10 * categorical_value_loss   # all samples
  + 0.05 * volatility_loss          # where t+6 exists
  + 0.05 * sf_volatility_loss       # where t+6 exists AND SF eval exists
  + 0.02 * moves_left_loss          # all samples
  + reg_weight * l2_regularization
)
```

In the current data scheme, **all recorded training samples are network-turn samples**. Stockfish targets (reply policy + eval) are attached to those samples when available, and their losses are masked by `has_sf_policy` / `has_sf_wdl`.

---

## Game Generation Pipeline

### Architecture

Run N concurrent games (N=16-64, tuned to keep GPU busy). Each game alternates between Stockfish (CPU) and the network (GPU MCTS).

```
Game Manager (Python asyncio or threading)
  ├── Stockfish Process Pool (CPU, M processes)
  │   └── Each process: UCI communication, node-limited search
  ├── MCTS Engine (GPU)  
  │   └── Batched neural network inference across all active games
  └── Training Data Writer
      └── Writes completed game records to replay buffer
```

### Game Flow

1. Position starts from opening book or random legal position
2. If it's the network's turn:
   - Run MCTS with network evaluation (batch NN calls across concurrent games)
   - Select move from visit distribution (with temperature for exploration)
   - Play the network move
   - Create a training record for the *pre-move* position containing the network policy targets and raw WDL estimate.
3. If the game did not end after the network move:
   - Query Stockfish on the resulting position via UCI `go nodes X` (node-limited)
   - Parse MultiPV to build a soft SF reply-policy target (with label smoothing)
   - Attach SF reply targets (sf policy + sf eval + sf chosen move index) to the previously created network-turn record
   - Play the Stockfish reply move
4. Repeat
5. After game ends:
   - Label all recorded network-turn positions with game outcome
   - Compute future-move targets (the next network move after the SF reply)
   - Compute volatility targets over 6 plies (t vs t+6)
   - Write to replay buffer

We do **not** store separate training samples for Stockfish-to-move positions; all useful SF supervision is attached to the preceding network-turn sample.

### MCTS Implementation

Use Gumbel MCTS (Danihelka et al., ICLR 2022) instead of PUCT:
- Guarantees policy improvement at ANY simulation budget by construction
- Critical early in training when policy is poor: vanilla MCTS fails below ~16 simulations, Gumbel works with as few as 2
- Algorithm: Gumbel-Top-k action sampling → sequential halving among candidates
- Start with low simulation count (50-100), increase as network improves

Progressive simulation budget: Start training with 50 simulations per move, gradually increase to 400-800 as network strengthens. This dramatically reduces wall-clock time for early game generation.

### Stockfish Configuration

**Primary difficulty control: node-limited search (`go nodes X`)**
- Range: 1,000 to 1,000,000 nodes
- Produces natural difficulty scaling — effective depth varies by position complexity
- More natural error profiles than fixed depth or Skill Level

**Adaptive difficulty via PID controller:**
- Target: 50-55% win rate for network (maximum information content)
- Measurement window: Exponential moving average over 50-100 games (α=0.03)
- Dead zone: ±5% around target — don't adjust if win rate between 45-55%
- Rate limiter: Cap adjustment to ≤10% node count change per period
- Anti-windup: Clamp integral term to prevent runaway at difficulty bounds
- Adjustment period: Minimum 20-50 games between changes

**Getting SF's WDL output:**
Modern Stockfish (>= SF12) outputs WDL in UCI info strings when `UCI_ShowWDL` is set to true:
```
info depth 20 ... wdl 450 500 50 ...
```
This gives win/draw/loss in per-mille directly. Use this as the SF eval prediction target.

### Opponent Mixing (optional)

Opponent mixing is optional and can be enabled later if we observe overspecialization.

- Default: 100% games against Stockfish (primary opponent, node-limited with PID)
- Optional: mix in games against older checkpoints / other engines as a regularization / generalization check

### MCTS Opponent Modeling During Search

We do **not** maintain a separate opponent-model network.

- Default: use the network's own policy for opponent moves inside MCTS.
- Optional: use the network's auxiliary opponent heads (`policy_sf`, `sf_eval`) as additional features/regularizers during training (they are trained from real Stockfish reply searches), without adding any new model.

We avoid querying real Stockfish inside tree search.

---

## Training Pipeline

### Replay Buffer

- Sliding window of most recent N game records (start with 50,000 games, grow to 500,000)
- Each record contains: all positions, MCTS data (network turns), SF data (SF turns), game outcome
- Shuffle buffer: 500,000 positions
- Batch size: 2048 (effective, via gradient accumulation if needed)

### Sampling Strategy

**Two-stage difficulty pipeline (combines KataGo surprise weighting + LC0 "diff focus"):**

Consecutive positions in a game are heavily correlated — moves 20-25 in a quiet middlegame are nearly identical. Training on all of them wastes compute and biases the network toward boring patterns. We address this with a two-stage pipeline:

**Stage 1 — Probabilistic skip (decorrelation + automatic curriculum):**
Before a position enters the training buffer, compute combined difficulty from both value AND policy surprise (LC0's "diff focus" insight — introduced in their T60 run):

```
q_surprise = |search_value - raw_network_value|  # value head was wrong
pol_surprise = KL(raw_policy || improved_policy)  # policy head was wrong
difficulty = q_surprise * diff_focus_q_weight + pol_surprise * diff_focus_pol_scale
keep_prob = clamp(difficulty * diff_focus_slope, diff_focus_min, 1.0)
```

Defaults (following LC0 T60): q_weight=6.0, pol_scale=3.5, slope=3.0, min=0.025. All are Ray Tune parameters.

Why value surprise matters: a position where the network picked the right move but misjudged the resulting position (value wrong, policy fine) is critical for value head training. Policy KLD alone misses this.

Why this is curriculum learning: early in training, the network is garbage → everything is hard → everything survives. As the network improves → easy positions get skipped → training naturally focuses on what the network still gets wrong. Self-adjusting with no manual scheduling.

**Stage 2 — Differential weighting (for survivors):**
Positions that survive the skip gate get weighted by search quality × remaining surprise:

```
base_weight = 1.0 if full_search else fast_search_policy_weight  # 0.3-0.5
surprise_multiplier = 0.5 + 0.5 * (difficulty / mean_difficulty_among_survivors)
final_policy_weight = base_weight * surprise_multiplier
value_weight = 1.0 for ALL positions (game outcome valid regardless of search depth)
```

`fast_search_policy_weight` range 0.2-0.6 (Ray Tune parameter).

**Half-position handling:**
In the current scheme we store only network-turn positions, so there is no separate "half-position" stream to manage.

### KataGo Playout Cap Randomization (constant ratio, NOT progressive)

KataGo does NOT progressively increase playouts. It uses a fixed random split on every move for the entire run:

- ~25% of moves: "full search" (200-400 sims)
- ~75% of moves: "fast search" (30-50 sims)

This ratio is CONSTANT throughout training. KataGo's ablation showed this outperformed every fixed playout count and every progressive schedule.

**What KataGo trained on:** KataGo used PUCT, so fast-search policy targets were near-useless. KataGo trained policy ONLY on full-search moves. Dropping 75% of moves for policy wasn't just about target quality — it was natural decorrelation of correlated consecutive game positions.

**What WE train on (Gumbel changes this):** Gumbel guarantees valid policy improvement at any sim budget. We use ALL moves for policy training, but filtered through the two-stage pipeline above (skip easy positions, weight survivors by quality × surprise). This is strictly better than KataGo — they discarded 75% entirely, we filter intelligently and use Gumbel's low-sim validity.

Progressive simulation budget is optional. Ramping simulations can speed up early selfplay, but it *does* change the policy-target quality distribution. If you use a progressive schedule, keep the full/fast split roughly constant (ramp both together) and treat the schedule as a tunable hyperparameter.

### Mixed Precision Training

- Use BF16 on Ampere/Blackwell GPUs (wider dynamic range than FP16, eliminates most loss-scaling issues)
- Gradient accumulation with 4-8 micro-batches for effective batch size 2048-4096
- Gradient checkpointing: reduces activation memory ~50% at ~20% speed cost (enables larger models)
- Gradient clipping: max norm 10.0 (following Chessformer)

### Optimizer

NAdamW (primary, following recent LC0 training config):
- β1 = 0.9, β2 = 0.98
- ε = 1e-7
- Weight decay: 0.0001
- Apply weight decay to embedding and encoder weights, not to biases or layer norms

SOAP (ablation alternative, following CeresTrain):
- Shampoo-like second-order optimizer
- CeresTrain reports ~30% faster convergence (iterations), ~20% wall-clock vs Adam
- Higher memory cost — may require smaller batch on consumer GPUs
- Run as Ray Tune ablation to compare vs NAdamW

Use `torch.compile(model, mode="reduce-overhead")` — CeresTrain measured 75% throughput increase (8k→14k pos/sec on 4×A100).

### Learning Rate Schedule

Start with cosine schedule with warm restarts:
- Warmup: 1500 steps linear ramp from 0 to peak LR
- Peak LR: 0.001 (scale if batch size differs)
- Cosine decay to 0.0001 over each cycle
- Monitor policy loss convergence to trigger LR reductions

LC0 community finding: "Dropping LR too soon is counterproductive, even when network quality isn't obviously improving." Fastest improvement occurs just after LR drops.

### Stochastic Weight Averaging (SWA)

Average model weights periodically for exported networks (following Chessformer). Produces smoother, more generalizable models.

---

## Game Termination Strategy

### KataGo-style Soft Resignation (recommended over hard resignation):

Do NOT resign games. Instead, when MCTS winrate for losing side drops below 5% for 5 consecutive turns:
- Continuously reduce playout count (interpolate between fast and full limits proportional to p/0.05)
- Record training samples with reduced probability: 0.1 + 0.9×(p/0.05)
- Games play to completion, preserving endgame and fortress training signal

**Why not hard resignation:** LC0's experience showed resignation creates endgame distribution bias. Past attempts to filter out "bad" positions were "all detrimental, as value head seems to need to see enough bad positions in training."

### Tablebase Rescoring

Apply Syzygy tablebase rescoring post-hoc:
- When position reaches ≤7 pieces (after capture or pawn push, no castling rights), look up WDL in Syzygy tables
- Relabel game result for ALL preceding positions to tablebase-proven result
- This provides perfect ground truth for endgame evaluation at zero inference cost

### Fortress Position Handling

**Critical: Never adjudicate games using Stockfish's evaluation when training to exploit SF's weaknesses.**

If materially stronger side's SF eval says "winning" but the position is actually a fortress (drawn), adjudicating with SF's eval gives incorrect training signal. This is exactly the scenario you're training to exploit.

Instead:
- Play fortress-candidate positions (material imbalance + blocked pawn structure) to completion
- Mine fortress positions from known databases for supplementary training
- Guid & Bratko (2012), "Detecting Fortresses in Chess," provides fortress detection algorithms

---

## Curriculum Design for Exploiting Stockfish Weaknesses

### Documented SF Failure Modes:

1. **Fortress blindness**: SF's NNUE is static evaluation — cannot reason about impossibility of progress. At 70-ply depth after 3+ hours, SF still claimed +1.14 for a known fortress position.

2. **Horizon-effect strategic failures**: SF's aggressive pruning (LMR, null move, futility) systematically cuts off slow, counter-intuitive moves. GitHub issue #1962: SF was "completely blind" to AlphaZero's killer pawn push g5-g4 in a famous game.

3. **Closed position overconfidence**: "In positions that are closed where neither side can make progress, computers will tend to overestimate their chances." (GM Naroditsky)

### Position Mining by Evaluation Divergence

**NOVEL — no published work does this explicitly:**
Select training positions where SF's evaluation at depth N diverges from depth N+K:
```
divergence = |sf_eval(depth=20) - sf_eval(depth=40)|
```
Positions with high divergence are where SF's shallow search is most wrong — exactly the positions to focus training on. Pre-compute these offline from large game databases.

### Curriculum Phases (overlapping, not sequential):

**Phase 1 (from start):** Endgame positions (≤7 pieces) with tablebase verification. Network learns accurate material evaluation and basic endgame technique. Use positions from SF games where SF's eval disagreed with TB result.

**Phase 2 (after ~10K games):** Add middlegame positions with tactical themes. Injected features (pins, forks, discovered attacks) provide explicit supervision for the hardest-to-learn concepts.

**Phase 3 (after ~50K games):** Full games against calibrated Stockfish with emphasis on fortress-candidate and closed positions. Maintain supplementary training set of known fortress positions.

All phases overlap with increasing weights. This follows KataGo's finding that simultaneous mixed training is more effective than strict sequencing.

### Monitoring for Overspecialization

**Critical risk:** Training against a fixed opponent produces a narrow specialist.

Monitor both:
1. SF-specific win rate (should increase)
2. General chess ability via standard puzzle test suite (should not decrease significantly)
3. Periodically play against LC0 and earlier network versions

If puzzle accuracy drops >5% while SF win rate rises, increase LC0/self-play mixing ratio.

---

## Data Format

Each training record (per position) contains:

```python
@dataclass
class TrainingPosition:
    # Board state (always network-to-move positions in current scheme)
    input_planes: np.ndarray     # (146, 8, 8) float32/float16
    legal_move_mask: np.ndarray  # (4672,) bool — LC0-style move encoding

    # Sampling metadata
    priority: float              # surprise/difficulty weight (for prioritized sampling)
    has_policy: bool             # whether this position has a valid policy target (e.g. full-search vs fast-search)

    # Provenance
    game_id: int
    ply: int
    is_network_turn: bool        # always True in current scheme (kept for compatibility)

    # Policy targets
    policy_target: np.ndarray                 # (4672,) float32 — MCTS-improved own policy
    policy_soft_target: np.ndarray | None     # (4672,) float32 — temp-2 regularizer target
    future_policy_target: np.ndarray | None   # (4672,) float32 — next network move after SF reply (t+2 plies)

    # Stockfish targets (attached to the network-turn sample)
    sf_policy_target: np.ndarray | None       # (4672,) float32 — MultiPV-derived soft SF reply policy (with label smoothing)
    sf_wdl: np.ndarray | None                 # (3,) float32 — Stockfish WDL from the reply search (same search used to pick SF move)
    sf_move_index: int | None                 # LC0 policy index of SF's chosen reply move

    # Value targets
    wdl_target: int                           # {0,1,2} from the network POV (win/draw/loss)
    categorical_target: np.ndarray | None      # (32,) HL-Gauss target (optional)

    # Aux targets
    moves_left: float | None
    volatility_target: np.ndarray | None      # (3,) float32 — network WDL volatility over 6 plies
    sf_volatility_target: np.ndarray | None   # (3,) float32 — SF WDL volatility over 6 plies
```

---

---

## Ray Tune Integration (From Day One)

All training runs go through Ray Tune with ASHA early stopping and Optuna Bayesian search. This is not optional — it's architectural. Every hyperparameter (loss weights, LR, architecture toggles, optimizer choice) is a Ray Tune config parameter.

Key benefits:
- ASHA scheduler aggressively kills bad trials after grace period (10 epochs), saving >80% of compute on failed configs
- Ablation studies are just restricted search spaces with 2 options
- All metrics (policy loss, value loss, SF move accuracy, win rate, puzzle accuracy, volatility calibration) are reported per epoch via `tune.report()`
- Checkpointing enables resume from any point
- Results are comparable across runs with consistent logging

Priority ablation studies (run on reduced 5-layer network, 20 epochs, 1000 games):
1. Smolgen on vs off
2. Volatility heads weight 0 vs 0.05 vs 0.10
3. SF move prediction head weight 0 vs 0.15
4. Feature dropout 0 vs 0.3
5. NAdamW vs SOAP optimizer
6. NonLinear Attention on vs off
7. Categorical bins 32 vs 64 vs 128
8. Diff focus skip on vs off (skip gate disabled = all positions kept, weighting still applies)

---

## Ceres Engine Compatibility

The tournament-ready deployment path is NOT a new engine. It's exporting trained weights to **Ceres** (github.com/dje-dev/Ceres), a production C# MCTS engine with ONNX inference backend.

CeresTrain (github.com/dje-dev/CeresTrain) has a PyTorch training backend that our architecture should align with:

- Postnorm encoder stack ✓
- Attention-based policy head (scaled dot product from→to) ✓
- Smolgen positional encoding ✓
- FFN output heads for value/policy ✓
- Export via ONNX (`torch.onnx.export`) ✓

Ceres-specific architecture features to incorporate as toggleable options:
- **NLA (NonLinear Attention)**: Adds `Linear → Mish → Linear` preprocessing of K, Q, V projections before dot-product attention. Reported modest gains.
- **SOAP optimizer**: Shampoo-like second-order optimizer. Ceres reports ~30% faster convergence (iterations), ~20% wall-clock vs Adam.
- **One-hot rank/file positional encoding**: Ceres uses simple one-hot vectors for positional encoding (alternative to smolgen). Include as ablation option.

The 34 additional feature planes are handled via either: (a) modifying Ceres to accept wider input, or (b) training a distillation step that maps 146→112 plane features.

---

## Implementation Order

### Step 1: Board representation and feature extraction
Build the 146-plane input encoder. Test against known positions. Verify feature planes match expected bitboard patterns.

### Step 2: Network architecture
Implement the transformer with smolgen and all 10 heads in PyTorch. Verify forward pass produces correct-shaped outputs. Test with random inputs. **Test ONNX export immediately.**

### Step 3: Ray Tune scaffolding
Wrap training in `train_trial(config)`. Define search space. Verify ASHA works with dummy metrics. All subsequent steps run through Tune.

### Step 4: Simple game generation (no MCTS)
Play games using network's raw policy (1-node, no search) against Stockfish at very low strength. This validates the game loop, UCI communication, and training data recording without MCTS complexity.

### Step 5: Gumbel MCTS
Implement Gumbel MCTS with batched inference. Start with 50 simulations. Verify policy improvement over raw policy.

### Step 6: Training loop
Connect game generation to training. Implement replay buffer, sampling, loss computation with per-position masking. Train on small scale (single GPU, ~1000 games).

### Step 7: Ablation studies
Run the 8 key ablations on 5-layer network, 20 epochs. Use ASHA to abort bad configs. Set architecture defaults from results.

### Step 8: Adaptive difficulty and curriculum
Add PID controller for SF difficulty. Add fortress position mining. Add evaluation divergence position selection.

### Step 9: ONNX export to Ceres
Export trained weights. Validate inference outputs match between PyTorch and ONNX runtime. Test loading into Ceres.

### Step 10: Distributed training infrastructure
Distributed client/server selfplay is implemented (server + workers + learner). Further work is optional hardening and scaling (auth, moderation/quarantine, manifest/versioning, arena gating).

---

## Key Technical References

**Architecture:**
- Monroe et al., "Mastering Chess with a Transformer Model" (arXiv:2409.12272) — Chessformer/BT4 architecture details, all head definitions, smolgen
- LC0 blog "Transformer Progress" (lczero.org/blog/2024/02/transformer-progress/) — BT1-BT4 progression, Elo results
- Wu, "Accelerating Self-Play Learning in Go" (2020) — KataGo auxiliary targets, playout cap randomization, policy surprise weighting

**Adversarial training:**
- Wang et al., "Adversarial Policies Beat Superhuman Go AIs" (ICML 2023, arXiv:2211.00241) — A-MCTS, training against frozen opponent, only training on adversary's turns

**MCTS:**
- Danihelka et al., "Policy Improvement by Planning with Gumbel" (ICLR 2022) — Gumbel MCTS, works with 2 simulations

**Interpretability:**
- Jenner et al., "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network" (arXiv:2406.00877) — LC0 internally represents future moves
- McGrath et al., "Acquisition of Chess Knowledge in AlphaZero" (PNAS 2022) — concept emergence order (material early, king safety late)

**Stockfish internals:**
- SF's Skill Level: MultiPV with minimum 4 candidates, depth = 1 + Skill Level, randomized score perturbation
- UCI_LimitStrength: maps UCI_Elo (1320-3190) to fractional Skill Level
- Node-limited (`go nodes X`): preferred difficulty control, produces natural variable depth
- UCI_ShowWDL: outputs win/draw/loss in per-mille in info strings

**Training techniques:**
- Czech et al., "AlphaVile" (arXiv:2304.14918) — WDL + moves-left = +180 Elo over AZ baseline
- LC0 "diff focus" (lczero-training chunkparser.py, T60 run) — probabilistic skip of easy positions using combined value + policy KLD, automatic curriculum learning
- Shao et al., "DeepSeekMath/GRPO" (arXiv:2402.03300) — group-relative advantage normalization, applicable to game RL
- LC0 WDL head: +53.88 Elo (PR #635), moves-left head: +15 Elo (PR #961)

**ChessCoach finding (important negative result):**
Butner (2021) tried KataGo's opponent reply prediction head in chess and found it harmful: "too painful for the network to guess badly and give a reply for the wrong move." Our setup differs — we train an *opponent policy imitation* head using Stockfish's *actual* search output (MultiPV-derived soft target, with one-hot fallback), not a hypothetical reply distribution conditioned on a move the network never played.

---

## Novel Contributions (no published work does these)

1. **Stockfish feature injection as input planes** — classical chess features (attack maps, pins, pawn structure) provided as additional transformer input, with feature dropout to prevent dependence

2. **SF eval prediction as auxiliary head** — predicting SF's search WDL (not static eval) builds opponent model; divergence from primary WDL head is a fortress/horizon-effect detector

3. **Evaluation divergence position mining** — `|sf_eval(depth_low) - sf_eval(depth_high)| > threshold` to find positions where SF's shallow search misleads, for targeted training

4. **Volatility head** — predicting `E[|ΔW|], E[|ΔD|], E[|ΔL|]` over next 6 plies. Directly actionable at inference for search allocation. Decomposition by WDL component identifies TYPE of uncertainty.

5. **Opponent-conditioned chess training** — no published work conditions a chess engine's play on the specific identity of the opponent

---

## Open Questions to Resolve Empirically

1. Whether depth-divergent position mining produces measurably better training data than uniform sampling
2. Optimal ratio of injected features to learned representations (KataGo found features helped "noticeably" but were "small fraction" of total gains)
3. Whether including SF's 1-node NNUE eval as an input plane helps or creates anchoring bias (make togglable, run ablation after a few days of training)
4. How GRPO-style policy updates compare to standard MCTS-improved policy targets
5. Optimal volatility prediction horizon (6 plies is a starting guess from BT4's short-term value head)