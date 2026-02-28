# Chess Anti-Engine: Implementation Prompt

You are building a chess neural network training framework in Python/PyTorch. The network trains by playing games against Stockfish, learning to exploit its known weaknesses (fortress blindness, horizon effects, closed-position overconfidence). This is a consumer-GPU project (RTX 3090/4090/5090, 24-32GB VRAM).

Build this in phases. Phase 1 is entirely Python. No C++ engine, no Rust, no compiled language. The bottleneck is GPU inference and CPU Stockfish computation, not tree traversal. Optimize later only if measured throughput is the bottleneck.

---

## DEPENDENCIES

```bash
pip install python-chess torch numpy tensorboard "ray[tune]" optuna onnx onnxruntime stockfish
```

- `python-chess` (>= 1.10): Board logic, legal move generation, bitboard feature extraction
- `PyTorch` (>= 2.0): Network, training, BF16, `torch.compile`
- `ray[tune]` + `optuna`: Hyperparameter tuning, ASHA early stopping, ablation studies — integrated from day one
- `onnx` + `onnxruntime`: Export path to Ceres engine for tournament play
- `stockfish`: UCI subprocess communication
- `numpy`, `tensorboard`

## CERES ENGINE COMPATIBILITY

The end-state for tournament play is NOT building a new engine. It's exporting trained weights to the **Ceres** chess engine (github.com/dje-dev/Ceres), which is a production-grade MCTS engine in C# with ONNX inference, battle-tested in TCEC.

**CeresTrain** (github.com/dje-dev/CeresTrain) already has a PyTorch training backend (16% of codebase is Python). Our architecture should align with Ceres's expected format:

- **Architecture**: Postnorm encoder stack + attention-based policy head + FFN value/policy output heads. Ceres uses embedding layer → normalization at input, simple FFN layers as output heads. Our transformer trunk is structurally compatible.
- **Smolgen**: Ceres v1.0 uses smolgen (replaced RPE). We use smolgen. ✓
- **Policy head**: Ceres uses attention-based policy (scaled dot product, from-square queries × to-square keys). We use the same. ✓ **No flat 1858-vector FC heads anywhere in this project.**
- **Export format**: ONNX. Design all modules to be `torch.onnx.export`-compatible. Avoid dynamic control flow in forward pass. Test ONNX export early.
- **Input encoding**: Ceres uses LC0-compatible 112-plane input. Our 34 additional feature planes are additive — Ceres can be modified to accept the wider input, or we can train a distillation head that produces equivalent outputs from standard 112 planes.
- **NLA (NonLinear Attention)**: Ceres adds learned nonlinear preprocessing of K, Q, V matrices before dot-product attention. Consider adding this — it's a small MLP on the projections: `Q = f(W_q · x)` where f is `Linear → Mish → Linear` instead of just `Linear`. Ceres reports modest gains. Make it a config toggle for ablation.
- **SOAP optimizer**: Ceres found SOAP (Shampoo-like) gives ~30% faster convergence (iterations) and ~20% faster wall-clock vs Adam. We start with NAdamW (proven in LC0), add SOAP as a Ray Tune ablation option.

**Export workflow (future, after training works):**
```python
dummy_input = torch.randn(1, 146, 8, 8).to(device)
torch.onnx.export(model, dummy_input, "chess_anti_engine.onnx",
                  input_names=["input_planes"],
                  output_names=["policy", "wdl", "moves_left"],
                  dynamic_axes={"input_planes": {0: "batch"}})
```

## RAY TUNE INTEGRATION

**Every training run goes through Ray Tune from day one.** Even single-config runs use Tune's logging, checkpointing, and metric tracking. This gives you ablation studies, early abort of bad configs, and hyperparameter search for free.

**Architecture:**
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# The training function Ray Tune calls per trial
def train_trial(config):
    """One complete training run with given hyperparameters."""
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    replay_buffer = ReplayBuffer(config["buffer_size"])
    game_gen = GameGenerator(model, config)
    
    for epoch in range(config["max_epochs"]):
        # Generate games
        new_positions = game_gen.play_batch(config["games_per_epoch"])
        replay_buffer.add(new_positions)
        
        # Train
        metrics = train_epoch(model, optimizer, replay_buffer, config)
        
        # Report to Ray Tune (enables ASHA early stopping)
        tune.report(
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            sf_move_acc=metrics["sf_move_accuracy"],
            sf_eval_loss=metrics["sf_eval_loss"],
            volatility_loss=metrics["volatility_loss"],
            win_rate_vs_sf=metrics["win_rate"],
            puzzle_accuracy=metrics["puzzle_acc"],
            total_loss=metrics["total_loss"],
        )
```

**Search space — what to tune:**
```python
search_space = {
    # Architecture (ablation)
    "num_layers": tune.choice([10, 12, 15]),
    "embed_dim": tune.choice([512, 768]),
    "num_heads": tune.choice([16, 24]),
    "ffn_multiplier": tune.choice([1, 2]),
    "use_smolgen": tune.choice([True, False]),        # Ablation: smolgen vs no smolgen
    "use_nla": tune.choice([True, False]),             # Ablation: NonLinear Attention
    "feature_dropout": tune.uniform(0.1, 0.5),        # How much to drop extra features
    
    # Loss weights (critical to tune)
    "w_sf_move": tune.uniform(0.05, 0.30),
    "w_sf_eval": tune.uniform(0.05, 0.30),
    "w_categorical": tune.uniform(0.05, 0.20),
    "w_volatility": tune.uniform(0.01, 0.15),
    "w_soft_policy": tune.uniform(0.2, 1.0),
    "w_future_move": tune.uniform(0.05, 0.30),
    
    # Optimizer
    "optimizer": tune.choice(["nadamw", "soap"]),
    "lr": tune.loguniform(1e-4, 5e-3),
    "weight_decay": tune.loguniform(1e-5, 1e-3),
    "warmup_steps": tune.choice([500, 1000, 1500, 2500]),
    
    # Training
    "batch_size": tune.choice([1024, 2048, 4096]),
    "mcts_simulations": tune.choice([50, 100, 200]),
    "playout_cap_fraction": tune.uniform(0.15, 0.35),  # Fraction getting full search
    "fast_search_policy_weight": tune.uniform(0.2, 0.6),  # Policy training weight for fast-search moves (full=1.0)
    
    # Diff focus (LC0-inspired skip + weighting)
    "diff_focus_q_weight": tune.uniform(3.0, 9.0),     # Value surprise sensitivity
    "diff_focus_pol_scale": tune.uniform(2.0, 5.0),    # Policy surprise sensitivity
    "diff_focus_slope": tune.uniform(1.5, 5.0),        # Skip steepness
    "diff_focus_min": tune.uniform(0.01, 0.05),        # Min keep probability for easy positions
    
    # HL-Gauss
    "categorical_bins": tune.choice([32, 64, 128]),
    "hlgauss_sigma": tune.uniform(0.02, 0.08),
    
    # Game generation
    "sf_initial_nodes": tune.choice([5000, 10000, 25000]),
    "opponent_mix_sf": tune.uniform(0.5, 0.85),         # % games vs SF
    
    # Fixed
    "max_epochs": 200,
    "games_per_epoch": 500,
    "buffer_size": 200000,
}
```

**ASHA scheduler for early stopping:**
```python
scheduler = ASHAScheduler(
    time_attr="training_iteration",
    metric="total_loss",
    mode="min",
    max_t=200,           # Max epochs
    grace_period=10,     # Don't kill before 10 epochs
    reduction_factor=3,  # Aggressive: keep top 1/3 at each rung
)

search_algo = OptunaSearch(metric="total_loss", mode="min")

tuner = tune.Tuner(
    tune.with_resources(train_trial, {"cpu": 4, "gpu": 1}),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        search_alg=search_algo,
        num_samples=50,      # Number of trials
    ),
    param_space=search_space,
    run_config=tune.RunConfig(
        name="chess_anti_engine",
        storage_path="~/ray_results",
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=5,
            num_to_keep=3,
        ),
    ),
)
results = tuner.fit()
```

**Key ablation studies to run early (before full-scale training):**

| Ablation | What it tests | Expected signal in ~20 epochs |
|----------|---------------|-------------------------------|
| `use_smolgen=True` vs `False` | Is chess topology encoding worth the compute? | Value loss divergence |
| `w_volatility=0` vs `0.05` vs `0.10` | Does volatility head help or hurt other heads? | Policy accuracy change |
| `w_sf_move=0` vs `0.15` | Does modeling SF's moves help own play? | Win rate vs SF |
| `feature_dropout=0` vs `0.3` | Does feature dropout improve generalization? | Puzzle accuracy |
| `optimizer=nadamw` vs `soap` | Which converges faster? | Loss curve comparison |
| `use_nla=True` vs `False` | Does NonLinear Attention help? | Value accuracy |
| `categorical_bins=32` vs `64` vs `128` | How many bins for distributional value? | Categorical loss |
| `diff_focus_min=1.0` (no skip) vs `0.025` | Does probabilistic skip help? | Policy + value loss curves |

Run these on reduced scale (5-layer network, 1000 games, 20 epochs) to get directional signal before committing to full training.

---

## PHASE 1: IMPLEMENT THESE MODULES IN ORDER

### Module 1: Input Feature Encoder

Build a function `encode_position(board: chess.Board) -> np.ndarray` that produces a `(C, 8, 8)` float32 tensor.

**Standard LC0 planes (112 planes):**

LC0 uses 112 input planes. The board is ALWAYS oriented from side-to-move perspective (flip rank indices for black).

| Planes | Content |
|--------|---------|
| 0-95 | 8 history positions × 12 planes each (P,N,B,R,Q,K × white,black). Current position = planes 0-11, previous = 12-23, etc. Binary: 1 if piece present on square. |
| 96-99 | Castling rights: us-kingside, us-queenside, them-kingside, them-queenside. All-ones or all-zeros plane each. |
| 100 | En passant file. All-zeros if no EP, otherwise ones on the EP target file. |
| 101 | Color to move. All-ones = white, all-zeros = black. (After orientation flip this is always all-ones for side-to-move.) |
| 102 | Rule50 counter. Scalar normalized to [0,1] broadcast to full plane. |
| 103-110 | Repetition count per history step. All-ones if that history position has been repeated, all-zeros otherwise. |
| 111 | All-ones plane. Constant bias. |

For initial implementation, use 1-step history (just current position = 12 planes) + metadata planes. Full 8-step history can be added later.

**Additional feature planes (~34 planes):**

These provide classical chess knowledge as inductive bias. Compute using `python-chess` bitboard operations.

King safety attack maps (10 planes):
```python
import chess

def king_zone(king_sq):
    """3x3 area around king + 3 squares in front."""
    zone = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]
    # Add squares in front of king (toward opponent's side)
    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)
    for df in [-1, 0, 1]:
        f = king_file + df
        for dr in [1, 2]:  # 1-2 ranks ahead
            r = king_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                zone |= chess.BB_SQUARES[chess.square(f, r)]
    return zone

# For each color: king zone mask (1 plane), attacks on zone by N,B,R,Q (4 planes) = 10 total
for color in [chess.WHITE, chess.BLACK]:
    kz = king_zone(board.king(color))
    # plane: king zone squares
    zone_plane = bitboard_to_plane(kz)
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        attack_plane = 0
        for sq in board.pieces(piece_type, not color):
            attack_plane |= (board.attacks_mask(sq) & kz)
        # Convert attack_plane bitboard to 8x8 numpy plane
```

Pin and x-ray geometry (6 planes):
```python
# python-chess pin detection
for color in [chess.WHITE, chess.BLACK]:
    pinned_mask = 0
    pin_ray_mask = 0
    for sq in chess.SQUARES:
        if board.piece_at(sq) and board.color_at(sq) == color:
            pin = board.pin_mask(color, sq)
            if pin != chess.BB_ALL:  # BB_ALL means not pinned
                pinned_mask |= chess.BB_SQUARES[sq]
                pin_ray_mask |= pin
    # pinned_mask → plane, pin_ray_mask → plane
    
    # Discovered attack potential: pieces that block check on opponent king
    # If moving them would give discovered check
    discovered_mask = 0
    opp_king = board.king(not color)
    if opp_king is not None:
        for sq in chess.SQUARES:
            if board.piece_at(sq) and board.color_at(sq) == color:
                # Remove piece, check if any slider now attacks opp king
                board_copy = board.copy()
                board_copy.remove_piece_at(sq)
                if board_copy.is_attacked_by(color, opp_king):
                    discovered_mask |= chess.BB_SQUARES[sq]
```

Pawn structure (8 planes): passed, isolated, backward, connected × per color.

```python
def passed_pawns(board, color):
    """Pawns with no enemy pawns ahead on same or adjacent files."""
    passed = 0
    direction = 1 if color == chess.WHITE else -1
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)
    for sq in board.pieces(chess.PAWN, color):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        blocking = 0
        for f in range(max(0, file-1), min(7, file+1)+1):
            for r in range(rank + direction, 8 if direction == 1 else -1, direction):
                blocking |= chess.BB_SQUARES[chess.square(f, r)]
        if not (blocking & enemy_pawns):
            passed |= chess.BB_SQUARES[sq]
    return passed

def isolated_pawns(board, color):
    """Pawns with no friendly pawns on adjacent files."""
    isolated = 0
    own_pawns = board.pieces_mask(chess.PAWN, color)
    for sq in board.pieces(chess.PAWN, color):
        file = chess.square_file(sq)
        adjacent_files = 0
        if file > 0:
            adjacent_files |= chess.BB_FILES[file - 1]
        if file < 7:
            adjacent_files |= chess.BB_FILES[file + 1]
        if not (adjacent_files & own_pawns):
            isolated |= chess.BB_SQUARES[sq]
    return isolated
```

Mobility (6 planes): For each piece type, count legal destination squares for each piece, normalize by max possible (N:8, B:13, R:14, Q:27, K:8, P:4), place normalized value at piece's square.

Outpost/space (4 planes): Outpost = square that can't be attacked by enemy pawns, supported by own pawn. Space = squares behind own pawn chain in center files.

**Total: ~146 input planes.**

**Feature dropout:** During training, zero out the 34 additional planes with 30% probability per sample. At inference, always include all planes. This prevents the network from depending exclusively on explicit features.

**Utility function needed:**
```python
def bitboard_to_plane(bb: int) -> np.ndarray:
    """Convert a python-chess bitboard (64-bit int) to 8x8 numpy array."""
    plane = np.zeros((8, 8), dtype=np.float32)
    for sq in chess.scan_reversed(bb):
        plane[chess.square_rank(sq)][chess.square_file(sq)] = 1.0
    return plane
    # Faster version using bit manipulation:
    # return np.unpackbits(np.array([bb], dtype=np.uint64).view(np.uint8))[:64].reshape(8,8).astype(np.float32)
```

### Module 2: Neural Network Architecture

**Transformer encoder, BT3-scale.**

```
Architecture:
  - Input: (batch, C_in, 8, 8) → reshape to (batch, 64, C_in) sequence of 64 square tokens
  - Linear embedding: C_in → 768
  - 15 encoder layers, each:
      - Multi-head self-attention: 768 dim, 24 heads (head_dim=32)
      - Post-LayerNorm (not Pre-LN)
      - FFN: 768 → 1536 → 768 (expansion ratio 2, NOT standard 4x — LC0 found minimal gain beyond 2x)
      - Mish activation in FFN (not GELU, not ReLU)
      - Attention bias from smolgen module added to attention logits
  - Output: (batch, 64, 768)
  
Total: ~105M parameters
Mixed precision: BF16 on Ampere+ GPUs
```

**Smolgen module** (critical for chess — solves topology problem where diagonal/knight distances ≠ Euclidean):

Generates a learned 64×64 attention bias matrix per layer per head. Chess has fixed topology: a1 is always "close" to b2 diagonally but "far" in grid distance. Smolgen encodes this.

```python
class Smolgen(nn.Module):
    """Generate 64x64 attention bias from position and board state."""
    def __init__(self, embed_dim, num_heads, smolgen_hidden=256):
        super().__init__()
        # Static component: learned bias per head (no input dependency)
        self.static_bias = nn.Parameter(torch.zeros(num_heads, 64, 64))
        
        # Dynamic component: input-dependent bias
        # Compress board state → small latent → generate bias
        self.compress = nn.Linear(embed_dim, smolgen_hidden)
        self.gen_bias = nn.Sequential(
            nn.Linear(smolgen_hidden, smolgen_hidden),
            nn.Mish(),
            nn.Linear(smolgen_hidden, num_heads * 64)  # per-head, per-query bias
        )
        # This generates query-side bias; key-side comes from outer product
        # Full implementation: compress to latent, generate Q-side and K-side vectors,
        # outer product gives 64x64 per head
        
    def forward(self, x):
        # x: (batch, 64, embed_dim)
        # Global pool → compress → generate
        pooled = x.mean(dim=1)  # (batch, embed_dim)
        latent = self.compress(pooled)  # (batch, hidden)
        bias_flat = self.gen_bias(latent)  # (batch, num_heads * 64)
        # Reshape and create outer product for 64x64 bias
        # ... (see LC0/CeresTrain implementations for exact formulation)
        return self.static_bias.unsqueeze(0) + dynamic_bias
```

Reference implementations: CeresTrain (github.com/dje-dev/CeresTrain) has PyTorch smolgen. LC0's lczero-training repo `tf/tfprocess.py` has TensorFlow version. The key insight: ~50% effective size increase with ~10% throughput cost for static component; another ~50% for dynamic.

**Policy encoding — AlphaZero 73-plane format:**

AlphaZero/LC0 encode moves as 8×8×73 = 4,672 logits (though only ~1,858 correspond to legal chess moves after removing impossible combinations).

| Planes 0-55 | "Queen moves": 7 distances × 8 directions = 56 planes. Covers all sliding moves + 1-square king/pawn moves. |
|---|---|
| Planes 56-63 | Knight moves: 8 possible L-shaped jumps. |
| Planes 64-72 | Underpromotions: 3 piece types (N,B,R) × 3 directions (left-capture, forward, right-capture) = 9 planes. Queen promotion uses the normal queen-move planes. |

The from-square is encoded by position in the 8×8 grid. The to-square is implicit from direction + distance.

LC0 historically mapped this to a flat 1858-length vector via FC layer — we do NOT use that approach. Instead, use the attention-based policy head (Chessformer/BT4 style):
```python
class AttentionPolicyHead(nn.Module):
    """Generate move logits via scaled dot-product between from-square and to-square."""
    def __init__(self, embed_dim, policy_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, policy_dim)  # from-square
        self.key_proj = nn.Linear(embed_dim, policy_dim)    # to-square
        self.scale = policy_dim ** -0.5
        # Promotion handling: small MLP for promotion type given from/to
        self.promo_head = nn.Linear(policy_dim * 2, 3)  # N, B, R underpromotions
        
    def forward(self, x):
        # x: (batch, 64, embed_dim) — one token per square
        queries = self.query_proj(x)  # (batch, 64, policy_dim)
        keys = self.key_proj(x)       # (batch, 64, policy_dim)
        # Logits for each from→to pair
        logits = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale  # (batch, 64, 64)
        # This gives 4096 from→to pairs; mask illegal moves
        # Promotion logits: for pawns on 7th rank, concat from+to embeddings → promo_head
        return logits  # Shape details depend on move encoding choice
```

Do NOT use a flat 1858-vector FC policy head. This is LC0's legacy encoding from pre-2019 and is strictly inferior (+50 Elo for attention policy in LC0 tests, even larger gains in Chessformer). The attention-based head is the only policy architecture we use. It also aligns with CeresTrain's implementation, enabling future weight portability to the Ceres engine.

### 10 Output Heads

All heads branch from the shared transformer trunk output `(batch, 64, 768)`.

**Important training convention ("network-turn only"):**
We only store training samples on the network-to-move positions. Stockfish targets (reply policy + eval) are attached to the *preceding* network-turn sample, representing Stockfish's response after the network plays its move.

```python
class ChessNet(nn.Module):
    def __init__(self):
        self.trunk = TransformerEncoder(...)  # 15 layers, 768 dim
        
        # Policy heads (4)
        self.policy_own = AttentionPolicyHead(768)       # MCTS visits, network turn
        self.policy_soft = AttentionPolicyHead(768)      # MCTS visits @ temp 2.0, network turn
        self.policy_sf = AttentionPolicyHead(768)        # SF reply policy imitation (MultiPV soft target), attached to network turn
        self.policy_future = AttentionPolicyHead(768)    # network's next move after SF reply (t+2 plies)
        
        # Value heads (3)
        self.value_wdl = ValueHead(768, 3)               # WDL softmax, trained on all samples
        self.value_sf_eval = ValueHead(768, 3)           # SF reply eval (WDL) attached to network turn
        self.value_categorical = ValueHead(768, 32)      # 32-bin categorical, trained on all samples
        
        # Volatility heads (2)
        self.volatility = VolatilityHead(768, 3)         # network WDL volatility over 6 plies
        self.sf_volatility = VolatilityHead(768, 3)      # SF WDL volatility over 6 plies
        
        # Moves-left head (1)
        self.moves_left = ScalarHead(768, 1)
    
    def forward(self, x):
        trunk_out = self.trunk(x)  # (batch, 64, 768)
        return {
            'policy_own': self.policy_own(trunk_out),
            'policy_soft': self.policy_soft(trunk_out),
            'policy_sf': self.policy_sf(trunk_out),
            'policy_future': self.policy_future(trunk_out),
            'wdl': self.value_wdl(trunk_out),
            'sf_eval': self.value_sf_eval(trunk_out),
            'categorical': self.value_categorical(trunk_out),
            'volatility': self.volatility(trunk_out),
            'sf_volatility': self.sf_volatility(trunk_out),
            'moves_left': self.moves_left(trunk_out),
        }

class ValueHead(nn.Module):
    """Global average pool → FC → FC → softmax."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Mish(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        pooled = x.mean(dim=1)  # (batch, embed_dim) — global average pool over 64 squares
        return self.net(pooled)

class VolatilityHead(nn.Module):
    """Predict E[|ΔW|], E[|ΔD|], E[|ΔL|] — expected absolute WDL shift over 6 plies."""
    def __init__(self, embed_dim, outputs=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Mish(),
            nn.Linear(64, outputs),
            nn.ReLU(),  # Outputs are non-negative (absolute values)
        )
    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.net(pooled)

class ScalarHead(nn.Module):
    def __init__(self, embed_dim, outputs=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.Mish(),
            nn.Linear(32, outputs),
        )
    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.net(pooled)
```

### Module 3: Loss Function

```python
def compute_loss(outputs, targets):
    """
    outputs: dict from model forward pass
    targets: dict with training targets

    NOTE: In the current scheme, all recorded samples are network-turn samples.
    Stockfish targets (reply policy + eval) are attached to these samples when available.
    """
    has_policy = targets.get('has_policy', 1.0)
    has_soft = targets.get('has_soft', 0.0)
    has_future = targets['has_future'].float()             # 1 if next network sample exists (t+2 plies)
    has_volatility = targets['has_volatility'].float()     # 1 if t+6 data exists
    has_sf_policy = targets.get('has_sf_policy', 0.0)      # 1 if SF reply exists
    has_sf_wdl = targets.get('has_sf_wdl', 0.0)            # 1 if SF reply eval exists
    has_sf_volatility = targets.get('has_sf_volatility', 0.0)

    # Policy losses
    policy_loss = cross_entropy(outputs['policy_own'], targets['mcts_visits']) * has_policy
    soft_policy_loss = kl_div(
        F.log_softmax(outputs['policy_soft'], dim=-1),
        targets['mcts_visits_temp2']
    ) * has_soft
    future_loss = cross_entropy(outputs['policy_future'], targets['future_visits']) * has_future

    # Stockfish prediction losses (attached to network-turn samples)
    sf_move_loss = soft_cross_entropy(outputs['policy_sf'], targets['sf_policy_target']) * has_sf_policy
    sf_eval_loss = soft_cross_entropy(outputs['sf_eval'], targets['sf_wdl']) * has_sf_wdl

    # Value losses
    wdl_loss = cross_entropy(outputs['wdl'], targets['wdl_target'])
    categorical_loss = cross_entropy(outputs['categorical'], targets['categorical_target'])

    # Volatility losses
    volatility_loss = F.huber_loss(
        outputs['volatility'], targets['volatility_target'], delta=0.1, reduction='none'
    ).mean(dim=-1) * has_volatility
    sf_volatility_loss = F.huber_loss(
        outputs['sf_volatility'], targets['sf_volatility_target'], delta=0.1, reduction='none'
    ).mean(dim=-1) * has_sf_volatility

    # Moves-left loss
    mlh_loss = F.huber_loss(outputs['moves_left'].squeeze(), targets['moves_left'])

    total = (
        1.0  * masked_mean(policy_loss, has_policy) +
        0.5  * masked_mean(soft_policy_loss, has_soft) +
        0.15 * masked_mean(sf_move_loss, has_sf_policy) +
        0.15 * masked_mean(future_loss, has_future) +
        1.0  * wdl_loss.mean() +
        0.15 * masked_mean(sf_eval_loss, has_sf_wdl) +
        0.10 * categorical_loss.mean() +
        0.05 * masked_mean(volatility_loss, has_volatility) +
        0.05 * masked_mean(sf_volatility_loss, has_sf_volatility) +
        0.02 * mlh_loss.mean()
    )
    return total

# Volatility ablation note:
# The network volatility target can be computed either from the raw network WDL head
# output (default) or from a search-adjusted WDL distribution (tunable via volatility_source).

def masked_mean(loss, mask):
    """Mean of loss over non-zero mask entries."""
    return (loss * mask).sum() / mask.sum().clamp(min=1)
```

**Categorical value target (HL-Gauss):**

The categorical value head uses HL-Gauss label smoothing (Imani et al. 2018, validated in chess by Google DeepMind's "Stop Regressing" paper, arXiv:2403.03950). Instead of one-hot target at the bin containing the true value, create a Gaussian distribution centered on the true value:

```python
def hlgauss_target(value, num_bins=32, vmin=-1.0, vmax=1.0, sigma=0.04):
    """Create HL-Gauss categorical target distribution."""
    bin_edges = np.linspace(vmin, vmax, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # CDF at each edge
    from scipy.stats import norm
    cdf_values = norm.cdf(bin_edges, loc=value, scale=sigma)
    # Probability mass in each bin
    probs = cdf_values[1:] - cdf_values[:-1]
    probs = probs / probs.sum()  # Normalize
    return probs
```

### Module 4: Game Generation Pipeline

```
Architecture:
  Game Manager (Python, asyncio or threading)
    ├── Stockfish Process Pool (M UCI subprocesses on CPU)
    ├── MCTS Engine (batched GPU inference)
    └── Training Data Writer (→ replay buffer)
    
  Run N=16-64 concurrent games.
  Each game alternates: network's turn (MCTS on GPU) vs SF's turn (UCI on CPU).
```

**Stockfish communication via UCI:**

```python
import subprocess

class StockfishUCI:
    def __init__(self, path="stockfish", nodes=10000):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name UCI_ShowWDL value true")  # Get WDL output
        self._send("setoption name Threads value 1")
        self._send("isready")
        self._wait_for("readyok")
        self.nodes = nodes
    
    def search(self, fen):
        self._send(f"position fen {fen}")
        self._send(f"go nodes {self.nodes}")
        bestmove, wdl = None, None
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("info") and "wdl" in line:
                # Parse WDL from info string: "info depth 20 ... wdl 450 500 50"
                parts = line.split()
                wdl_idx = parts.index("wdl")
                w, d, l = int(parts[wdl_idx+1]), int(parts[wdl_idx+2]), int(parts[wdl_idx+3])
                wdl = np.array([w, d, l], dtype=np.float32) / 1000.0
            if line.startswith("bestmove"):
                bestmove = line.split()[1]
                break
        return bestmove, wdl
```

**Use `go nodes X` as primary difficulty control**, NOT depth or Skill Level:
- Produces natural difficulty scaling — effective depth varies by position complexity
- More natural error profiles than fixed depth
- Range: 1,000 to 1,000,000 nodes

**Adaptive difficulty via PID controller:**
```python
class DifficultyPID:
    """Adjust SF node count to maintain ~52% win rate for network."""
    def __init__(self, initial_nodes=10000, target_winrate=0.52):
        self.nodes = initial_nodes
        self.target = target_winrate
        self.integral = 0
        self.prev_error = 0
        self.ema_winrate = 0.5  # Exponential moving average
        self.alpha = 0.03       # EMA smoothing factor
        self.kp, self.ki, self.kd = 0.5, 0.1, 0.2
        self.min_nodes, self.max_nodes = 500, 2_000_000
        self.update_counter = 0
        self.update_interval = 25  # Games between adjustments
        
    def record_result(self, network_won: bool):
        self.ema_winrate = self.alpha * float(network_won) + (1 - self.alpha) * self.ema_winrate
        self.update_counter += 1
        if self.update_counter >= self.update_interval:
            self._adjust()
            self.update_counter = 0
    
    def _adjust(self):
        error = self.ema_winrate - self.target
        if abs(error) < 0.05:  # Dead zone: don't adjust if within ±5%
            return
        self.integral = max(-1, min(1, self.integral + error))  # Anti-windup clamp
        derivative = error - self.prev_error
        adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative
        # Clamp adjustment to ≤10% change
        adjustment = max(-0.10, min(0.10, adjustment))
        # Network winning too much → increase SF nodes (make harder)
        self.nodes = int(self.nodes * (1 + adjustment))
        self.nodes = max(self.min_nodes, min(self.max_nodes, self.nodes))
        self.prev_error = error
```

**MCTS — Use Gumbel MCTS (Danihelka et al., ICLR 2022):**

Gumbel MCTS guarantees policy improvement at ANY simulation budget. Critical early in training when policy is poor — vanilla PUCT MCTS fails below ~16 simulations, Gumbel works with as few as 2.

Algorithm:
1. Sample Gumbel noise: `g(a) ~ Gumbel(0,1)` for each action a
2. Score each action: `score(a) = g(a) + log π(a)` where π is the network's policy prior
3. Select top-k actions by score (k = min(num_actions, budget-dependent))
4. Apply Sequential Halving among k candidates:
   - Split remaining simulations across candidates
   - Evaluate each candidate
   - Eliminate bottom half
   - Repeat until 1 remains
5. Final action selection: argmax of `g(a) + logits(a) + σ(q̂(a))` among most-visited actions
   - σ transforms Q-values: `σ(q) = (c_visit + max_visits) * c_scale * q` where `c_visit=50, c_scale=1.0`

At non-root nodes, use simple deterministic selection: pick action minimizing a scoring function based on improved policy and visit counts (no Gumbel noise needed at non-root).

Training target from Gumbel MCTS: the **completed Q-values** and **improved policy** (not raw visit counts). The improved policy `π̂(a) ∝ π(a) * exp(q̂(a))` is used as the policy target with KL divergence loss.

For Phase 1, start with standard PUCT MCTS as a debugging stepping stone (validate tree structure, batching, backprop work correctly), then switch to Gumbel MCTS before any real training begins. PUCT reference for debugging only:
```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```
Where `c_puct = 2.5` (LC0 default), P(s,a) is network policy prior, N is visit count.

**Batched inference across concurrent games:**

The key throughput optimization: collect leaf nodes from MCTS across ALL concurrent games, batch them into a single GPU forward pass.

```python
class BatchedMCTS:
    """Runs MCTS across N concurrent games, batching neural network calls."""
    def __init__(self, model, num_games=32, simulations=100):
        self.model = model
        self.games = [MCTSTree() for _ in range(num_games)]
        self.simulations = simulations
    
    def run_batch(self):
        """One iteration: collect leaves from all games, batch evaluate, backpropagate."""
        leaves = []
        leaf_game_indices = []
        for i, game in enumerate(self.games):
            if game.needs_evaluation():
                leaf = game.select_leaf()  # Traverse tree to unexpanded node
                leaves.append(encode_position(leaf.board))
                leaf_game_indices.append(i)
        
        if leaves:
            batch = torch.stack(leaves).to(device)
            with torch.no_grad():
                outputs = self.model(batch)
            
            for j, game_idx in enumerate(leaf_game_indices):
                policy = outputs['policy_own'][j]
                value = outputs['wdl'][j]
                self.games[game_idx].backpropagate(policy, value)
```

**KataGo Playout Cap Randomization (constant ratio — NOT progressive):**

KataGo does NOT start with low simulations and increase. It uses a **fixed random split on every move for the entire run**:

- 25% of moves get full search (200-400 Gumbel sims)
- 75% of moves get fast search (30-50 Gumbel sims)

This ratio stays constant throughout training. KataGo's ablation showed this fixed split outperformed every fixed playout count and every progressive schedule they tried.

**What KataGo trained on:** KataGo used PUCT, so fast search policy targets were near-useless. KataGo trained policy ONLY on full-search moves. Fast-search moves contributed ONLY value targets (game outcome). Dropping 75% of moves for policy wasn't just about quality — consecutive positions in a game are heavily correlated, so this also served as natural decorrelation.

**What WE train on — two-stage pipeline: SKIP then WEIGHT**

Gumbel MCTS guarantees valid policy improvement at ANY simulation budget, so unlike KataGo our fast searches produce meaningful policy targets. But we still need decorrelation (consecutive game positions are ~identical) and curriculum learning (focus compute on what the network gets wrong). We use a two-stage pipeline inspired by LC0's "diff focus" and KataGo's surprise weighting:

**Stage 1 — Probabilistic skip (decorrelation + curriculum):**
Before a position enters the training buffer, compute its combined difficulty from BOTH value and policy surprise (LC0's key insight):

```python
# Compute combined difficulty score
q_surprise = abs(search_value - raw_network_value)  # value head was wrong
pol_surprise = KL(raw_policy || gumbel_improved_policy)  # policy head was wrong

difficulty = (q_surprise * config["diff_focus_q_weight"]) + \
             (pol_surprise * config["diff_focus_pol_scale"])

# Probabilistic keep — easy positions get skipped
keep_prob = max(config["diff_focus_min"], min(1.0, difficulty * config["diff_focus_slope"]))

if random.random() > keep_prob:
    skip this position for policy training (still use for value training)
```

Defaults: `diff_focus_q_weight=6.0`, `diff_focus_pol_scale=3.5`, `diff_focus_slope=3.0`, `diff_focus_min=0.025` (following LC0's T60 run). The `min=0.025` means even trivially easy positions have a 2.5% chance of surviving — never completely blind to easy positions.

**Why this is curriculum learning:** Early in training, the network is garbage, so almost everything has high difficulty → almost everything survives → lots of data for initial learning. As the network improves, easy positions get correctly predicted → low difficulty → skipped → training naturally focuses on what the network still gets wrong. No manual scheduling needed.

**Why both signals matter:** A position where value changed dramatically but policy didn't (network picked the right move but misjudged the resulting position) is critical for value head training. A position where policy changed but value didn't (network knew the eval but picked a bad move) is critical for policy head training. Either signal alone misses half the story.

**Stage 2 — Differential weighting (for surviving positions):**
Positions that survive the skip gate get weighted by search quality × remaining surprise:

```python
# Base weight by search tier
base_weight = 1.0 if full_search else config["fast_search_policy_weight"]  # 0.3-0.5

# Surprise multiplier (among survivors): 50% uniform, 50% proportional to difficulty
surprise_multiplier = 0.5 + 0.5 * (difficulty / mean_difficulty_among_survivors_in_game)

final_policy_weight = base_weight * surprise_multiplier
# Value weight = 1.0 for ALL positions (game outcome equally valid regardless)
```

**Combined effect:** A boring middlegame position with no surprises has ~2.5% chance of entering training at all. If it does survive, it gets low weight. A sharp tactical position where the network was very wrong survives with ~100% probability AND gets high weight. Full-search moves always outweigh fast-search at the same surprise level. This is decorrelation + curriculum + quality weighting in one system.

Progressive simulation schedule is optional. Ramping sims can speed up early training, but it changes the quality distribution of policy targets. If you use a ramp, keep the full/fast split ratio roughly constant (ramp both together) and treat it as a tunable schedule.

The Gumbel paper's "works with 2 simulations" result means the algorithm guarantees policy improvement at any budget — it does NOT mean "train with 2 sims then switch to more." Use a constant playout cap ratio from day one.

### Module 5: Training Loop

```python
# Optimizer: NAdamW (proven in LC0) or SOAP (Ceres found 30% faster convergence)
# Make this a config parameter for Ray Tune ablation
if config.get("optimizer", "nadamw") == "nadamw":
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=config.get("lr", 0.001),
        betas=(0.9, 0.98),
        eps=1e-7,
        weight_decay=config.get("weight_decay", 0.0001),
    )
else:
    # SOAP: pip install soap-optimizer
    # from soap import SOAP
    # optimizer = SOAP(model.parameters(), lr=config.get("lr", 0.001), ...)
    # Fallback to NAdamW until SOAP is validated
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=config.get("lr", 0.001),
        betas=(0.9, 0.98),
        eps=1e-7,
        weight_decay=config.get("weight_decay", 0.0001),
    )

# torch.compile for ~75% training speed boost (CeresTrain measured 8k→14k pos/sec)
model = torch.compile(model, mode="reduce-overhead")

# LR schedule: cosine with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5000, T_mult=2, eta_min=1e-5
)
# Warmup: linear ramp from 0 to peak LR over first 1500 steps
warmup_steps = 1500

# Mixed precision
scaler = torch.amp.GradScaler('cuda')  # BF16 on Ampere+ doesn't strictly need this
                                         # but good practice

# Batch size: 2048 effective via gradient accumulation
micro_batch = 512  # Fits in VRAM
accum_steps = 4    # 512 * 4 = 2048 effective batch

# Training loop
for step in range(num_steps):
    optimizer.zero_grad()
    for _ in range(accum_steps):
        batch = replay_buffer.sample(micro_batch)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch['input_planes'])
            loss = compute_loss(outputs, batch['targets'], batch['is_network_turn'])
            loss = loss / accum_steps
        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    scaler.step(optimizer)
    scaler.update()
    
    if step < warmup_steps:
        lr = 0.001 * step / warmup_steps
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    else:
        scheduler.step()
```

**Replay buffer:**
- Sliding window of 200K-500K positions (start small, grow as training progresses)
- Shuffle buffer: sample randomly from window
- **Two-stage difficulty pipeline (see Playout Cap Randomization section for full details):**
  Stage 1: Probabilistic skip based on combined value + policy surprise (LC0 "diff focus").
  Easy positions dropped → decorrelation of correlated consecutive game positions + automatic curriculum.
  Stage 2: Surviving positions weighted by `base_weight * surprise_multiplier`.
  Full-search moves base=1.0, fast-search moves base=0.3-0.5, surprise multiplier boosts high-KL positions.

**Sampling strategy:**
All recorded samples are network-turn samples.
- Policy targets exist for all samples (but `has_policy` can downweight fast-search targets).
- Stockfish move/eval targets exist for the subset of samples where an SF reply exists.
- Future policy targets exist except near game end.
- Volatility targets exist except near game end.

### Module 6: Game Termination

**DO NOT implement hard resignation.** LC0's experience: "All attempts to filter out 'bad' positions were detrimental, as value head seems to need to see enough bad positions in training."

**KataGo-style soft resignation:**
When losing side's MCTS winrate p < 5% for 5 consecutive moves:
- Reduce playout count proportionally: `playouts = int(min_playouts + (full_playouts - min_playouts) * p / 0.05)`
- Record positions with reduced probability: `sample_prob = 0.1 + 0.9 * (p / 0.05)`
- Game still plays to completion — preserves endgame/fortress training signal

**Tablebase rescoring (post-hoc):**
When a position reaches ≤7 pieces (after capture/pawn push, no castling), look up Syzygy WDL.
Relabel game result for ALL preceding positions to tablebase-proven result.
Free perfect ground truth. Use `python-chess`'s Syzygy tablebase probing.

**NEVER adjudicate using Stockfish's evaluation.** If SF says "winning" but it's actually a fortress, that's exactly the scenario you're training to exploit. Terminating with SF's judgment corrupts training signal.

### Module 7: Monitoring and Evaluation

Track these metrics in TensorBoard:

- Policy accuracy (top-1, top-3) on held-out positions, separately for network turns and SF turns
- Value accuracy (WDL cross-entropy) on held-out positions
- SF move prediction accuracy (is the SF head learning to predict SF's moves?)
- SF eval prediction accuracy (is it learning SF's evaluations?)
- Volatility head calibration: do high-volatility predictions correlate with actual large eval swings?
- Win rate vs SF at current difficulty setting
- Win rate vs SF at fixed difficulty (e.g., nodes=50000) for absolute progress tracking
- Puzzle test accuracy (e.g., STS, WAC) for general chess ability regression detection
- Divergence between primary value head and SF eval head — this is the fortress/horizon-effect detector

---

## TRAINING DATA STRUCTURE

```python
@dataclass
class TrainingPosition:
    input_planes: np.ndarray        # (C, 8, 8) float16
    legal_move_mask: np.ndarray     # (num_moves,) bool — depends on encoding
    is_network_turn: bool           # always True in current scheme (kept for compatibility)
    game_id: int
    ply: int

    # Network policy targets
    mcts_visits: np.ndarray                  # (4672,) MCTS visit distribution
    mcts_visits_temp2: Optional[np.ndarray]  # (4672,) temp-2 target

    # Stockfish reply targets (attached to the network-turn sample)
    sf_policy_target: Optional[np.ndarray]   # (4672,) MultiPV-derived soft reply policy (fallback to one-hot)
    sf_move_index: Optional[int]             # Index of SF's chosen reply move
    sf_wdl: Optional[np.ndarray]             # (3,) SF reply-search WDL

    # Game targets
    wdl_target: int                          # {0,1,2} win/draw/loss label (network POV)
    categorical_target: np.ndarray           # (32,) HL-Gauss smoothed categorical bins
    moves_left: float                        # Remaining game length, normalized

    # Computed post-game
    future_visits_t2: Optional[np.ndarray]   # Next network move after SF reply (t+2 plies)
    volatility_target: Optional[np.ndarray]  # (3,) |WDL_{t+6} - WDL_t| per component
    sf_volatility_target: Optional[np.ndarray] # (3,) |SF_WDL_{t+6} - SF_WDL_t| per component
```

---

## OPPONENT MIXING (optional)

Opponent mixing is optional and can be enabled later if we observe overspecialization.

- Default: 100% games against Stockfish (node-limited, PID-controlled)
- Optional: mix in earlier checkpoints / other engines
- Always monitor general chess ability via a puzzle test suite

---

## CURRICULUM PHASES (overlapping, not sequential)

**Phase 1 (from start):** All positions from games. Tablebase rescoring for endgames. Value learning on everything, policy learning on network's turns.

**Phase 2 (after ~10K games):** Start mining positions where SF eval diverges between shallow and deep search: `|sf_eval(depth=20) - sf_eval(depth=40)| > threshold`. These are where SF's shallow search is most wrong. Pre-compute offline from large game databases. Weight these higher in sampling.

**Phase 3 (after ~50K games):** Inject known fortress positions from databases. Inject positions from historical games where neural network engines beat Stockfish. Weight fortress-candidate positions (material imbalance + blocked pawn structure) to play to completion.

---

## WHAT THE VOLATILITY HEAD IS AND WHY IT'S NOVEL

Predicting WDL at t+6 directly (like BT4's short-term value head) loses uncertainty information — on average, expected WDL_t+6 ≈ WDL_now, so the signal vanishes across positions. What matters is the SPREAD, not the center.

The volatility head predicts `E[|ΔW|], E[|ΔD|], E[|ΔL|]` — the expected absolute change per WDL component. This is directly actionable at inference:

- High volatility → allocate more MCTS nodes to this position
- High E[|ΔW|] specifically → network believes it's about to gain (hidden tactic); play confidently
- High E[|ΔD|] with low E[|ΔW|], E[|ΔL|] → draw is unstable, classic fortress territory
- High E[|ΔL|] → about to walk into trouble, consider safer alternatives

Per-component decomposition matters because single-scalar volatility loses the TYPE of uncertainty.

Nothing published does this. Closest: BT4's value error head (predicts error magnitude, not temporal volatility), KataGo's ownership head (spatial uncertainty, not temporal).

---

## KEY REFERENCES

| Paper | What to take from it |
|-------|---------------------|
| Monroe et al. "Mastering Chess with a Transformer Model" (arXiv:2409.12272) | Chessformer/BT4 architecture, smolgen, all head definitions, attention-based policy |
| Wu "Accelerating Self-Play Learning in Go" (2020) — KataGo | Auxiliary targets, playout cap randomization, policy surprise weighting |
| Danihelka et al. "Policy Improvement by Planning with Gumbel" (ICLR 2022) | Gumbel MCTS, works with 2 simulations, sequential halving |
| Imani et al. "Improving Regression with Distributional Losses" (ICML 2018) + Farebrother et al. "Stop Regressing" (arXiv:2403.03950) | HL-Gauss for categorical value head |
| Wang et al. "Adversarial Policies Beat Superhuman Go AIs" (ICML 2023) | Training against frozen opponent, only adversary's turns |
| Jenner et al. "Evidence of Learned Look-Ahead" (arXiv:2406.00877) | LC0 internally represents future moves — justifies future move head |
| Czech et al. "AlphaVile" (arXiv:2304.14918) | WDL + moves-left = +180 Elo over AZ baseline |
| LC0 "diff focus" (lczero-training/tf/chunkparser.py, T60 run) | Probabilistic skip of easy positions via combined value + policy KLD, automatic curriculum |
| Butner "ChessCoach" (2021) | NEGATIVE result: opponent reply head harmful in chess. Our setup differs: SF policy imitation uses SF's actual MultiPV output (soft target, one-hot fallback), not a hypothetical reply distribution. |

---

## IMPLEMENTATION ORDER

1. `encode_position()` — feature extraction, test against known positions
2. `ChessNet` — forward pass with all 10 heads, verify shapes with random input. Test ONNX export.
3. Ray Tune scaffolding — wrap training in `train_trial()`, define search space, verify ASHA works with dummy metrics
4. Simple game loop — network's raw policy (no search) vs SF at 1000 nodes. Validates UCI, data recording.
5. PUCT MCTS with batched inference, 50 simulations
6. Training loop — replay buffer, loss, optimizer. Small scale: 1 GPU, 1000 games. Run through Ray Tune.
7. **Ablation studies** — run the 8 key ablations on reduced scale (5-layer, 20 epochs). Use results to set defaults.
8. PID difficulty controller, playout cap randomization
9. Upgrade to Gumbel MCTS
10. Position mining, fortress injection, curriculum
11. ONNX export to Ceres, validate inference matches
12. Distributed training infrastructure (already implemented)
The project includes a server + workers + learner pipeline for distributed selfplay. Further work is optional hardening/scaling.

Start with step 1. Build and test each module independently before integrating.
