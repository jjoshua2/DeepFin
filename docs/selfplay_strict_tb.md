# Strict finite selfplay tablebases

`selfplay.syzygy_outcome_mode` defaults to `theoretical_wdl`, preserving the
legacy collector and its historical whole-game rescoring. The opt-in
`rule50_match_v1` contract is supported by **finite `play_batch` calls only**.
Ray Tune, continuous Worker sessions, suspension and persisted-game resume are refused;
the Worker checks before asset/model synchronization. This feature does not
enable a running experiment or qualify a teacher, dataset or learned model.

Strict mode requires the same full WDL+DTZ path for selfplay and Stockfish,
`syzygy_in_search: true`, `syzygy_adjudicate: true` and adjudication fraction
one. The native Gumbel TB consumer is required; the volatility Python path and
other search routes refuse instead of silently ignoring TB. The caller supplies
an actual Stockfish engine/pool initialized with `syzygy_50_move_rule=True` and
`syzygy_probe_limit=6`; capability checks and a ready barrier must have completed.
Pool replacements retain these explicit options. Policy rescoring is optional.

Each call owns a safe no-eviction Python tablebase handle. Both WDL and DTZ
must contain every canonical 3–6-man material. Missing eligible probes raise.
The handle closes after synchronous search/finalization returns, on success or
exception; globally cached legacy handles are never closed by this session.
Live changes to the strict TB settings require a new finite call. The outcome
mode is restart-required in YAML reload and in the Worker manifest fingerprint.

Rules decisions replay the original opening stack plus played actions and
verify the exact native final FEN, including clocks. This separate rules board
does not alter `SelfplayState.replay_board` or the model's encoding history.
Natural endings and automatic claim policy (`claim_draw=True`) take precedence.
The shared conservative rule50 primitive treats cursed/blessed WDL ±1 as draw;
decisive ±2 is qualified at a reset clock. Positive-clock decisive results stay
unresolved and play on. At a fixed cap, unresolved attempts emit **no replay
rows**, no SF/default-draw label, and increment `BatchStats.censored_games`.
`BatchStats.games` counts finalized attempts, including these censored attempts;
it is not a count of qualified training games.

An `on_game_provenance` callback is mandatory, including for zero-row attempts.
The frozen witness distinguishes `NATURAL`, `TB_ADJUDICATED` and `UNRESOLVED`;
it records root/history, terminal position/clock/turn, played-move offset,
raw WDL/DTZ where probed, conservative status/result and max pieces. Asset
identity is explicitly a digest of resolved file paths, sizes and modification
times, **not a content hash**. An experiment must separately bind its actual
asset inventory and content authority. `native_result` retains the native
board string; strict `natural_result` uses the full-history claim policy.

Strict finalization never replaces an actual result with the last theoretical
TB result found earlier in the game. If policy rescoring is enabled, it uses
the same owned handle, original legal histories and exact record ply/offset/POV
joins. It ranks only fully resolved legal children with the existing DTZ
ranking helper; ambiguous roots/children abstain, missing covered children
raise. Applied and abstained record plies are retained separately. It cannot
change hard game outcomes. No new loss, categorical-rebuild, optimizer or
architecture behavior is introduced.

SF node queries snapshot full `root_fen` and immutable ordered UCI moves at
the actual query position. A pending label retains that P1 history for later
escalation even after its slot advances. The public transport validates the
final FEN/clocks before I/O; strict queries never catch an unsupported-history
error and retry as bare FEN. Legacy callers omit these optional fields.

`tests/test_selfplay_strict_tb.py` contains synthetic consumer regressions for
configuration, opening-boundary claims, strict outcomes, caps, ownership,
policy-only finalization, per-record POV and SF snapshot/replacement custody.
These tests do not substitute for separately admitted actual CPU tablebase/SF
qualification and an excluded producer qualification before research collection.
