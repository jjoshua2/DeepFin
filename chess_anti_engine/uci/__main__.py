"""CLI entry point: ``python3 -m chess_anti_engine.uci --checkpoint PATH``.

Loads the checkpoint, constructs a DirectGPUEvaluator (CUDA if available,
CPU otherwise), and runs the UCI stdin loop until ``quit``. Model load +
evaluator construction run on a background thread so the ``uci`` handshake
can reply instantly; ``isready`` and later commands block until the
engine is actually ready.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
from collections.abc import Mapping
import logging
import os
import sys
import threading

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_cache import EncodedEvalCache
from chess_anti_engine.inference_dispatcher import (
    BatchCoalescingDispatcher,
    CUDAOwnerDispatcher,
    MultiGPUDispatcher,
    ThreadSafeGPUDispatcher,
    bootstrap_cudagraph_tls,
)
from chess_anti_engine.mcts.gumbel import (
    PLAY_PUCT_DEFAULTS,
    PLAY_SEARCH_DEFAULTS,
    PLAY_SEARCH_VLOSS_WEIGHT,
    GumbelConfig,
)
from chess_anti_engine.mcts.search_options import SEARCH_OPTIONS

from .engine import Engine, EngineOptions, _emit_info_string, _println, emit_handshake
from .model_loader import load_model_from_checkpoint
from .protocol import CmdQuit, CmdUci, parse_command
from .search import SearchWorker
import contextlib

# torch.compile modes that enable Inductor CUDA graphs (cudagraph_trees).
_CUDAGRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})
# Fallback when a cudagraph multi-GPU factory fails strict warmup: inductor
# fusion without graph capture. Prefer ``default`` over
# ``max-autotune-no-cudagraphs`` so a fallback path stays inside the
# tournament prewarm budget (max-autotune cold can exceed 10 min on
# mid-range GPUs). Disk cache still applies.
_MULTI_GPU_NO_CUDAGRAPH_MODE = "default"
# Serialize compile + first-forward capture across multi-GPU factory
# workers. Concurrent torch.compile/cudagraph capture on two threads
# races dynamo + the CUDA caching allocator even with per-thread TLS
# bootstrapped; serial capture is a few minutes cold and seconds warm.
# Concurrent REPLAY after capture is the multi-GPU NPS win.
_MULTI_GPU_COMPILE_LOCK = threading.Lock()


def resolve_multi_gpu_compile_mode(
    compile_mode: str | None,
    n_devices: int,
    *,
    announce: bool = True,
) -> str | None:
    """Return the compile mode used for multi-GPU worker factories.

    Default: keep the caller's mode (including CUDA-graph modes). Multi-GPU
    factories bootstrap cudagraph TLS and serialize first capture so
    reduce-overhead / max-autotune work on pool threads.

    Override with ``CAE_UCI_MULTI_GPU_COMPILE_MODE`` (empty → eager). Use
    ``max-autotune-no-cudagraphs`` or ``default`` to force a no-graph path.
    """
    if compile_mode is None or n_devices <= 1:
        return compile_mode
    override = os.environ.get("CAE_UCI_MULTI_GPU_COMPILE_MODE")
    if override is not None:
        chosen = override.strip() or None
        if announce:
            _println(
                "info string multi-GPU compile mode override via "
                f"CAE_UCI_MULTI_GPU_COMPILE_MODE={override!r} → {chosen!r}"
            )
        return chosen
    if announce and compile_mode in _CUDAGRAPH_COMPILE_MODES:
        _println(
            f"info string multi-GPU: using compile mode {compile_mode!r} with "
            "per-worker cudagraph TLS bootstrap + serialized capture "
            f"(fallback {_MULTI_GPU_NO_CUDAGRAPH_MODE!r} on warmup failure; "
            "override with CAE_UCI_MULTI_GPU_COMPILE_MODE)"
        )
    return compile_mode


def _warmup_evaluator(
    evaluator,
    *,
    n_walkers: int = 1,
    walker_gather: int = 1,
    n_devices: int = 1,
    input_history_encoding: str = "legacy",
    input_extra_features: str | None = None,
    compute_relations: bool = False,
) -> None:
    """Trigger torch.compile + CUDA graph capture for the shapes the UCI
    search will actually hit, so the first `go` doesn't pay compile
    latency (~3-5s per new shape).

    Walker path (n_walkers > 1) hits batches 1..G per walker (where G is
    walker_gather) plus up to N_walkers×G from coalesced wavefronts.
    Gumbel path (n_walkers == 1) hits batch=1 for root eval and batch=128
    for the single-game bucket (gumbel_c._BUCKETS). We warm the endpoints
    for the selected path — intermediate sizes fall through the same
    compiled graph once the endpoints are captured.

    ``n_devices``: each shape is issued this many times so a routing chain
    containing ``MultiGPUDispatcher`` warms EVERY device, not just the
    first few. Sequential warmup calls always tie on in-flight count (0),
    and the dispatcher's round-robin tiebreaker cycles ties, so N calls
    visit all N devices deterministically. A cold device would otherwise
    pay its compile/capture stall on a mid-game root eval, on the clock
    and outside the chunk clock-guard.

    Warmup is skipped silently on failure — the real ``go`` will see the
    same error and surface it there.
    """
    cb = CBoard.from_board(chess.Board())
    encoded = encode_cboard(
        cb,
        input_history_encoding=input_history_encoding,
        input_extra_features=input_extra_features,
    )
    if n_walkers > 1:
  # batch=1 covers the pre-gather single-descent phase (e.g. when
  # budget runs out mid-gather). walker_gather is the per-walker
  # submit; n_walkers*walker_gather is the fully-coalesced wavefront.
        batches = sorted({1, int(walker_gather), int(n_walkers) * int(walker_gather)})
    else:
  # Gumbel path: root eval at 1, bucket flushes at 128.
        batches = [1, 128]
    rel_row = cb.compute_relations() if compute_relations else None
    for batch in batches:
        xs = np.broadcast_to(encoded, (batch, *encoded.shape)).astype(np.float32, copy=True)
        try:
            for _ in range(max(1, int(n_devices))):
                if rel_row is None:
                    evaluator.evaluate_encoded(xs)
                else:
  # Warm WITH relations so compile/cudagraph captures the graph the
  # real search will replay (relations change the traced forward).
                    rels = np.broadcast_to(rel_row, (batch, *rel_row.shape)).copy()
                    evaluator.evaluate_encoded(xs, relations=rels)
            # Compact UCI search uses a distinct compiled forward/gather graph.
            # Warm it before readyok as well; otherwise the first timed move
            # pays max-autotune compilation on the clock even though the dense
            # endpoint above was already warm.
            if (
                rel_row is None
                and bool(getattr(evaluator, "supports_legal_bf16", False))
                and hasattr(evaluator, "evaluate_legal_bf16")
            ):
                legal_one = cb.legal_move_indices().astype(np.int32, copy=False)
                legal_counts = np.full((batch,), legal_one.size, dtype=np.int32)
                legal_flat = np.tile(legal_one, batch)
                for _ in range(max(1, int(n_devices))):
                    evaluator.evaluate_legal_bf16(xs, legal_flat, legal_counts)
        except Exception:
            break


def _pucv_warmup_batch_sizes(gather: int) -> list[int]:
    """Batch sizes MultiGpuPucvPool / RPG submit in the steady state.

    Full-gather is the common case; batch=1 covers root / short-budget
    edges. CUDA graphs are process-local (not disk-cached), so every
    isready re-captures — keep this set minimal for the tournament
    ~30s warm-launch budget. Mid-size remainders recompile once mid-
    search under the factory lock's warm shapes only if they appear;
    production gather-aligned budgets rarely need them.
    """
    g = max(1, int(gather))
    return sorted({1, g})


def _warmup_pucv_evaluator(
    evaluator,
    *,
    gather: int,
    input_history_encoding: str = "legacy",
    input_extra_features: str | None = None,
    compute_relations: bool = False,
    strict: bool = False,
) -> None:
    """Warm the exact inplace-async shapes used by MultiGpuPucvPool.

    ``strict=True`` re-raises the first failure so multi-GPU factory install
    cannot silently return a cold evaluator that then crashes on the first
    real ``go`` (tournament clock). Non-strict keeps the historical best-
    effort path used by opportunistic warmups.
    """
    cb = CBoard.from_board(chess.Board())
    encoded = encode_cboard(
        cb,
        input_history_encoding=input_history_encoding,
        input_extra_features=input_extra_features,
    )
    rel_row = cb.compute_relations() if compute_relations else None
    batches = _pucv_warmup_batch_sizes(gather)
    n_slots = min(2, int(getattr(evaluator, "n_slots", 1)))
    for batch in batches:
        for slot in range(n_slots):
            try:
                buf = evaluator.get_input_buffer(batch, slot=slot)
                buf[:] = np.broadcast_to(encoded, (batch, *encoded.shape))
                if rel_row is None:
                    _pol, _wdl, event = evaluator.evaluate_inplace_async(batch, slot=slot)
                else:
                    rels = np.broadcast_to(rel_row, (batch, *rel_row.shape)).copy()
                    _pol, _wdl, event = evaluator.evaluate_inplace_async(
                        batch, slot=slot, relations=rels,
                    )
                if event is not None:
                    event.synchronize()
            except Exception:
                if strict:
                    raise
                return


def _load_models(checkpoint: str, devices: list[str]):
    """Load one model per device. Cached at startup and reused across
    evaluator rebuilds (e.g., when the UCI ``MaxBatch`` option changes)."""
    if len(devices) > 1:
  # Parallel load — each is hundreds of MB of weight copy + CUDA init.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            return list(pool.map(
                lambda d: load_model_from_checkpoint(checkpoint, device=d),
                devices,
            ))
    return [load_model_from_checkpoint(checkpoint, device=devices[0])]


def _make_evaluator_factory(
    models, devices, coalesce, n_walkers, walker_gather,
    *, compile_mode: str | None = None,
):
    """Return a ``build(max_batch) -> evaluator`` closure. The models are
    captured once at startup; each call constructs fresh evaluator
    wrappers at the new max_batch and warms them at the shapes the
    walker count + gumbel bucket will actually hit.

    ``compile_mode`` (e.g. ``"reduce-overhead"``) wraps the model with
    ``torch.compile`` before warmup. Compile + cudagraph capture must happen
    on the same thread that later replays the graph. In UCI the search itself
    runs on worker threads, so compiled evaluators use the submitter-thread
    dispatcher even at walkers=1; eager walkers=1 can keep the direct inplace
    evaluator path.
    """
    input_history_encoding = str(getattr(models[0], "input_history_encoding", "legacy"))
    input_extra_features = str(getattr(models[0], "input_extra_features", "v1"))
    use_relations = bool(getattr(models[0], "use_dynamic_relations", False))
    compact_bf16 = os.environ.get("CAE_UCI_COMPACT_BF16", "0") == "1"

  # Live-updatable walker count: the runtime `Threads` setoption crosses
  # the dispatcher tier (CUDAOwner at 1 walker vs BatchCoalescing at >1),
  # so rebuilds must see the NEW count, and later rebuilds (MaxBatch /
  # EvalCacheEntries) must keep it — hence remembered, not re-frozen.
    walker_state = {"n_walkers": int(n_walkers)}

    def build(
        max_batch: int,
        eval_cache_entries: int = 0,
        *,
        primary_only: bool = False,
        n_walkers: int | None = None,
    ):
        """Build a primary (root / walker) evaluator stack.

        ``primary_only=True`` forces a single-device stack on ``devices[0]``
        even when multiple devices were loaded. Used when the multi-GPU PUCV
        pool owns search concurrency — root eval only needs one device, and
        building MultiGPUDispatcher *plus* pool factories was ≈2× per-device
        evaluator/buffer/cudagraph memory after auto-enable made the pool
        the multi-device default.
        """
        if n_walkers is not None:
            walker_state["n_walkers"] = max(1, int(n_walkers))
        nw = walker_state["n_walkers"]
        active_models = models
        active_devices = devices
        if primary_only and len(devices) > 1:
            active_models = models[:1]
            active_devices = devices[:1]

        if compile_mode:
            import torch
            from typing import cast
            compiled_models = [
                cast("torch.nn.Module", torch.compile(m, mode=compile_mode))
                for m in active_models
            ]
        else:
            compiled_models = list(active_models)

        if len(active_devices) > 1:
            evaluators = [
                DirectGPUEvaluator(
                    m, device=d, max_batch=max_batch, n_slots=2,
                    input_bf16=compact_bf16, legal_bf16=compact_bf16,
                )
                for m, d in zip(compiled_models, active_devices)
            ]
            evaluator = MultiGPUDispatcher(evaluators)
        else:
            evaluator = DirectGPUEvaluator(
                compiled_models[0], device=active_devices[0],
                max_batch=max_batch, n_slots=2, input_bf16=compact_bf16, legal_bf16=compact_bf16,
            )
  # Always wrap in ThreadSafeGPUDispatcher so the UCI `Threads`
  # option can bump walker count at runtime without a race. Lock
  # is uncontended at 1 thread — overhead is ~10ns.
            evaluator = ThreadSafeGPUDispatcher(evaluator)
  # A persistent owner thread is mandatory for compiled evaluators: CUDA graph
  # capture and replay need to stay on the same thread. Single-walker search
  # forwards the direct evaluator's pinned in-place API; concurrent walkers use
  # the heavier coalescer so their requests can still become one GPU batch.
  # For eager evaluation coalescing is only a batching optimization, so
  # --no-coalesce can bypass it.
        if compile_mode is not None or (coalesce and nw > 1):
            if compile_mode is not None and nw == 1:
                evaluator = CUDAOwnerDispatcher(evaluator)
            else:
                evaluator = BatchCoalescingDispatcher(evaluator, max_batch=max_batch)
  # Warm BELOW the eval cache: EncodedEvalCache dedupes identical rows and
  # serves full-hit batches without touching the inner evaluator, so warming
  # through it would compile/capture only the first call's shape on one
  # device and leave every other shape/device cold. Warming the pre-cache
  # chain still runs on the persistent owner/submitter thread (the thread that
  # replays cudagraphs in real searches).
        warm_target = evaluator
        if eval_cache_entries > 0:
            evaluator = EncodedEvalCache(
                evaluator, max_entries=int(eval_cache_entries),
  # The model, not the evaluator stack: it is what declares the
  # encoding, and reading it live means a reload behind the
  # evaluator cannot serve entries built under the old one.
                encoding_source=models[0],
            )
        _warmup_evaluator(
            warm_target, n_walkers=nw, walker_gather=walker_gather,
            n_devices=len(active_devices),
            input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
            compute_relations=use_relations,
        )
        return evaluator
    return build


def _make_multi_gpu_pucv_factory_builder(
    models,
    devices,
    *,
    compile_mode: str | None = None,
):
    """Return ``build(max_batch) -> list[factory]`` for MultiGpuPucvPool.

    Each factory compiles, constructs, and warms its evaluator on the pool
    worker thread that will later run search. Compile+warmup is serialized
    across workers (see ``_MULTI_GPU_COMPILE_LOCK``) and uses a no-cudagraph
    mode when the requested mode would enable CUDA graphs — concurrent
    cudagraph capture/replay across GPU worker threads is not safe in
    PyTorch 2.6 (TLS tree_manager + mempool races).
    """
    input_history_encoding = str(getattr(models[0], "input_history_encoding", "legacy"))
    input_extra_features = str(getattr(models[0], "input_extra_features", "v1"))
    use_relations = bool(getattr(models[0], "use_dynamic_relations", False))
    compact_bf16 = os.environ.get("CAE_UCI_COMPACT_BF16", "0") == "1"
    # Resolve once at builder construction so every factory agrees and we
    # only emit the downgrade notice once (not once per device per rebuild).
    effective_compile_mode = resolve_multi_gpu_compile_mode(
        compile_mode, len(devices), announce=True,
    )

    def build(max_batch: int, gather: int):
        effective_gather = min(max(1, int(gather)), int(max_batch))
        factories = []
        for model, device in zip(models, devices):
            def make_one(m=model, d=device):
  # Bind this pool worker thread to its device BEFORE compile /
  # construction / warmup: threads inherit current-device cuda:0.
  # cpu devices (tests) skip set_device.
                import torch
                bootstrap_cudagraph_tls()
                # Multi-device compile reuses dynamo guards that key on
                # device index; raise the recompile ceiling so GPU 1..N-1
                # don't fall off the cache after GPU 0's shapes.
                try:
                    import torch._dynamo
                    torch._dynamo.config.cache_size_limit = max(
                        int(torch._dynamo.config.cache_size_limit),
                        64,
                    )
                except Exception:
                    pass
                dev = torch.device(d)
                if dev.type == "cuda":
  # bare "cuda" yields index None; `or 0` collapses None/0 to the default device.
                    torch.cuda.set_device(dev.index or 0)
                mode = effective_compile_mode
                with _MULTI_GPU_COMPILE_LOCK:
                    evaluator = _compile_and_warm_pucv_evaluator(
                        m, d,
                        max_batch=max_batch,
                        gather=effective_gather,
                        compile_mode=mode,
                        compact_bf16=compact_bf16,
                        input_history_encoding=input_history_encoding,
                        input_extra_features=input_extra_features,
                        use_relations=use_relations,
                    )
                return evaluator

            factories.append(make_one)
        return factories

    return build


def _compile_and_warm_pucv_evaluator(
    model,
    device: str,
    *,
    max_batch: int,
    gather: int,
    compile_mode: str | None,
    compact_bf16: bool,
    input_history_encoding: str,
    input_extra_features: str,
    use_relations: bool,
):
    """Compile + strict-warmup one multi-GPU worker evaluator.

    Caller holds ``_MULTI_GPU_COMPILE_LOCK`` and has already bound the
    CUDA device + bootstrapped cudagraph TLS. On cudagraph-mode warmup
    failure, retries once with ``_MULTI_GPU_NO_CUDAGRAPH_MODE`` so a
    single bad graph capture cannot take the whole tournament offline.
    """
    from typing import cast

    import torch

    modes: list[str | None]
    if compile_mode in _CUDAGRAPH_COMPILE_MODES:
        modes = [compile_mode, _MULTI_GPU_NO_CUDAGRAPH_MODE]
    else:
        modes = [compile_mode]

    last_exc: BaseException | None = None
    for i, mode in enumerate(modes):
        try:
            if mode:
                compiled = cast(
                    "torch.nn.Module",
                    torch.compile(model, mode=mode),
                )
            else:
                compiled = model
            evaluator = DirectGPUEvaluator(
                compiled, device=device, max_batch=max_batch, n_slots=2,
                input_bf16=compact_bf16, legal_bf16=compact_bf16,
            )
            _warmup_pucv_evaluator(
                evaluator,
                gather=gather,
                input_history_encoding=input_history_encoding,
                input_extra_features=input_extra_features,
                compute_relations=use_relations,
                strict=True,
            )
            if i > 0:
                _println(
                    f"info string multi-GPU: cudagraph warmup failed on "
                    f"{device}; fell back to compile mode {mode!r}"
                )
            return evaluator
        except BaseException as exc:
            last_exc = exc
            if i + 1 < len(modes):
                _println(
                    f"info string multi-GPU: compile/warmup failed on "
                    f"{device} with mode {mode!r}: {exc!r}; retrying "
                    f"{modes[i + 1]!r}"
                )
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _build_engine(
    *,
    evaluator,
    primary_device: str,
    chunk_sims: int,
    topk: int,
    c_scale: float = 0.025,  # UCI/high-sim tuned default (was 0.1); see --c-scale help
    c_visit: float = 50.0,
    c_puct: float = 2.5,
    cpuct_factor: float = 3.89,
    cpuct_base: float = 38739.0,
    fpu_reduction: float = 1.2,
  # Gumbel root/descent split (legacy defaults; see main()'s --c-visit-root etc.)
    c_visit_root: float = -1.0,
    q_visit_floor: float = -1.0,
    q_visit_exp: float = 1.0,
    q_global_scale: bool = False,
  # Root-ONLY LOG value-transform (UCI default; descent keeps linear q_visit_exp).
  # Linear root q_scale=c_scale*(c_visit_root+max_visit) EXPLODES with max_visit
  # (~25000 at 1M nodes) -> the root saturates sigma(q) and blindly trusts the
  # overconfident value head, so deeper search REVERSES (audit: +3.8cp at 32k).
  # Log q_scale=c_scale_root*log1p(c_visit_root+max_visit) stays ~100 from 256 to
  # millions of nodes -> sim-invariant scaling (the TCEC regret curve keeps
  # dropping). Verified no-op/improvement 256->16k, fixes the 32k reversal.
    c_scale_root: float = 7.0,
    q_visit_exp_root: float = -1.0,
  # Search-time policy-prior temperature (UCI `PolicyTemperature`). 1.0 is a
  # no-op; it had no plumbing here at all before, which is why no play-path
  # config could set it. != 1.0 costs ~1.9x end-to-end search — see the option
  # table in docs/operations.md.
    policy_temp: float = 1.0,
    halving_div: int = 2,
  # Root Gumbel-noise strength (UCI `GumbelScale`). 0 = the deterministic
  # search every prior build ran.
    root_noise_scale: float = 0.0,
    n_walkers: int,
    vloss_weight: int,
    walker_gather: int,
    pucv_vloss_mode: int,
    max_batch: int,
    vl_gather: int,
    eval_cache_entries: int,
    use_multi_gpu_pucv: bool,
    use_root_parallel_gumbel: bool = False,
    devices: tuple[str, ...] | None = None,
    input_history_encoding: str = "legacy",
    input_extra_features: str = "v1",
    policy_encoding: str = "lc0_1858",
    compute_relations: bool = False,
    rebuild_evaluator=None,
    rebuild_multi_gpu_pucv_factories=None,
    options: EngineOptions | None = None,
    gil_profile: bool = False,
) -> Engine:
    worker = SearchWorker(
        evaluator,
        device=primary_device,
        gumbel_cfg=GumbelConfig(
            simulations=chunk_sims,
            topk=topk,
            c_scale=c_scale,
            c_visit=c_visit,
            c_puct=c_puct,
            cpuct_factor=float(cpuct_factor),
            cpuct_base=float(cpuct_base),
            fpu_reduction=fpu_reduction,
            c_visit_root=c_visit_root,
            q_visit_floor=q_visit_floor,
            q_visit_exp=q_visit_exp,
            q_global_scale=q_global_scale,
            c_scale_root=c_scale_root,
            q_visit_exp_root=q_visit_exp_root,
            policy_temp=policy_temp,
            halving_div=halving_div,
            add_noise=False,
            input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
            policy_encoding=policy_encoding,
            compute_relations=compute_relations,
        ),
        chunk_sims=chunk_sims,
        n_walkers=n_walkers,
        vloss_weight=vloss_weight,
        walker_gather=walker_gather,
        pucv_vloss_mode=pucv_vloss_mode,
        eval_cache_entries=eval_cache_entries,
    )
    worker.set_root_noise_scale(root_noise_scale)
    # Same default Engine applies when options is None; resolved here too so
    # the RPG install below reads concrete knobs.
    opts = options if options is not None else EngineOptions()
    engine = Engine(
        worker,
        rebuild_evaluator=rebuild_evaluator,
        rebuild_multi_gpu_pucv_factories=rebuild_multi_gpu_pucv_factories,
        search_devices=devices,
        options=opts,
        gil_profile=gil_profile,
    )
  # The two multi-GPU modes are mutually exclusive; the gumbel branch reuses
  # the same per-device evaluator factories as the pucv pool (identical
  # evaluator interface + warmup; only the search orchestration differs).
    if use_root_parallel_gumbel and rebuild_multi_gpu_pucv_factories is not None:
        effective_gather = min(vl_gather, max_batch)
        factories = rebuild_multi_gpu_pucv_factories(max_batch, effective_gather)
        if len(factories) >= 2:
            worker.install_root_parallel_gumbel(
                factories, gather=effective_gather, as_factories=True,
                devices=list(devices) if devices else None,
                info_string_cb=_emit_info_string,
                open_vpa=int(opts.rpg_open_vpa),
                min_vpa=int(opts.rpg_min_vpa),
                min_keep=int(opts.rpg_min_keep),
                open_budget_frac=float(opts.rpg_open_budget_frac),
            )
    elif use_multi_gpu_pucv and rebuild_multi_gpu_pucv_factories is not None:
        effective_gather = min(vl_gather, max_batch)
        factories = rebuild_multi_gpu_pucv_factories(max_batch, effective_gather)
        if len(factories) >= 2:
            worker.install_multi_gpu_pucv(
                factories, gather=effective_gather, as_factories=True,
            )
  # Advertise what was BUILT, not what EngineOptions happens to default to.
  # These knobs arrive as CLI args and land on the worker; the handshake reads
  # EngineOptions. Copying worker -> options here makes the worker the single
  # direction of truth, so `--c-scale 0.3` can never be advertised as 0.025.
  #
  # AFTER the pool installs, not before: `realized_search_values` answers the
  # PUCT family off the installed descent object, so a copy taken before
  # `install_root_parallel_gumbel` reads the classic-Gumbel config and then
  # advertises it for a search that is about to run on the RPG pool's.
    for _opt in SEARCH_OPTIONS:
        opts.set_search_value(_opt.field, worker.realized_search_values()[_opt.field])
    return engine


def _pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_use_multi_gpu_pucv(
    flag: bool | None, n_devices: int, *, announce: bool = True,
) -> bool:
    """Decide whether the multi-GPU PUCV pool is active at startup.

    With one device the pool cannot install (needs >= 2 factories) — always
    False, keeping single-device behavior identical. With multiple devices
    the pool is the only path that runs the GPUs concurrently: the routing
    chain (BatchCoalescingDispatcher over MultiGPUDispatcher) has ONE
    submitter thread that calls the inner evaluator synchronously, so
    exactly one GPU is busy at a time — worse than a single device once
    cache behavior is counted. Hence:

    - ``flag`` unset (None): AUTO-ENABLE the pool and say so on stderr.
    - ``flag`` explicitly False: honor it, but warn LOUDLY — the serialized
      routing path is a debugging configuration, not a performance one.
    """
    if n_devices <= 1:
        return False
    if flag is None:
        if announce:
            print(
                f"info: {n_devices} devices and UseMultiGpuPUCV not set — "
                "auto-enabling the multi-GPU PUCV pool (without it the routing "
                "dispatcher serializes GPU work; pass --no-multi-gpu-pucv to "
                "force that debug path)",
                file=sys.stderr, flush=True,
            )
        return True
    if not flag:
        if announce:
            print(
                "WARNING: --no-multi-gpu-pucv with multiple devices — the "
                "routing dispatcher's single submitter runs the GPUs ONE CALL "
                "AT A TIME (serialized, at or below single-GPU throughput). "
                "This configuration is for debugging only.",
                file=sys.stderr, flush=True,
            )
        return False
    return True


def _resolve_multi_gpu_startup(
    search_parallel: str, pucv_flag: bool | None, n_devices: int,
) -> tuple[bool, bool, bool]:
    """Return ``(root_gumbel, pucv_fallback, active_pucv)``.

    The two pools are mutually exclusive, but retaining the PUCV intent while
    root Gumbel is active lets a later UCI ``SearchParallel=pucv`` switch
    restore concurrent multi-GPU search instead of the serialized dispatcher.
    """
    root_gumbel = search_parallel == "gumbel" and n_devices > 1
    pucv_fallback = _resolve_use_multi_gpu_pucv(
        pucv_flag, n_devices, announce=not root_gumbel,
    )
    return root_gumbel, pucv_fallback, pucv_fallback and not root_gumbel


def _build_parser() -> argparse.ArgumentParser:
    """The CLI. Split out of ``main`` so a test can read the REAL defaults.

    ``_seed_search_options_from_args`` claims every registry option is seeded
    from a named argparse ``dest``; that claim is only checkable if the parser
    can be built without running the engine.
    """
    p = argparse.ArgumentParser(prog="chess-anti-engine-uci")
  # --checkpoint is required, but we accept DEEPFIN_CKPT as the default so
  # the `deepfin` console-script entry point (from pyproject.toml) can be
  # launched as a bare executable by chess GUIs that don't pass CLI args.
    p.add_argument("--checkpoint", default=os.environ.get("DEEPFIN_CKPT"),
                   help="path to trainer.pt or checkpoint dir (falls back to $DEEPFIN_CKPT)")
    p.add_argument("--device", default="auto", help="cpu|cuda|cuda:N (default: auto)")
    p.add_argument("--devices", default=None,
                   help="comma-separated device list for multi-GPU (e.g. 'cuda:0,cuda:1'). Overrides --device.")
  # Defaults from the 2026-04-21 bench sweep (bench_uci_engine.py --sweep):
  # chunk=512/topk=32/mb=1024 gave ~7.3x startpos nps vs 32/16/32. Chunk cap
  # of 512 (not the full node budget) keeps `stop` latency under ~400ms on
  # single-game CUDA searches.
    p.add_argument("--chunk-sims", type=int, default=2048,
                   help="sims per start_gumbel_sims call (default: 2048). Higher = fuller GPU batches "
                        "(more NPS) at the cost of coarser stop/abort latency; 2048 favours throughput "
                        "for long time controls while leaving ample chunks per move for the abort.")
    p.add_argument("--topk", type=int, default=PLAY_SEARCH_DEFAULTS["topk"], help="Gumbel root candidates (default: 32)")
    p.add_argument("--c-scale", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale"],
                   help="Gumbel value-transform scale in sigma(q): lower leans on the prior "
                        "policy, higher trusts the search Q more. Default 0.025 — tuned "
                        "2026-06-16 (was 0.1; +270 Elo). q_scale=c_scale*(c_visit+max_visit) "
                        "explodes at high sims, so 0.1 over-trusted the overconfident value head.")
    p.add_argument("--c-visit", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit"], help="Gumbel c_visit constant (default: 50.0)")
    p.add_argument(
        "--policy-temp", type=float, default=float(PLAY_SEARCH_DEFAULTS["policy_temp"]),
        help="search-time temperature on the policy PRIOR (logits/T) before it "
             "seeds the tree. >1 softens, <1 sharpens, 1.0 (default) = no-op. "
             "Also UCI `PolicyTemperature`, range 0.5-5.0. COSTS ~1.9x search "
             "time when != 1.0 (disables the compact-legal bf16 leaf transport). "
             "NOT policy_target_temp, which is a training-time knob on the "
             "policy TARGET and never runs during play.",
    )
    p.add_argument(
        "--halving-div", type=int, default=2,
        help="Gumbel sequential-halving divisor: each round keeps "
             "ceil(n_cands/div). 2 = standard halving (default). Also UCI "
             "`HalvingDiv`.",
    )
    p.add_argument(
        "--gumbel-scale", type=float, default=0.0,
        help="root Gumbel-noise strength for match play (default 0.0 = the "
             "deterministic search every prior build ran). >0 re-enables root "
             "noise at that strength for opening diversity. Also UCI "
             "`GumbelScale`.",
    )
  # PUCT-descent knobs (walker pool / multi-GPU PUCV). They do NOT reach the
  # Gumbel search -- see mcts.gumbel.INERT_GUMBEL_KNOBS -- which is why their
  # defaults come from PLAY_PUCT_DEFAULTS and not PLAY_SEARCH_DEFAULTS.
    p.add_argument(
        "--c-puct", type=float, default=PLAY_PUCT_DEFAULTS["c_puct"],
        help="PUCT init (Lc0 CPuct, default: 1.75). With --cpuct-factor>0 this is "
             "the init term in c(N)=c_puct+factor*log((N+base)/base). PUCT walkers "
             "only -- inert in the Gumbel search.",
    )
    p.add_argument(
        "--cpuct-factor", type=float,
        default=float(PLAY_PUCT_DEFAULTS["cpuct_factor"]),
        help="Lc0 CPuctFactor (default: 3.89). 0 = fixed c_puct (no log ramp). "
             "PUCT walkers only -- inert in the Gumbel search.",
    )
    p.add_argument(
        "--cpuct-base", type=float,
        default=float(PLAY_PUCT_DEFAULTS["cpuct_base"]),
        help="Lc0 CPuctBase in log((N+base)/base) (default: 38739). PUCT walkers "
             "only -- inert in the Gumbel search.",
    )
    p.add_argument("--fpu-reduction", type=float, default=PLAY_PUCT_DEFAULTS["fpu_reduction"],
                   help="first-play-urgency reduction for unvisited children "
                        "(default: 0.33). PUCT walkers only -- inert in the Gumbel search.")
  # Gumbel root/descent split params (merged dormant in #68; C path only —
  # run_gumbel_root_many_c). All default to legacy: a GumbelConfig built with
  # the defaults below reproduces the pre-#68 single-floor/linear behavior.
    p.add_argument("--c-visit-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit_root"],
                   help="Gumbel root-halving c_visit override (value-transform floor at the "
                        "root sequential-halving site only; descent keeps --c-visit). A large "
                        "root floor (900, default) makes one c_scale fit every sim count's root "
                        "q_scale — the root/descent SPLIT that fixes search scaling (avoidable "
                        "regret +33%% vs legacy on the audit set; value scales coherently with "
                        "depth). <0 = use --c-visit at both sites (legacy, no split).")
    p.add_argument("--q-visit-floor", type=float, default=-1.0,
                   help="Gumbel decoupled (additive) value-transform floor: when >=0, "
                        "q_scale=q_visit_floor+c_scale*max_visit^exp instead of the legacy "
                        "c_scale*(c_visit+max_visit^exp) whose floor scales WITH c_scale. "
                        "Lets c_scale rise (for sublinear --q-visit-exp) without inflating the "
                        "early-round floor. <0 (default) = legacy coupled floor.")
    p.add_argument("--q-visit-exp", type=float, default=1.0,
                   help="Gumbel exponent on max_visit in q_scale=c_scale*(c_visit+max_visit^exp). "
                        "1.0 = standard linear Gumbel (default). <1 makes search-Q trust grow "
                        "sublinearly with depth so the optimal c_scale is less sim-count-dependent.")
    p.add_argument("--c-scale-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale_root"],
                   help="ROOT-ONLY c_scale (descent keeps --c-scale). Pairs with "
                        "--q-visit-exp-root<0 for a LOG root q_scale=c_scale_root*log1p(c_visit_root"
                        "+max_visit), which needs a large c_scale (~7) vs the tiny descent (~0.025). "
                        "<0 = use --c-scale at the root too (legacy linear).")
    p.add_argument("--q-visit-exp-root", type=float, default=PLAY_SEARCH_DEFAULTS["q_visit_exp_root"],
                   help="ROOT-ONLY value-transform exponent (descent keeps --q-visit-exp). DEFAULT "
                        "-1 = LOG slow-growth: linear root q_scale explodes ~25000 at 1M nodes and "
                        "saturates sigma(q) (deeper search reverses, +3.8cp@32k audit); log stays "
                        "~100 from 256 to millions of nodes (sim-invariant). >=90 = use --q-visit-exp.")
    p.add_argument("--q-global-scale", action="store_true",
                   help="Gumbel descent scales the value-transform by the ROOT max child-visit "
                        "(global/search-size) instead of the local node's max_visit, so q_scale is "
                        "uniform across the tree. Pairs with --q-visit-exp<1 for sim-invariance. "
                        "Default off = legacy local behavior.")
    p.add_argument("--max-batch", type=int, default=1024,
                   help="DirectGPUEvaluator max batch (default: 1024). Must be >= expected leaf count per wavefront.")
    p.add_argument("--eval-cache-entries", type=int, default=0,
                   help="LRU entries for encoded eval caches, including single-thread PUCV (default: 0/off)")
    p.add_argument("--abort-factor", type=float, default=1.0,
                   help="visit-margin early-abort aggressiveness (Lc0 smart-pruning factor): "
                        "1.0 = ~provable bound (default); < 1.0 aborts earlier to bank time")
    p.add_argument("--time-budget-scale", type=float, default=1.0,
                   help="per-move time-budget multiplier (default: 1.0). > 1.0 schedules more "
                        "time per move, relying on --abort-factor to bank it back on easy moves; "
                        "still capped at 50%% of remaining time so it cannot flag.")
    p.add_argument("--optimum-fraction", type=float, default=0.7,
                   help="soft-target fraction of the clock budget at which a settled move banks "
                        "the rest (default: 0.7). 0 turns the visit-margin abort fully off "
                        "(spend the whole deadline — the pre-time-management baseline); clamped to (0, 1].")
    p.add_argument("--moves-horizon", type=int, default=50,
                   help="upper cap on the pieces-driven moves estimate (default: 50; the material "
                        "estimate tops out near 30 so this rarely binds). Time allocation is driven by "
                        "pieces on the board, not move number; ignored when the GUI sends movestogo.")
    p.add_argument("--log-level", default="WARNING",
                   help="stderr log level (DEBUG|INFO|WARNING). DEBUG enables per-search gumbel profile with GPU-calls/avg-batch.")
    p.add_argument(
        "--gil-profile", action="store_true",
        help="emit per-search delayed-GIL-reacquisition telemetry as UCI info strings",
    )
  # --walkers > 1 switches from the Gumbel-chunked path to a PUCT walker
  # pool with virtual loss. Better dispatch-bound throughput, no
  # sequential-halving semantics. walkers=2 is ~10x the baseline on CUDA
  # (bench_uci_engine --walker-sweep, 2026-04-22) — we default to 2 so the
  # ship path gets the win. --walkers 1 opts into the classic Gumbel path.
    p.add_argument("--walkers", type=int, default=2,
                   help="PUCT walker threads (default: 2; 1 = classic Gumbel; >2 = noisy scaling)")
    p.add_argument("--vloss-weight", type=int, default=PLAY_SEARCH_VLOSS_WEIGHT,
                   help="virtual-loss weight, walker AND classic-Gumbel modes "
                        f"(default: {PLAY_SEARCH_VLOSS_WEIGHT}, lc0 default)")
  # Per-walker leaf gather: each walker does G descents → one NN batch.
  # Default 1 = current behavior. Increase (4-8) to amplify effective
  # submit batch size without more walker threads. Matches lc0's
  # MinibatchSize semantic (our --minibatch-size UCI option controls
  # the separate Gumbel path's C state machine, not this).
    p.add_argument("--walker-gather", type=int, default=1,
                   help="per-walker leaf gather (default: 1; lc0-style amplification at 4-8)")
    p.add_argument("--vl-gather", type=int, default=512,
                   help="leaf gather for UseVL and UseMultiGpuPUCV (default: 512)")
  # Tri-state: None = unset (auto-enable with >1 devices), True/False =
  # explicit. See _resolve_use_multi_gpu_pucv for why unset defaults ON.
    p.add_argument("--multi-gpu-pucv", dest="multi_gpu_pucv",
                   action="store_true", default=None,
                   help="with --devices, use shared-tree per-GPU PUCV workers instead of the "
                        "routing dispatcher (default: auto-enabled when --devices lists more "
                        "than one device)")
    p.add_argument("--no-multi-gpu-pucv", dest="multi_gpu_pucv",
                   action="store_false",
                   help="force the routing dispatcher even with multiple --devices "
                        "(DEBUG ONLY: its single submitter serializes GPU work)")
    p.add_argument("--search-parallel", choices=["pucv", "gumbel"], default="pucv",
                   help="multi-GPU search mode (requires --devices with >= 2 entries): "
                        "pucv = shared-tree PUCT+virtual-loss pool (default, throughput "
                        "baseline); gumbel = root-parallel Gumbel over evaluator groups "
                        "(quality-preserving — classic sequential-halving root decisions, "
                        "per-candidate subtrees parallelized across devices). With a single "
                        "device both values leave the classic search paths untouched.")
    p.add_argument("--pucv-pending-mode", choices=["legacy", "virtual-mean"],
                   default="legacy",
                   help="pending accounting for batched PUCV paths (default: legacy)")
  # Coalesce concurrent walker calls into batched submits. On by default
  # when walkers > 1 since batch=1 per walker wastes GPU.
    p.add_argument("--no-coalesce", dest="coalesce", action="store_false",
                   help="disable walker-call coalescing (debug / A-B bench only)")
    p.set_defaults(coalesce=True)
  # torch.compile + cudagraph wins are large for 384-dim/9-layer at small
  # batches (selfplay sees ~3-5x). Selfplay path uses ``reduce-overhead``;
  # we mirror that here. Cache lives at --compile-cache-dir so repeated
  # UCI launches reuse Inductor's FX graph cache + Triton kernels (cold
  # compile is ~30-90s; warm is ~1s).
    p.add_argument("--compile", dest="compile", action="store_true",
                   help="apply torch.compile to the model (default: on)")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="disable torch.compile (eager mode; useful for debugging)")
    p.set_defaults(compile=True)
  # 2026-04-28 sweep on 10-layer ckpt @ chunk=8192 mb=4096 walkers=1:
  # max-autotune = ~52k nps, reduce-overhead = ~44k nps. Cold compile is
  # ~2-3min slower under max-autotune but the FX graph + autotune cache
  # at --compile-cache-dir means second-run cost is ~seconds, not minutes.
    p.add_argument("--compile-mode", default="max-autotune",
                   help="torch.compile mode: max-autotune (default; ~52k nps benchmark) | "
                   "reduce-overhead (~44k nps; faster cold compile) | default (eager-ish)")
    p.add_argument("--compile-cache-dir",
                   default=os.environ.get(
                       "DEEPFIN_COMPILE_CACHE",
                       os.path.expanduser("~/.cache/deepfin/worker_cache"),
                   ),
                   help="shared TorchInductor/Triton cache root (env: DEEPFIN_COMPILE_CACHE; "
                   "default: ~/.cache/deepfin/worker_cache)")
    return p


# Registry field -> the argparse `dest` that decides its value AT STARTUP.
# `None` means "no CLI argument": the worker is built at the registry default
# and the handshake already advertises it.
#
# This map exists because `uci` is answered from `startup_options` on the READER
# thread while the worker is still being built on the build thread, so the copy
# `_build_engine` makes worker -> options lands strictly AFTER the first
# handshake a GUI ever sees. Without seeding, `--c-scale 0.077 --topk 9` was
# advertised as 0.025 / 32 on that first `uci` and only corrected on a second
# one most GUIs never send. `_seed_search_options_from_args` REFUSES to run when
# a registry entry is missing here, so a new option cannot quietly go back to
# advertising a default the engine is not running.
_SEARCH_OPTION_ARG: Mapping[str, str | None] = {
    "policy_temp": "policy_temp",
    "c_scale": "c_scale",
    "c_visit": "c_visit",
    "c_scale_root": "c_scale_root",
    "c_visit_root": "c_visit_root",
    "q_visit_exp": "q_visit_exp",
    "q_visit_exp_root": "q_visit_exp_root",
    "q_visit_floor": "q_visit_floor",
    "q_global_scale": "q_global_scale",
    "halving_div": "halving_div",
    "topk": "topk",
    "root_noise_scale": "gumbel_scale",
    "chunk_sims": "chunk_sims",
    "vloss_weight": "vloss_weight",
  # No CLI flag: `run_gumbel_root_many_c(target_batch=...)` starts at 0 (the
  # C-side default) and only a `setoption` moves it.
    "minibatch_size": None,
    "c_puct": "c_puct",
    "cpuct_factor": "cpuct_factor",
    "cpuct_base": "cpuct_base",
    "fpu_reduction": "fpu_reduction",
}


def _seed_search_options_from_args(
    options: EngineOptions, args: argparse.Namespace,
) -> None:
    """Point ``options`` at the values the worker is ABOUT to be built with.

    Coerces through ``SearchOption.kind`` so the handshake prints the same type
    the ``setoption`` handler would parse back.
    """
    unmapped = [o.name for o in SEARCH_OPTIONS if o.field not in _SEARCH_OPTION_ARG]
    if unmapped:
        raise AssertionError(
            f"search options with no startup source: {unmapped}. Add them to "
            "_SEARCH_OPTION_ARG (or map them to None if no CLI flag sets "
            "them) — otherwise the first `uci` advertises a default the "
            "engine is not running."
        )
    for opt in SEARCH_OPTIONS:
        dest = _SEARCH_OPTION_ARG[opt.field]
        if dest is None:
            continue
        raw = getattr(args, dest)
        value: float | int | bool
        if opt.kind == "check":
            value = bool(raw)
        elif opt.kind == "spin":
            value = int(raw)
        else:
            value = float(raw)
        options.set_search_value(opt.field, value)


def _startup_engine_options(
    args: argparse.Namespace,
    *,
    search_parallel: str,
    restore_multi_gpu_pucv: bool,
) -> EngineOptions:
    """The options object the pre-build ``uci`` handshake is answered from.

    Construction and search-option seeding live together on purpose: a separate
    ``_seed_search_options_from_args(...)`` call at the call site is a line
    somebody can drop, and the symptom would be a silently wrong handshake
    rather than an error.
    """
    options = EngineOptions(
        threads=max(1, int(args.walkers)),
        leaf_gather=max(1, int(args.walker_gather)),
        use_multi_gpu_pucv=restore_multi_gpu_pucv,
        pucv_pending_mode=str(args.pucv_pending_mode),
        search_parallel=search_parallel,
        vl_gather=max(32, int(args.vl_gather)),
        max_batch=max(64, int(args.max_batch)),
        eval_cache_entries=max(0, int(args.eval_cache_entries)),
  # Clamp the time knobs (like the spin options above): a non-positive
  # time_budget_scale would collapse every move to the 20ms floor, and a
  # negative abort_factor would abort the instant the move leads. 0.0 is the
  # maximally-aggressive-but-valid abort_factor (stop on any positive lead).
        abort_factor=max(0.0, float(args.abort_factor)),
        time_budget_scale=max(0.1, float(args.time_budget_scale)),
  # <= 0 is the OFF sentinel (preserved through the clamp); positive values are
  # clamped to (0, 1] inside limits_from_go so an out-of-range soft target can't
  # exceed the hard deadline.
        optimum_fraction=min(1.0, float(args.optimum_fraction)),
        moves_horizon=max(1, int(args.moves_horizon)),
    )
  # The whole search surface — including the CPuct family, which used to be
  # spelled out as three constructor keywords here.
    _seed_search_options_from_args(options, args)
    return options


def main() -> int:
    p = _build_parser()
    args = p.parse_args()

    if not args.checkpoint:
        p.error(
            "--checkpoint is required (or set DEEPFIN_CKPT in the environment). "
            "GUI launchers typically set DEEPFIN_CKPT in the shell profile."
        )

  # Logs must go to stderr — stdout is reserved for UCI protocol.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

  # Configure shared inductor + triton cache before any torch.compile
  # call. Mirrors selfplay (worker._configure_shared_compile_cache) so a
  # `deepfin` UCI invocation reuses the same compiled kernels as training.
    if args.compile:
        from chess_anti_engine.worker import _configure_shared_compile_cache
        from pathlib import Path
        _configure_shared_compile_cache(
            cache_dir=Path(args.compile_cache_dir).expanduser(),
        )

  # UCI assumes line-buffered I/O. When a GUI pipes stdout, Python defaults
  # to block-buffered, which swallows our responses until the buffer fills.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

  # --devices wins over --device when set (explicit multi-GPU list).
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = [_pick_device(args.device)]
  # gumbel mode requires >= 2 devices; with one device the new module is
  # never constructed and dispatch falls through to the classic paths.
    (
        use_root_parallel_gumbel,
        restore_multi_gpu_pucv,
        use_multi_gpu_pucv,
    ) = _resolve_multi_gpu_startup(
        str(args.search_parallel), args.multi_gpu_pucv, len(devices),
    )
    startup_options = _startup_engine_options(
        args,
        search_parallel="gumbel" if use_root_parallel_gumbel else "pucv",
        restore_multi_gpu_pucv=restore_multi_gpu_pucv,
    )

  # Background-build so `uci` can be answered before model load finishes.
  # Any command other than uci/quit blocks on `engine_ready` below, which
  # gives us correct `readyok` semantics (readyok only fires once the
  # engine truly exists) for free.
    engine_ref: list[Engine | None] = [None]
    engine_error: list[BaseException | None] = [None]
    engine_ready = threading.Event()

    def _build() -> None:
        try:
            n_walkers = max(1, int(args.walkers))
            walker_gather = max(1, int(args.walker_gather))
  # c_scale / c_visit shape the Gumbel sigma(q) value transform; the PUCT
  # walker pool (n_walkers > 1, the --walkers/Threads default) selects on
  # raw Q and never reads them, so the tuned c_scale=0.025 win only lands on
  # the classic Gumbel path. Warn so the inert tuning isn't silent. The
  # multi-GPU pucv pool is plain PUCT too and overrides both other paths,
  # so it gets the same warning.
            if use_multi_gpu_pucv:
                _println(
                    "info string c-scale/c-visit are Gumbel-only and ignored "
                    "while the multi-GPU PUCV pool is active (plain PUCT); "
                    "run a single device with Threads/--walkers 1 for the "
                    "c_scale-tuned classic Gumbel path"
                )
            elif n_walkers > 1:
                _println(
                    "info string c-scale/c-visit are Gumbel-only and ignored at "
                    f"walkers={n_walkers} (PUCT walker pool); set Threads/--walkers 1 "
                    "for the c_scale-tuned classic Gumbel path"
                )
            models = _load_models(args.checkpoint, devices)
            input_history_encoding = str(getattr(models[0], "input_history_encoding", "legacy"))
            input_extra_features = str(getattr(models[0], "input_extra_features", "v1"))
            policy_encoding = str(getattr(models[0], "policy_encoding", "lc0_1858"))
            use_dynamic_relations = bool(getattr(models[0], "use_dynamic_relations", False))
            if use_dynamic_relations:
                _println(
                    "info string model uses dynamic relations; transporting "
                    "relation matrices through search"
                )
            compile_mode = str(args.compile_mode) if args.compile else None
  # When multi-GPU pool / RPG owns search, the primary stack is only for
  # rare root-eval / option-fallback. Compiling it with CUDA graphs first
  # (on the build / CUDA-owner thread) poisons subsequent pool-worker
  # captures on the same process ("captures_underway" / capture failures).
  # Keep primary eager (or explicitly no-graph) so pool workers own the
  # only cudagraph capture sequence under _MULTI_GPU_COMPILE_LOCK.
            multi_gpu_search = len(devices) > 1 and (
                use_multi_gpu_pucv or use_root_parallel_gumbel
            )
            primary_compile_mode = None if multi_gpu_search else compile_mode
            if multi_gpu_search and compile_mode is not None:
                _println(
                    "info string multi-GPU: primary evaluator left eager "
                    "(pool workers own compiled/cudagraph capture)"
                )
            build_eval = _make_evaluator_factory(
                models, devices, coalesce=bool(args.coalesce),
                n_walkers=n_walkers, walker_gather=walker_gather,
                compile_mode=primary_compile_mode,
            )
            build_pucv_factories = (
                _make_multi_gpu_pucv_factory_builder(
                    models,
                    devices,
                    compile_mode=compile_mode,
                )
                if len(devices) > 1
                else None
            )
  # Primary stack: when the multi-GPU PUCV pool owns search, only device 0
  # is needed for root eval (and for the walker fallback after opt-out until
  # rebuild expands it). Avoid building MultiGPUDispatcher + pool factories
  # (≈2× per-device memory). rebuild_evaluator closes over startup_options so
  # MaxBatch / UseMultiGpuPUCV toggles keep the sizing correct.
            def rebuild_evaluator(
                max_batch: int, eval_cache_entries: int = 0,
                *, n_walkers: int | None = None,
            ):
                return build_eval(
                    max_batch,
                    eval_cache_entries,
                    n_walkers=n_walkers,
                    primary_only=bool(
                        startup_options.use_multi_gpu_pucv
                        and startup_options.search_parallel == "pucv"
                    ),
                )

  # Initial build: warms the evaluator too (see factory body).
            evaluator = rebuild_evaluator(
                args.max_batch, startup_options.eval_cache_entries,
            )
            engine_ref[0] = _build_engine(
                evaluator=evaluator, primary_device=devices[0],
                chunk_sims=args.chunk_sims, topk=args.topk,
                c_scale=args.c_scale, c_visit=args.c_visit,
                c_puct=args.c_puct,
                cpuct_factor=float(args.cpuct_factor),
                cpuct_base=float(args.cpuct_base),
                fpu_reduction=args.fpu_reduction,
                c_visit_root=args.c_visit_root, q_visit_floor=args.q_visit_floor,
                q_visit_exp=args.q_visit_exp, q_global_scale=bool(args.q_global_scale),
                c_scale_root=args.c_scale_root, q_visit_exp_root=args.q_visit_exp_root,
                policy_temp=float(args.policy_temp),
                halving_div=int(args.halving_div),
                root_noise_scale=float(args.gumbel_scale),
                n_walkers=n_walkers, vloss_weight=int(args.vloss_weight),
                walker_gather=walker_gather,
                pucv_vloss_mode=1 if args.pucv_pending_mode == "virtual-mean" else 0,
                max_batch=startup_options.max_batch,
                vl_gather=startup_options.vl_gather,
                eval_cache_entries=startup_options.eval_cache_entries,
                use_multi_gpu_pucv=use_multi_gpu_pucv,
                use_root_parallel_gumbel=use_root_parallel_gumbel,
                devices=tuple(devices),
                input_history_encoding=input_history_encoding,
                input_extra_features=input_extra_features,
                policy_encoding=policy_encoding,
                compute_relations=use_dynamic_relations,
                rebuild_evaluator=rebuild_evaluator,
                rebuild_multi_gpu_pucv_factories=build_pucv_factories,
                options=startup_options,
                gil_profile=bool(args.gil_profile),
            )
  # Warm the SEARCH path (not just the evaluator) before reporting ready, so
  # the first real `go` doesn't pay cudagraph capture mid-move and forfeit on
  # the clock. readyok gates on engine_ready below, so a GUI that waits for
  # readyok (the standard handshake) never starts the clock until this returns.
            engine_ref[0].warmup_search()
        except BaseException as exc:  # pragma: no cover — surfaced via readyok
            engine_error[0] = exc
        finally:
            engine_ready.set()

    threading.Thread(target=_build, daemon=True, name="deepfin-build").start()

    try:
        for raw in sys.stdin:
            cmd = parse_command(raw)
            if isinstance(cmd, CmdUci):
                emit_handshake(startup_options)
                continue
            if isinstance(cmd, CmdQuit):
                break
            if not engine_ready.is_set():
                engine_ready.wait()
            if engine_error[0] is not None:
                _println(f"info string engine load failed: {engine_error[0]!r}")
                raise engine_error[0]
            engine = engine_ref[0]
            assert engine is not None
            engine.dispatch(cmd)
            if engine.quit_requested:
                break
    finally:
  # Close whatever evaluator is CURRENT at shutdown time, not a
  # one-shot snapshot — ``MaxBatch`` setoption rebuilds the evaluator
  # via ``SearchWorker.set_evaluator``, which already closes the old
  # one; ``engine.close()`` then closes the live one. Guarantees the
  # non-daemon submitter thread joins before Python's interpreter
  # shutdown starts tearing down PyTorch's CUDA context.
        eng = engine_ref[0]
        if eng is not None:
            with contextlib.suppress(Exception):
                eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
