"""Load a foreign ONNX chess net (CeresNets / LC0) for use in our search.

Wraps an ``onnxruntime.InferenceSession`` in a ``torch.nn.Module`` shape so it
plugs into the existing inference paths (``DirectGPUEvaluator`` and the
``run_gumbel_root_many_c`` head). The callers expect ``model(x) -> dict`` with
``policy_own`` and ``wdl`` keys; this wrapper provides that contract.

Plane convention: our 146-plane encoding has the first 112 planes
LC0-identical (history) and 34 extra classical planes. Foreign LC0/Ceres
nets only consume the first 112, so we slice on the way in.

Policy convention: we use 4672 (square × 73 directions). LC0/Ceres use their
own 1858 enumeration, which is NOT our compact 1858 — the two orderings agree
on 46 of 1858 slots. Worse, two families of move cannot be mapped by any static
table, because the same slot means different things depending on the board:

  - the plain back-rank slot (``a7a8``) covers a non-pawn slide OR a promotion
    to QUEEN for us, and a slide OR a promotion to KNIGHT for Leela;
  - castling is ``e1g1`` for us and king-takes-rook ``e1h1`` for Leela — and
    Leela's table also contains an ordinary ``e1g1`` slide entry, so a static
    lookup silently reads an unrelated logit instead of failing.

Both are resolved here by reading the board context off the input planes and
using :mod:`chess_anti_engine.moves.leela_index`. The castling case is the
bigger error of the two: measured on BT4, the O-O prior read through the old
static table came out 49x-120x too small, dropping castling from the top move
to nowhere. The previous ``build_lc0_policy_remap`` helper implemented that
static table and has been removed rather than left as a footgun.

Ceres's own 1858 table is Leela's, verbatim: ``EncodedMove.NEURAL_NET_MOVE_STR``
(``src/Ceres.Chess/LC0/Positions/Basic/EncodedMove.cs``) agrees with our
``LC0_1858_UCI_TO_IDX`` on all 1858 slots, so the same remap serves both. Only
the INPUT differs, which is what ``input_format`` selects:

``lc0_planes``
    (B, planes, 8, 8); the first ``plane_count`` planes are fed to the net.
``ceres_tpg``
    (B, 64, 137) ``TPGSquareRecord`` values from
    :mod:`chess_anti_engine.encoding.ceres_tpg`. Ceres C1 nets take this and
    ONLY this — the 137 per-square features are a different feature set from
    LC0's 112 planes, not a reshape of them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.encoding.ceres_tpg import (
    CERES_TPG_NUM_FEATURES,
    CERES_TPG_NUM_SQUARES,
    ceres_tpg_gather_context,
    encode_ceres_tpg,
)
from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    encode_lc0_full,
    fill_lc0_history_repeat,
    lc0_gather_context_from_planes,
)
from chess_anti_engine.moves import COMPACT_POLICY_SIZE, policy_batch_to_full_if_needed
from chess_anti_engine.moves.leela_index import leela_gather_indices

# What ORT accepts in `InferenceSession(providers=...)`: a bare provider name,
# or a (name, options) pair. The pair form is the only one that can carry
# `device_id` and `gpu_mem_limit`.
OnnxProvider = str | tuple[str, Mapping[str, object]]

INPUT_FORMAT_LC0_PLANES = "lc0_planes"
INPUT_FORMAT_CERES_TPG = "ceres_tpg"
ONNX_INPUT_FORMATS = (INPUT_FORMAT_LC0_PLANES, INPUT_FORMAT_CERES_TPG)

WDL_OUTPUT_LOGITS = "logits"
WDL_OUTPUT_PROBABILITIES = "probabilities"
WDL_OUTPUT_AUTO = "auto"
WDL_OUTPUT_KINDS = (WDL_OUTPUT_LOGITS, WDL_OUTPUT_PROBABILITIES, WDL_OUTPUT_AUTO)

_ORT_GRAPH_OPTIMIZATION_LEVELS: dict[str, str] = {
    "disabled": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}

_DEFAULT_GRAPH_OPTIMIZATION: dict[str, str | None] = {
    INPUT_FORMAT_LC0_PLANES: None,  # ORT's own default — BT4 session unchanged
    INPUT_FORMAT_CERES_TPG: "extended",
}

_ORT_INPUT_DTYPES: dict[str, type[np.floating]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
}


def declared_input_contract(input_format: str) -> tuple[str, str]:
    """``(input_history_encoding, input_extra_features)`` a net of this format
    should advertise to the UCI/match/evaluator helpers.

    For ``lc0_planes`` that is the LC0-canonical layout those helpers must use
    for an LC0/Ceres plane net + the history fill in ``forward``; anything else
    and they would feed legacy 112-plane inputs the fill then corrupts.

    For ``ceres_tpg`` it is deliberately a name NO plane encoder accepts. Those
    helpers can only emit planes, and a 64x137 TPG record is not planes — so the
    honest declaration is one that makes ``normalize_lc0_history_encoding``
    RAISE. Declaring ``lc0_root`` here (as this class did before the input-format
    split) would let a helper build 112 LC0 planes for a net that cannot read
    them: accepted, and silently meaningless. Callers of the ceres path build the
    input with ``encoding.ceres_tpg.encode_ceres_tpg_batch`` instead.

    A pure function rather than four lines inside ``__init__`` so the choice is
    observable without opening an ONNX session.
    """
    if input_format == INPUT_FORMAT_CERES_TPG:
        return INPUT_FORMAT_CERES_TPG, INPUT_FORMAT_CERES_TPG
    return "lc0_root", "v1"  # extras past plane 112 are sliced off


class OnnxChessNet(torch.nn.Module):
    """Adapter: ``onnxruntime`` session ↔ our 146-plane / 4672-policy contract.

    Parameters
    ----------
    path:
        ``.onnx`` model on disk.
    input_name:
        Name of the input tensor in the ONNX graph (use ``inspect_onnx.py``
        to find it).
    policy_output_name:
        Name of the policy logits output. Shape ``(B, 1858)`` expected.
    wdl_output_name:
        Name of the WDL output. Shape ``(B, 3)`` expected, ordered W/D/L from
        the SIDE TO MOVE's point of view. ``forward`` returns something the
        search's softmax can consume either way: an LC0 head that emits
        softmaxed PROBABILITIES is converted to log-probs, and a head that
        emits LOGITS is passed through. Which one a net is cannot be assumed —
        BT4 emits probabilities, while Ceres's ``value`` is logits
        (``NNEvaluatorONNX`` is built with ``valueHeadLogistic: true``, and the
        raw rows carry negatives and do not sum to 1) — see ``wdl_output_kind``.
        For Ceres pass ``value``, the PRIMARY head (``ONNXNetExecutor``'s
        ``INDEX_WDL``, which feeds ``W1/L1``); ``value2`` is a secondary head
        Ceres blends in at ``FractionValueHead2`` 0.4 with its own temperature,
        not an equivalent WDL output.
    providers:
        ORT execution providers, in priority order. Default tries CUDA then
        falls back to CPU. Each entry is a bare provider NAME or an
        ``(name, options)`` pair — the form ORT needs for ``device_id`` and
        ``gpu_mem_limit``, which are the ONLY way to bound a CUDA session.
        ⚑ ``torch.cuda.set_per_process_memory_fraction`` does NOT reach here:
        it bounds the torch caching allocator, and ORT allocates through its
        own CUDA arena. A ruler that caps torch and hands this bare names has
        an uncapped ~700M session on the trainer's GPU.
    plane_count:
        How many of our 146 planes the ONNX model expects. LC0/Ceres = 112.
        Ignored when ``input_format`` is ``ceres_tpg``.
    input_format:
        ``lc0_planes`` (default, the BT4 path) or ``ceres_tpg``. Selects both
        what ``forward`` accepts and where the policy remap reads its board
        context from.
    graph_optimization_level:
        ORT level name (``basic`` / ``extended`` / ``all`` / ``disabled``).
        ``None`` uses the per-format default: ORT's own for ``lc0_planes``
        (so the BT4 session is byte-for-byte what it always was) and
        ``extended`` for ``ceres_tpg``, because ORT 1.23's ALL-level
        ``SimplifiedLayerNormFusion`` throws on these fp16 Ceres graphs
        (``Attempting to get index by a name which does not exist:
        InsertedPrecisionFreeCast_...``) and the session never opens.
    intra_op_num_threads:
        Optional cap on ORT's intra-op pool. ``None`` leaves ORT's default.
    wdl_output_kind:
        ``logits`` / ``probabilities`` — a DECLARATION about the WDL head — or
        ``auto`` (default), which probes the net ONCE at construction with a
        start position and freezes the answer in ``self.wdl_output_kind``.

        ⚑ It is deliberately not re-decided per batch. The classifier is a
        heuristic over the batch's values ("non-negative and sums to ~1"), so
        re-running it every call makes the answer depend on batch content — and
        at batch 1 that is a coin flip on an axis that fails SILENTLY: a
        probabilities head passed through unchanged crushes a near-certain
        [1, 0, 0] to ~0.58, which looks like a merely cautious evaluation rather
        than a bug. Declare it explicitly for anything that reaches search.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        input_name: str,
        policy_output_name: str,
        wdl_output_name: str,
        providers: Sequence[OnnxProvider] = (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
        plane_count: int = LC0_FULL.num_planes,
        input_format: str = INPUT_FORMAT_LC0_PLANES,
        graph_optimization_level: str | None = None,
        intra_op_num_threads: int | None = None,
        wdl_output_kind: str = WDL_OUTPUT_AUTO,
    ) -> None:
        super().__init__()
        # Local import — onnxruntime is heavy and only needed when this class is used.
        import onnxruntime as ort

        if input_format not in ONNX_INPUT_FORMATS:
            raise ValueError(
                f"input_format must be one of {ONNX_INPUT_FORMATS}; got {input_format!r}",
            )
        if wdl_output_kind not in WDL_OUTPUT_KINDS:
            raise ValueError(
                f"wdl_output_kind must be one of {WDL_OUTPUT_KINDS}; got {wdl_output_kind!r}",
            )
        level = graph_optimization_level or _DEFAULT_GRAPH_OPTIMIZATION[input_format]
        if level is not None and level not in _ORT_GRAPH_OPTIMIZATION_LEVELS:
            raise ValueError(
                f"graph_optimization_level must be one of "
                f"{tuple(_ORT_GRAPH_OPTIMIZATION_LEVELS)}; got {level!r}",
            )
        session_options = None
        if level is not None or intra_op_num_threads is not None:
            session_options = ort.SessionOptions()
            if level is not None:
                session_options.graph_optimization_level = getattr(
                    ort.GraphOptimizationLevel, _ORT_GRAPH_OPTIMIZATION_LEVELS[level],
                )
            if intra_op_num_threads is not None:
                session_options.intra_op_num_threads = int(intra_op_num_threads)
        self._path = str(Path(path).expanduser().resolve())
        # Normalise Mapping options to plain dicts: ORT's pybind layer indexes
        # and copies them as dicts, and a non-dict Mapping is not accepted.
        ort_providers: list[str | tuple[str, dict[str, object]]] = [
            p if isinstance(p, str) else (p[0], dict(p[1])) for p in providers
        ]
        self._session = ort.InferenceSession(
            self._path, session_options, providers=ort_providers,
        )
        self._input_name = input_name
        self._policy_out = policy_output_name
        self._wdl_out = wdl_output_name
        self._plane_count = plane_count
        self.input_format = input_format
        # Feed the dtype the graph declares. BT4 declares tensor(float) so this
        # stays a no-op there; the Ceres C1 exports declare tensor(float16) and
        # ORT rejects a float32 feed outright.
        declared = {i.name: i.type for i in self._session.get_inputs()}
        self._input_dtype = _ORT_INPUT_DTYPES.get(declared.get(input_name, ""), np.float32)

        # Declare the input contract the UCI/match/evaluator helpers read off the
        # model (they default to legacy/v1/az_4672). For lc0_planes that is the
        # LC0-canonical layout the net + the history fill in forward() expect;
        # otherwise they'd feed legacy 112-plane inputs and the fill would
        # corrupt them.
        #
        # For ceres_tpg it is deliberately a name NO plane encoder accepts.
        # Those helpers cannot build a 64x137 TPG record — they only know how to
        # emit planes — so the honest declaration is one that makes
        # `normalize_lc0_history_encoding` RAISE. Declaring "lc0_root" here (as
        # this class did before the split) would have let a helper encode 112
        # LC0 planes for a net that cannot read them, which is silent wrongness
        # rather than a crash. Callers of the ceres path build the input with
        # `encoding.ceres_tpg.encode_ceres_tpg_batch` and pass it to forward().
        self.input_history_encoding, self.input_extra_features = declared_input_contract(
            input_format,
        )
        self.use_dynamic_relations = False
        self.policy_encoding = "az_4672"  # forward() returns 4672-wide policy

        # Resolve WDL logits-vs-probabilities ONCE, never per batch (see the
        # class docstring). An explicit declaration skips the probe entirely.
        self.wdl_output_kind = (
            wdl_output_kind if wdl_output_kind != WDL_OUTPUT_AUTO else self._probe_wdl_kind()
        )

    @property
    def device(self) -> torch.device:
        # ORT picks its own device; report CPU so torch consumers don't try to .to() us.
        return torch.device("cpu")

    def session_providers(self) -> list[str]:
        """The providers that ACTUALLY initialised, read off the live session.

        ⚑ Not ``onnxruntime.get_available_providers()``. That is the list the
        wheel was COMPILED with, and it can name a provider that does not
        start: ORT drops an unusable provider with a warning rather than
        failing, and the compiled list does not move when it does. So this is
        the only reading that can distinguish "capped CUDA session" from
        "silent CPU fallback".
        """
        return [str(p) for p in self._session.get_providers()]

    def _canonical_probe_input(self) -> np.ndarray:
        """A single start-position input in this net's format, for the WDL probe."""
        import chess

        if self.input_format == INPUT_FORMAT_CERES_TPG:
            return encode_ceres_tpg(chess.Board())[None, ...]
        planes = encode_lc0_full(
            chess.Board(), input_history_encoding=self.input_history_encoding,
        )
        if planes.shape[0] < self._plane_count:
            raise ValueError(
                f"probe encoder produced {planes.shape[0]} planes, model needs "
                f"{self._plane_count}",
            )
        return fill_lc0_history_repeat(planes[: self._plane_count].copy())[None, ...]

    def _probe_wdl_kind(self) -> str:
        """Classify the WDL head once, on a known position, at construction.

        Probabilities are non-negative AND sum to ~1 (loose tolerance so an
        fp16/quantised row summing to e.g. 0.98 still qualifies); raw logits are
        unbounded and ~never both. The SAME two signals the old per-batch check
        used — the change is that the answer is now fixed for the life of the
        object instead of being re-decided from whatever happens to be in the
        current batch. At batch 1 that re-decision was a coin flip on an axis
        whose failure mode is silent: feeding probabilities to a consumer that
        softmaxes them crushes a near-certain [1, 0, 0] to ~0.58.
        """
        probe = self._canonical_probe_input().astype(self._input_dtype, copy=False)
        raw = self._session.run([self._wdl_out], {self._input_name: probe})[0]
        row = np.asarray(raw, dtype=np.float64)
        looks_like_probs = (
            bool((row >= -1e-4).all())
            and bool(np.abs(row.sum(axis=-1) - 1.0).max() < 0.1)
        )
        return WDL_OUTPUT_PROBABILITIES if looks_like_probs else WDL_OUTPUT_LOGITS

    def _prepare_lc0_planes(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(net input, gather map) for the 112-plane LC0/BT4 contract."""
        if x.dim() != 4 or x.shape[-2:] != (8, 8):
            raise ValueError(f"expected (B, planes, 8, 8); got {tuple(x.shape)}")
        if x.shape[1] >= self._plane_count:
            x_in = x[:, : self._plane_count]
        else:
            raise ValueError(
                f"input has {x.shape[1]} planes, ONNX model needs >= {self._plane_count}"
            )
        # ORT wants numpy float32 on the CPU ingress; the session moves to CUDA itself.
        # .copy() so the LC0 history fill below never writes through to the
        # caller's tensor (numpy() can alias an already-CPU/float32 input).
        np_in = x_in.detach().to(dtype=torch.float32, device="cpu").numpy().copy()
        # LC0/Ceres nets read history-sensitively and break on zero-filled
        # history (e.g. a rootless UCI `position fen ...` or a debug position).
        # Replicate the live frame into any all-zero history frame, exactly as
        # bt4_audit.py does; real-history inputs have non-empty frames and are
        # left untouched.
        np_in = fill_lc0_history_repeat(np_in)
        gather = leela_gather_indices(*lc0_gather_context_from_planes(
            np_in, input_history_encoding=self.input_history_encoding,
        ))
        return np_in, gather

    def _prepare_ceres_tpg(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(net input, gather map) for the Ceres (B, 64, 137) square-record contract.

        No history fill happens here: the TPG record already carries all 8
        history slots, and short histories were resolved by the encoder's
        ``fill_in_history`` (Ceres's own converter does the same).
        """
        if x.dim() != 3 or tuple(x.shape[1:]) != (CERES_TPG_NUM_SQUARES, CERES_TPG_NUM_FEATURES):
            raise ValueError(
                f"expected (B, {CERES_TPG_NUM_SQUARES}, {CERES_TPG_NUM_FEATURES}) Ceres "
                f"square records; got {tuple(x.shape)}",
            )
        np_in = x.detach().to(dtype=torch.float32, device="cpu").numpy().copy()
        return np_in, leela_gather_indices(*ceres_tpg_gather_context(np_in))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.input_format == INPUT_FORMAT_CERES_TPG:
            np_in, gather = self._prepare_ceres_tpg(x)
        else:
            np_in, gather = self._prepare_lc0_planes(x)
        out_pol_1858, out_wdl = self._session.run(
            [self._policy_out, self._wdl_out],
            {self._input_name: np_in.astype(self._input_dtype, copy=False)},
        )
        # Reorder Leela's 1858 into OUR compact 1858, then widen to 4672. The
        # reorder is per-position because the shared back-rank and castling
        # slots depend on the board — which is read from `np_in`, the very
        # input the net just saw, so the two can never drift apart.
        pol_leela = np.asarray(out_pol_1858, dtype=np.float32)
        if pol_leela.shape[-1] != COMPACT_POLICY_SIZE:
            raise ValueError(
                f"expected policy shape (B, {COMPACT_POLICY_SIZE}), got {tuple(pol_leela.shape)}"
            )
        pol_compact = np.take_along_axis(pol_leela, gather, axis=1)
        # Slots with no geometric move become -1e9 so the legal mask filters them.
        pol_4672 = torch.from_numpy(
            policy_batch_to_full_if_needed(pol_compact, fill_value=-1e9),
        )
        # The search value path (_value_scalar_from_wdl_logits) softmaxes `wdl`,
        # so it must receive logits. A head that emits PROBABILITIES (BT4) is
        # converted to log-probs, which the downstream softmax inverts; a head
        # that emits LOGITS (Ceres `value`) passes through. Which one this net is
        # was decided ONCE at construction — `self.wdl_output_kind` — so the
        # answer cannot vary with batch content. Getting it wrong is silent, not
        # loud: feeding probabilities through unchanged crushes a near-certain
        # [1, 0, 0] to ~0.58.
        wdl_raw = torch.from_numpy(out_wdl).to(torch.float32)
        wdl = (
            torch.log(wdl_raw.clamp_min(1e-9))
            if self.wdl_output_kind == WDL_OUTPUT_PROBABILITIES
            else wdl_raw
        )
        return {"policy_own": pol_4672, "policy": pol_4672, "wdl": wdl}

