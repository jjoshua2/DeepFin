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
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    fill_lc0_history_repeat,
    lc0_gather_context_from_planes,
)
from chess_anti_engine.moves import COMPACT_POLICY_SIZE, policy_batch_to_full_if_needed
from chess_anti_engine.moves.leela_index import leela_gather_indices


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
        Name of the WDL output. Shape ``(B, 3)`` expected. LC0/Ceres value
        heads emit softmaxed PROBABILITIES here; ``forward`` returns them as
        log-probs so the search value path's softmax recovers the distribution.
    providers:
        ORT execution providers, in priority order. Default tries CUDA then
        falls back to CPU.
    plane_count:
        How many of our 146 planes the ONNX model expects. LC0/Ceres = 112.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        input_name: str,
        policy_output_name: str,
        wdl_output_name: str,
        providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        plane_count: int = LC0_FULL.num_planes,
    ) -> None:
        super().__init__()
        # Local import — onnxruntime is heavy and only needed when this class is used.
        import onnxruntime as ort

        self._path = str(Path(path).expanduser().resolve())
        self._session = ort.InferenceSession(self._path, providers=list(providers))
        self._input_name = input_name
        self._policy_out = policy_output_name
        self._wdl_out = wdl_output_name
        self._plane_count = plane_count

        # Declare the LC0-canonical input contract so the UCI/match/evaluator
        # helpers (which read these off the model, defaulting to legacy/v1/
        # az_4672) encode positions the way an LC0/Ceres net + the lc0_root
        # history fill in forward() expect — otherwise they'd feed legacy
        # 112-plane inputs and the fill would corrupt them.
        self.input_history_encoding = "lc0_root"
        self.input_extra_features = "v1"  # extras past plane 112 are sliced off
        self.use_dynamic_relations = False
        self.policy_encoding = "az_4672"  # forward() returns 4672-wide policy

    @property
    def device(self) -> torch.device:
        # ORT picks its own device; report CPU so torch consumers don't try to .to() us.
        return torch.device("cpu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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
        out_pol_1858, out_wdl = self._session.run(
            [self._policy_out, self._wdl_out],
            {self._input_name: np_in},
        )
        # Reorder Leela's 1858 into OUR compact 1858, then widen to 4672. The
        # reorder is per-position because the shared back-rank and castling
        # slots depend on the board — which is read from `np_in`, the very
        # planes the net just saw, so the two can never drift apart.
        pol_leela = np.asarray(out_pol_1858, dtype=np.float32)
        if pol_leela.shape[-1] != COMPACT_POLICY_SIZE:
            raise ValueError(
                f"expected policy shape (B, {COMPACT_POLICY_SIZE}), got {tuple(pol_leela.shape)}"
            )
        gather = leela_gather_indices(*lc0_gather_context_from_planes(
            np_in, input_history_encoding=self.input_history_encoding,
        ))
        pol_compact = np.take_along_axis(pol_leela, gather, axis=1)
        # Slots with no geometric move become -1e9 so the legal mask filters them.
        pol_4672 = torch.from_numpy(
            policy_batch_to_full_if_needed(pol_compact, fill_value=-1e9),
        )
        # The search value path (_value_scalar_from_wdl_logits) softmaxes `wdl`,
        # so it must receive logits. LC0/Ceres value heads emit softmaxed
        # PROBABILITIES; feeding those through unchanged would crush a near-certain
        # [1,0,0] to ~0.58. Auto-detect by the two signals that separate probs
        # from logits: probabilities are non-negative AND sum to ~1 (a loose
        # tolerance so fp16/quantized rows summing to e.g. 0.98 still qualify);
        # raw logits are unbounded and ~never both. Probs -> log-probs (softmax
        # recovers them); logits pass through unchanged.
        wdl_raw = torch.from_numpy(out_wdl).to(torch.float32)
        row_sums = wdl_raw.sum(dim=-1)
        is_probs = bool((wdl_raw >= -1e-4).all()) and bool((row_sums - 1.0).abs().lt(0.1).all())
        wdl = torch.log(wdl_raw.clamp_min(1e-9)) if is_probs else wdl_raw
        return {"policy_own": pol_4672, "policy": pol_4672, "wdl": wdl}

