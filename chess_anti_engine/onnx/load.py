"""Load a foreign ONNX chess net (CeresNets / LC0) for use in our search.

Wraps an ``onnxruntime.InferenceSession`` in a ``torch.nn.Module`` shape so it
plugs into the existing inference paths (``DirectGPUEvaluator`` and the
``run_gumbel_root_many_c`` head). The callers expect ``model(x) -> dict`` with
``policy_own`` and ``wdl`` keys; this wrapper provides that contract.

Plane convention: our 146-plane encoding has the first 112 planes
LC0-identical (history) and 34 extra classical planes. Foreign LC0/Ceres
nets only consume the first 112, so we slice on the way in.

Policy convention: we use 4672 (square × 73 directions). LC0/Ceres use 1858
(canonical legal-move enumeration). The mapping is supplied as an integer
tensor of length 4672, where idx[i] is the corresponding 1858 slot, or -1
if move i has no LC0 equivalent (rare; underpromotion edge cases). At
inference, we scatter the 1858 logits into a 4672 buffer, masking
unmapped slots to ``-inf`` so the legal-move filter downstream still works.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from chess_anti_engine.encoding.lc0 import LC0_FULL, fill_lc0_history_repeat
from chess_anti_engine.moves import COMPACT_POLICY_SIZE, POLICY_SIZE


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
    policy_4672_to_1858:
        Length-4672 ``int64`` tensor mapping our move index → LC0 1858 slot,
        or -1 if absent. Use ``build_lc0_policy_remap()`` to construct.
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
        policy_4672_to_1858: torch.Tensor,
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

        if policy_4672_to_1858.shape != (POLICY_SIZE,):
            raise ValueError(
                f"policy_4672_to_1858 must be shape ({POLICY_SIZE},), got {tuple(policy_4672_to_1858.shape)}"
            )
        self.register_buffer("_remap", policy_4672_to_1858.to(torch.int64), persistent=False)

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
        # Remap policy 1858 → 4672. -1 entries become -inf so legal-mask filters them.
        pol_1858 = torch.from_numpy(out_pol_1858)
        if pol_1858.shape[-1] != COMPACT_POLICY_SIZE:
            raise ValueError(f"expected policy shape (B, {COMPACT_POLICY_SIZE}), got {tuple(pol_1858.shape)}")
        # Pad with a sentinel column so gather of -1 grabs -inf safely.
        sentinel = torch.full((pol_1858.shape[0], 1), float("-inf"), dtype=pol_1858.dtype)
        padded = torch.cat([pol_1858, sentinel], dim=1)  # shape (B, COMPACT_POLICY_SIZE + 1)
        # _remap stores -1 for missing — translate to the sentinel index COMPACT_POLICY_SIZE.
        # nn.Module buffer typing widens to Tensor|Module; cast back since
        # register_buffer above only takes a Tensor.
        remap = self._remap
        assert isinstance(remap, torch.Tensor)
        gather_idx = remap.clone()
        gather_idx[gather_idx < 0] = COMPACT_POLICY_SIZE
        # Broadcast gather across batch.
        idx = gather_idx.unsqueeze(0).expand(pol_1858.shape[0], -1)
        pol_4672 = torch.gather(padded, 1, idx)
        # The search value path (_value_scalar_from_wdl_logits) softmaxes `wdl`,
        # so it must receive logits. LC0/Ceres value heads emit softmaxed
        # PROBABILITIES; feeding those through unchanged would crush a near-certain
        # [1,0,0] to ~0.58. Auto-detect: if the rows already sum to ~1 they are
        # probabilities → return log-probs (softmax recovers them); otherwise the
        # net already emits raw logits → pass through unchanged.
        wdl_raw = torch.from_numpy(out_wdl).to(torch.float32)
        row_sums = wdl_raw.sum(dim=-1)
        if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2):
            wdl = torch.log(wdl_raw.clamp_min(1e-9))
        else:
            wdl = wdl_raw
        return {"policy_own": pol_4672, "policy": pol_4672, "wdl": wdl}


def build_lc0_policy_remap() -> torch.Tensor:
    """Build the 4672 → 1858 lookup.

    For each of our 4672 (square, direction) move slots, store the matching
    canonical LC0/Leela 1858 policy slot, or -1 if our encoding has no move
    there. Built from the verbatim lc0 ``kMoveStrs`` table (White-POV) and
    verified against the BT4 ONNX policy head (startpos → Nf3/d4, back-rank →
    mate, audit positions → deep-SF bestmove).

    Caveat — the 22 promotion-rank squares (e.g. ``a7a8``/``a7a8q``): Leela
    keeps SEPARATE slots for a piece sliding to the back rank vs a pawn
    promoting to a queen, while our AZ-4672 conflates them into one direction
    plane. A static 4672->1858 lookup therefore CANNOT disambiguate them
    without board context. We map the shared slot to the queen-PROMOTION leela
    index because that is the overwhelmingly common case for that square pair.

    Known bounded inaccuracy (flagged by Codex on #80): a NON-pawn piece sliding
    to the back rank (e.g. a rook on a7 playing ``a7a8``) reuses that same
    AZ-4672 slot, so ``OnnxChessNet`` would gather the LC0 queen-promotion logit
    for it. This is reachable only through ``OnnxChessNet`` (running an LC0/BT4
    net via our 4672 interface) and only in the rare position where a piece —
    not a pawn — moves onto the 8th/1st rank; legal masking keeps the two move
    families position-disjoint so nothing is double-counted, but the logit read
    for such a slide is the queen-promo logit, not the slide's. The BT4 *audit*
    path is unaffected: ``bt4_audit.py`` gathers in 1858 space directly from the
    legal UCIs (board-aware), never through this lossy 4672 remap. A fully
    correct remap would require per-move board/piece context at gather time.
    """
    from chess_anti_engine.moves.encode import uci_to_policy_index
    from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX

    # Iterate the alias dict (not the bare table) so the "...n" knight-promotion
    # UCIs are mapped too: LC0 has no knight-promo slot (it reuses the bare
    # from/to entry), but AZ-4672 keeps a DISTINCT knight-underpromotion plane,
    # so without this the OnnxChessNet gather returns -inf for legal knight
    # promotions and masks them from search. Insertion order (bare/q/r/b first,
    # then the appended "...n" aliases) preserves the queen-promo-wins choice for
    # the shared slot while filling the otherwise-unmapped knight slots.
    remap = torch.full((POLICY_SIZE,), -1, dtype=torch.int64)
    for uci, leela_idx in LC0_1858_UCI_TO_IDX.items():
        our = uci_to_policy_index(uci, True)
        if our >= 0:
            remap[int(our)] = leela_idx
    return remap
