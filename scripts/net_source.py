#!/usr/bin/env python3
"""One way for a ruler to say WHICH net it is scoring: a checkpoint or an ONNX file.

``scripts/value_regret.py`` and ``scripts/audit_targets.py`` used to take
``--checkpoint`` and nothing else, so a foreign net (LC0/BT4, Ceres) could not
be put on the frozen audit set at all. This module adds the second source and
makes the two mutually exclusive, so a run can never be ambiguous about the
weights behind its number.

⚑ THE FAILURE THIS FILE EXISTS TO PREVENT is a flag that is parsed and then
silently ignored — ``--onnx`` accepted while a checkpoint is scored, or an
ONNX graph whose policy head is not the one that was read. So:

* :class:`NetSource` refuses to exist unless EXACTLY ONE source is set. There
  is no default and no fallback: an unloadable net raises, it never degrades to
  something that produces a plausible number.
* the ONNX graph's input/policy/WDL tensor names are RESOLVED AND PRINTED at
  argument-parse time, before the audit set or Stockfish costs anything. An
  ambiguous graph (two 1858-wide outputs) raises rather than picking one.
* every caller prints :attr:`NetSource.label` on its report line, so the number
  and the net that produced it cannot be separated afterwards.

The foreign net itself is loaded through
:class:`chess_anti_engine.onnx.load.OnnxChessNet`, which is the ONLY correct
way in: it slices our 146-plane encoding down to LC0's 112, fills the LC0
history the way BT4 expects, and — the part no static table can do — remaps
Leela's 1858 policy ordering into ours per position, reading the board off the
very planes the net saw (``moves/leela_index.py``). Our ``lc0_1858`` and
Leela's 1858 agree on 46 of 1858 slots, and castling/promotion cannot be mapped
statically at all; the old static table under-weighted the O-O prior by
49-120x without raising. Do not hand-roll a second path in here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chess_anti_engine.moves import COMPACT_POLICY_SIZE

if TYPE_CHECKING:
    import torch

CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"

# LC0/Ceres value heads are 3-wide (win/draw/loss).
WDL_WIDTH = 3


def onnx_providers_for_device(device: str) -> tuple[str, ...]:
    """ORT providers implied by the ruler's ``--device``.

    ``--device cpu`` yields CPU ONLY — not "CPU preferred". A ruler asked for
    CPU must be structurally unable to allocate on the GPU that a live training
    run is using, and the way to guarantee that is to never offer ORT the CUDA
    provider in the first place.

    A CUDA device must have passed :func:`validate_onnx_device` first; this
    function only maps it.
    """
    if str(device).startswith("cuda"):
        return (CUDA_PROVIDER, CPU_PROVIDER)
    return (CPU_PROVIDER,)


def validate_onnx_device(device: str) -> None:
    """Refuse the two ways ``--onnx --device cuda...`` silently means something else.

    Both are the same defect in different clothing — the run claims a device it
    is not on — and both are cheap to catch before anything is loaded:

    1. **An INDEXED device.** ``OnnxChessNet`` takes provider NAMES only, so ORT
       gets no ``device_id`` and uses CUDA device 0, while torch and
       ``--gpu-mem-fraction`` target the index that was asked for. On a
       multi-GPU box that quietly puts a 700M foreign net on whichever GPU is
       device 0 — here, the one the live trainer owns. Until the adapter takes a
       provider-options mapping (its signature is ``Sequence[str]``, and it is
       not this PR's file), an explicit non-zero index is REFUSED rather than
       silently relocated.
    2. **A CUDA provider that is not installed.** ORT does not fail when a
       requested provider is unavailable — it drops it and runs the next one in
       the list, so a `--device cuda` run of a BT4-sized net silently becomes an
       hours-long CPU run that still reports itself as CUDA. The availability
       list is what decides that fallback, so checking it up front is equivalent
       to checking the session afterwards, and it costs nothing.
    """
    dev = str(device)
    if not dev.startswith("cuda"):
        return
    if ":" in dev:
        index = dev.split(":", 1)[1]
        if index not in ("", "0"):
            raise SystemExit(
                f"--onnx --device {dev}: onnxruntime would use CUDA device 0 here, "
                "not the index you asked for (the ONNX adapter takes provider names "
                "only, with no device_id), so the net would land on a GPU the rest "
                "of this command is not using. Pass --device cuda (device 0), "
                "--device cpu, or select the GPU with CUDA_VISIBLE_DEVICES."
            )
    import onnxruntime as ort

    available = list(ort.get_available_providers())
    if CUDA_PROVIDER not in available:
        raise SystemExit(
            f"--onnx --device {dev}: this onnxruntime has no {CUDA_PROVIDER} "
            f"(providers: {available}). ORT would silently fall back to CPU and the "
            "run would report itself as a CUDA run while taking hours. Install the "
            "GPU runtime or pass --device cpu deliberately."
        )


@dataclass(frozen=True)
class OnnxNetSpec:
    """A foreign ONNX net plus the three tensor names needed to drive it."""

    path: Path
    input_name: str
    policy_output: str
    wdl_output: str

    @property
    def label(self) -> str:
        return (
            f"onnx:{self.path} "
            f"[in={self.input_name} policy={self.policy_output} wdl={self.wdl_output}]"
        )


def _pick_output(
    outputs: list[tuple[str, int | str | None]], width: int, *, kind: str, path: Path,
) -> str:
    """The unique output of the given last-dim width, or a loud failure.

    Ambiguity RAISES. Silently taking the first 1858-wide head of two would be
    exactly this codebase's signature defect: a number that means something
    other than what its name says, with nothing in the output to show it.
    """
    named = [n for n, w in outputs if w == width]
    if len(named) == 1:
        return named[0]
    described = ", ".join(f"{n}(last_dim={w})" for n, w in outputs) or "<none>"
    if not named:
        raise SystemExit(
            f"{path}: no {width}-wide {kind} output in the ONNX graph; outputs are "
            f"{described}. Name it explicitly with --onnx-{kind}-output."
        )
    raise SystemExit(
        f"{path}: {len(named)} candidate {kind} outputs of width {width} "
        f"({', '.join(named)}); pick one with --onnx-{kind}-output."
    )


def resolve_onnx_spec(
    path: str | Path,
    *,
    input_name: str | None = None,
    policy_output: str | None = None,
    wdl_output: str | None = None,
) -> OnnxNetSpec:
    """Read the graph's tensor names (or take the explicit ones) and validate them.

    Costs one throwaway ``InferenceSession`` on the CPU provider, which for a
    BT4-sized graph is a few seconds and ~1G of RSS, both released before the
    scoring session is built. The graph is opened even when all three names are
    given explicitly, deliberately: a typo'd override would otherwise be
    accepted here and only surface from ORT after the audit set had loaded and
    Stockfish had spent an hour labelling — which is the cost this function
    exists to avoid.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"--onnx: no such file: {p}")
    # Local import: onnxruntime is heavy and only the --onnx path needs it.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(p), providers=[CPU_PROVIDER])
    try:
        inputs = [str(i.name) for i in sess.get_inputs()]
        outputs: list[tuple[str, int | str | None]] = [
            (str(o.name), o.shape[-1] if o.shape else None) for o in sess.get_outputs()
        ]
    finally:
        # The scoring session is built later by OnnxChessNet; do not hold two
        # copies of a 700M graph alive at once.
        del sess
    if input_name is None:
        if len(inputs) != 1:
            raise SystemExit(
                f"{p}: expected exactly one graph input, found {inputs}; "
                "name it with --onnx-input-name."
            )
        input_name = inputs[0]
    elif input_name not in inputs:
        raise SystemExit(f"{p}: --onnx-input-name {input_name!r} is not in {inputs}")
    known = {n for n, _ in outputs}
    for flag, chosen in (("policy", policy_output), ("wdl", wdl_output)):
        if chosen is not None and chosen not in known:
            raise SystemExit(
                f"{p}: --onnx-{flag}-output {chosen!r} is not an output of this "
                f"graph ({sorted(known)})"
            )
    return OnnxNetSpec(
        path=p.resolve(),
        input_name=input_name,
        policy_output=policy_output
        or _pick_output(outputs, COMPACT_POLICY_SIZE, kind="policy", path=p),
        wdl_output=wdl_output or _pick_output(outputs, WDL_WIDTH, kind="wdl", path=p),
    )


@dataclass(frozen=True)
class NetSource:
    """EXACTLY ONE of a checkpoint path or a resolved ONNX spec."""

    checkpoint: str | None = None
    onnx: OnnxNetSpec | None = None

    def __post_init__(self) -> None:
        if (self.checkpoint is None) == (self.onnx is None):
            raise SystemExit(
                "pass exactly one of --checkpoint (one of ours) or --onnx (a "
                "foreign LC0/Ceres net); "
                + (
                    "both were given, and the ruler must not guess which net the "
                    "number belongs to"
                    if self.checkpoint is not None
                    else "neither was given"
                )
            )

    @property
    def label(self) -> str:
        """What the report line prints. Never empty, never a bare 'model'."""
        return self.onnx.label if self.onnx is not None else str(self.checkpoint)

    @property
    def is_onnx(self) -> bool:
        return self.onnx is not None

    def load(self, *, device: str) -> torch.nn.Module:
        """Build the net in eval mode. Raises rather than returning a stand-in."""
        if self.onnx is not None:
            from chess_anti_engine.onnx.load import OnnxChessNet

            model = OnnxChessNet(
                self.onnx.path,
                input_name=self.onnx.input_name,
                policy_output_name=self.onnx.policy_output,
                wdl_output_name=self.onnx.wdl_output,
                providers=onnx_providers_for_device(device),
            )
        else:
            # Imported here (not at module scope) so the monkeypatch the audit
            # tests apply to the module attribute is what this call resolves.
            from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

            assert self.checkpoint is not None  # __post_init__ guarantees it
            model = load_model_from_checkpoint(self.checkpoint, device=device)
        model.eval()
        return model


def reject_stored_encoding_for_onnx(net: NetSource, input_encoding: str) -> None:
    """``--input-encoding stored`` cannot describe a foreign net. Say so at once.

    Stored rows are production's 175-plane ``v2_threats``/``stored`` bytes; every
    ONNX source built here declares ``lc0_root``/``v1``. The mismatch IS caught
    downstream (``MatchedAuditRows.require_model_compatible``), but only once the
    model loads — which in ``audit_targets`` is after ``_shallow_sf_records``, i.e.
    potentially an hour of Stockfish spent on a combination that was knowably
    invalid the moment the flags were parsed.
    """
    from chess_anti_engine.eval.audit_history import normalize_input_encoding

    if net.is_onnx and normalize_input_encoding(input_encoding) == "stored":
        raise SystemExit(
            "--input-encoding stored is not compatible with --onnx: stored rows are "
            "production's 175-plane input, and a foreign LC0/Ceres net reads the "
            "112-plane lc0_root layout. Score foreign nets under the default "
            "fen_only (and compare them only against fen_only numbers)."
        )


def add_net_source_args(ap: argparse.ArgumentParser, *, checkpoint_help: str = "") -> None:
    """Add ``--checkpoint`` / ``--onnx`` (+ tensor-name overrides) to a ruler.

    ``--checkpoint`` is deliberately NOT ``required``: requiredness moves to
    :func:`net_source_from_args`, which enforces exactly-one across both flags.
    """
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=checkpoint_help
        or "one of our checkpoints (trainer.pt or checkpoint dir). Mutually "
        "exclusive with --onnx; exactly one is required.",
    )
    ap.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="score a FOREIGN net instead: an LC0/Ceres .onnx file (e.g. "
        "data/lc0/onnx/BT4-it332-vanilla-winner.onnx). Fed through "
        "chess_anti_engine/onnx/load.py, which slices our planes to LC0's 112, "
        "fills the LC0 history and remaps Leela's 1858 policy order into ours "
        "per position. The number lands on the SAME frozen audit set as a "
        "checkpoint's, so the two are directly comparable at a pinned "
        "--batch-size and --input-encoding.",
    )
    ap.add_argument(
        "--onnx-input-name",
        default=None,
        help="override the ONNX input tensor name (default: the graph's only "
        "input; ambiguity raises).",
    )
    ap.add_argument(
        "--onnx-policy-output",
        default=None,
        help=f"override the ONNX policy output name (default: the graph's only "
        f"{COMPACT_POLICY_SIZE}-wide output; ambiguity raises).",
    )
    ap.add_argument(
        "--onnx-wdl-output",
        default=None,
        help=f"override the ONNX WDL output name (default: the graph's only "
        f"{WDL_WIDTH}-wide output; ambiguity raises).",
    )


def net_source_from_args(args: argparse.Namespace) -> NetSource:
    """Build the :class:`NetSource` a parsed ruler CLI asks for.

    Resolves (and prints) the ONNX tensor names here, at parse time, so a typo
    or a graph with no 1858 head fails before the audit set is loaded and long
    before Stockfish spends an hour labelling.
    """
    onnx_path = getattr(args, "onnx", None)
    onnx_name_flags = {
        "--onnx-input-name": getattr(args, "onnx_input_name", None),
        "--onnx-policy-output": getattr(args, "onnx_policy_output", None),
        "--onnx-wdl-output": getattr(args, "onnx_wdl_output", None),
    }
    if onnx_path is None:
        # A name override without --onnx means the caller believes it is
        # scoring a foreign net and is not. Say so instead of ignoring it.
        given = sorted(f for f, v in onnx_name_flags.items() if v is not None)
        if given:
            raise SystemExit(f"{', '.join(given)} given without --onnx")
        return NetSource(checkpoint=args.checkpoint)
    if args.checkpoint is not None:
        # NetSource would reject this anyway; saying it before the ONNX graph
        # is opened keeps the message about the mistake, not about the file.
        raise SystemExit(
            "pass exactly one of --checkpoint or --onnx; both were given, and "
            "the ruler must not guess which net the number belongs to"
        )
    # Device first: it is the cheapest check and the one whose failure mode is
    # "the number is real but came from somewhere else".
    validate_onnx_device(str(getattr(args, "device", "cpu")))
    spec = resolve_onnx_spec(
        onnx_path,
        input_name=onnx_name_flags["--onnx-input-name"],
        policy_output=onnx_name_flags["--onnx-policy-output"],
        wdl_output=onnx_name_flags["--onnx-wdl-output"],
    )
    print(f"[net-source] foreign ONNX net {spec.label}")
    return NetSource(onnx=spec)
