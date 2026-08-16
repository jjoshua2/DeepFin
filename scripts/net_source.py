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
* every caller prints :attr:`NetSource.label` on its report line AND stamps it
  into every per-position dump row, so the number and the net that produced it
  cannot be separated afterwards.
* the same rule is applied to the DEVICE, to the GPU MEMORY CAP and to the
  input encoding, because each has a silent-wrongness mode of its own:

  - :func:`validate_onnx_device` gates the DEVICE at parse time, before the
    audit set is loaded and before Stockfish labels anything. ⚑ Its first check
    — ORT's compile-time provider list — is NECESSARY and nothing more: it can
    name a provider that does not start, so on its own it is a gate that cannot
    fail. What makes it a real gate is :func:`probe_onnx_device_providers`, a
    76-byte throwaway session on the requested device. The same reading is
    repeated on the scoring session by :func:`verify_onnx_session_device`, so a
    ``--device cuda`` run that ORT quietly moved to CPU raises rather than
    reporting a CUDA number.
  - :func:`onnx_providers_for_device` hands ORT ``(name, options)`` pairs, never
    bare names. Bare names were how ``--gpu-mem-fraction`` came to be printed as
    applied while doing nothing on this path:
    ``torch.cuda.set_per_process_memory_fraction`` bounds the TORCH allocator,
    the ONNX session allocates through ORT's own CUDA arena, and only
    ``gpu_mem_limit`` in the provider options bounds that. The options also
    carry ``device_id``, so ``--device cuda:1`` lands on card 1 rather than
    silently on card 0.
  - :func:`reject_stored_encoding_for_onnx` refuses ``--input-encoding stored``
    for a foreign net at parse time rather than after the model loads.

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


def cuda_device_index(device: str) -> int:
    """The GPU index a ``--device`` string names. Bare ``cuda`` is device 0.

    One parser for torch's cap, ORT's ``device_id`` and the log line, so the
    three cannot drift onto different GPUs while all three claim the same one.

    A malformed index is a ``SystemExit`` like every other bad-CLI-input path
    here, not a bare ``ValueError`` traceback: ``--device cuda:x`` is a typo,
    and a typo should print one line rather than a stack.
    """
    dev = str(device)
    if not dev.startswith("cuda"):
        raise ValueError(f"not a CUDA device: {dev!r}")
    _, _, index = dev.partition(":")
    if not index:
        return 0
    try:
        return int(index)
    except ValueError:
        raise SystemExit(
            f"--device {dev}: the CUDA index must be an integer (e.g. cuda, "
            f"cuda:0, cuda:1); got {index!r}",
        ) from None


def validate_gpu_mem_fraction(gpu_mem_fraction: float) -> float:
    """The single range check for ``--gpu-mem-fraction``. Raises ``SystemExit``.

    It lives on its own because there are three entry points — the torch cap,
    the ORT byte conversion, and the conversion's card lookup — and only one of
    them used to check. An out-of-range fraction then reached
    ``torch.cuda.get_device_properties`` or
    ``set_per_process_memory_fraction`` first and surfaced as a torch error
    about something else, or (for a fraction the card lookup survived) as a cap
    computed from a nonsense number.
    """
    value = float(gpu_mem_fraction)
    if not 0.0 < value <= 1.0:
        raise SystemExit(
            f"--gpu-mem-fraction must be in (0, 1]; got {gpu_mem_fraction}",
        )
    return value


def gpu_mem_limit_bytes(gpu_mem_fraction: float, total_bytes: int) -> int:
    """``gpu_mem_limit`` (BYTES) for a ``--gpu-mem-fraction`` of a card.

    ORT's CUDA arena takes an absolute byte budget while our rulers take a
    fraction, so the conversion has to happen somewhere; it lives here, pure,
    so it is checkable without a GPU. Rounded DOWN and floored at 1 byte: a
    rounded-up limit is a cap that is larger than the share it was asked for.
    """
    return max(1, int(validate_gpu_mem_fraction(gpu_mem_fraction) * int(total_bytes)))


def onnx_cuda_mem_limit(
    device: str, gpu_mem_fraction: float | None, *, total_bytes: int | None = None,
) -> int | None:
    """Resolve ``--gpu-mem-fraction`` to ORT's byte budget, or ``None``.

    ``None`` means "there is nothing to cap" — no fraction was asked for, or
    the run is on CPU where ORT allocates no device memory. It never means
    "could not work it out": an unresolvable card raises.

    ``total_bytes`` is injectable so the arithmetic can be exercised on a box
    with no GPU; production leaves it unset and reads the real card.
    """
    if gpu_mem_fraction is None or not str(device).startswith("cuda"):
        return None
    # Range-check BEFORE the card is touched, so a typo'd fraction prints the
    # one line about the fraction rather than a torch error about the device.
    fraction = validate_gpu_mem_fraction(gpu_mem_fraction)
    if total_bytes is None:
        import torch

        total_bytes = int(
            torch.cuda.get_device_properties(cuda_device_index(device)).total_memory,
        )
    return gpu_mem_limit_bytes(fraction, total_bytes)


def onnx_providers_for_device(
    device: str, *, gpu_mem_bytes: int | None = None,
) -> tuple[str | tuple[str, dict[str, object]], ...]:
    """ORT providers implied by the ruler's ``--device`` and memory cap.

    ``--device cpu`` yields CPU ONLY — not "CPU preferred". A ruler asked for
    CPU must be structurally unable to allocate on the GPU that a live training
    run is using, and the way to guarantee that is to never offer ORT the CUDA
    provider in the first place.

    A CUDA device yields the ``(name, options)`` pair form, never a bare name.
    Bare names are what made ``--gpu-mem-fraction`` inert on this path: they
    carry neither ``device_id`` (so ORT used card 0 whatever ``--device`` said)
    nor ``gpu_mem_limit`` (so the ~700M session was unbounded next to a live
    trainer, while the ruler printed that memory had been capped).
    ``scripts/foreign_net_audit.py`` has always passed the pair form; this is
    the same shape.

    ``gpu_mem_bytes`` is ``None`` when no cap was requested — the options then
    carry ``device_id`` alone rather than a made-up limit.

    A CUDA device must have passed :func:`validate_onnx_device` first; this
    function only maps it.
    """
    if not str(device).startswith("cuda"):
        return (CPU_PROVIDER,)
    options: dict[str, object] = {"device_id": cuda_device_index(device)}
    if gpu_mem_bytes is not None:
        options["gpu_mem_limit"] = int(gpu_mem_bytes)
    return ((CUDA_PROVIDER, options), CPU_PROVIDER)


# A 76-byte one-node ONNX graph (Identity, opset 17, ir_version 9), FROZEN as
# bytes rather than rebuilt at runtime. Building it would put an ``onnx`` import
# and that package's ir_version default on the path of every ``--onnx --device
# cuda`` run, and the default is exactly what breaks: onnx 1.20.1 stamps
# ir_version 13, which onnxruntime 1.23.2 refuses to open. A frozen literal
# cannot drift with either package.
#
# ⚑ It is not an opaque blob: ``test_the_device_probe_graph_matches_its_generator``
# rebuilds it with ``onnx.helper`` and compares byte for byte, so the literal is
# pinned to a readable generator, and ``test_the_device_probe_graph_opens``
# proves the installed onnxruntime still loads it.
PROBE_MODEL_ONNX = bytes.fromhex(
    "08093a42 0a100a01 78120179 22084964 656e7469 7479120c 64657669 "
    "63655f70 726f6265 5a0f0a01 78120a0a 08080112 040a0208 01620f0a "
    "0179120a 0a080801 12040a02 08014204 0a001011 ",
)


def probe_onnx_device_providers(device: str) -> list[str]:
    """Providers a REAL session comes up with, measured on a throwaway graph.

    This is what makes the parse-time gate non-vacuous. There is no ORT API
    that answers "would the CUDA provider initialise here" without building a
    session, so the check builds the smallest possible one: 76 bytes, no model
    file, discarded immediately.

    ``get_providers()`` reports the providers that REGISTERED, which is a
    property of the runtime and the device rather than of the graph — so this
    tiny session answers the same question the 700MB one will, an hour of
    Stockfish earlier. The memory cap is deliberately NOT applied here: the
    probe's own arena is a few bytes, and reading the card's size to compute a
    limit would add a torch CUDA call to a function whose whole point is to be
    cheap and early. The providers it IS given are the REQUESTED device's, which
    is the part that has to be asserted rather than assumed — a probe pinned to
    CPU would pass everywhere and mean nothing.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        PROBE_MODEL_ONNX,
        providers=list(onnx_providers_for_device(device)),
    )
    try:
        return [str(p) for p in session.get_providers()]
    finally:
        del session


def validate_onnx_device(device: str) -> None:
    """PARSE-TIME gate on ``--onnx --device cuda...``, before any expensive work.

    Two checks, cheapest first, because they fail for different reasons:

    1. ``onnxruntime.get_available_providers()`` — the list the ORT wheel was
       **compiled** with. ⚑ This is a NECESSARY condition and nothing more. It
       is not a reading of what will initialise, and it must never be treated
       as one: observed on this box, ``get_available_providers()`` reports
       ``[Tensorrt, CUDA, CPU]`` while every ORT session seen here has come
       back ``['CPUExecutionProvider']``. Judged on that check alone the gate
       passes unconditionally, which is worth nothing.
    2. :func:`probe_onnx_device_providers` — a real 76-byte session on the
       requested device. THIS is the check that can fail. Measured here (under
       `nice 19` beside a live trainer, so noisy): the whole call is 150-240ms
       on its FIRST invocation, which is one-time onnxruntime initialisation,
       and 20-50ms thereafter. It runs once per ruler run and only when a CUDA
       device was asked for — `--device cpu` returns above without building
       anything (0.0003ms), and a `--checkpoint` run never reaches this
       function at all.

    Both run before the audit set is loaded and before Stockfish labels
    anything, which is the point: the cost this gate exists to avoid is an
    hour of labelling followed by an abort. The same reading is repeated on
    the real session by :func:`verify_onnx_session_device` — a probe is a
    strong predictor of the scoring session, not the scoring session itself.

    An indexed device (``cuda:1``) is not refused: providers carry ``device_id``
    (see :func:`onnx_providers_for_device`), so ORT lands on the index that was
    asked for instead of silently on card 0.
    """
    dev = str(device)
    if not dev.startswith("cuda"):
        return
    cuda_device_index(dev)  # reject `cuda:x` here, not from inside ORT
    import onnxruntime as ort

    available = list(ort.get_available_providers())
    if CUDA_PROVIDER not in available:
        raise SystemExit(
            f"--onnx --device {dev}: this onnxruntime has no {CUDA_PROVIDER} "
            f"(providers: {available}). ORT would silently fall back to CPU and the "
            "run would report itself as a CUDA run while taking hours. Install the "
            "GPU runtime or pass --device cpu deliberately."
        )
    active = probe_onnx_device_providers(dev)
    if CUDA_PROVIDER not in active:
        raise SystemExit(
            f"--onnx --device {dev}: {CUDA_PROVIDER} is in this onnxruntime's "
            f"compiled provider list but does NOT initialise here — a probe session "
            f"came up on {active}. ORT drops a provider it cannot start, with a "
            "warning, and runs the next one; the run would report a CUDA number "
            "produced on CPU. Fix the GPU runtime or pass --device cpu."
        )


def verify_onnx_session_device(
    model: object, device: str, *, gpu_mem_bytes: int | None = None, tag: str = "net-source",
) -> None:
    """THE gate: what did the session actually initialise, and is it capped?

    Reads ``session.get_providers()`` off the live session — the only reading
    that reflects what ORT will run on. If ``--device cuda...`` was asked for
    and CUDA is absent from that list, ORT has silently fallen back to CPU: the
    number would still be produced, the report would still say CUDA, and a
    BT4-sized net would take hours instead of minutes. That raises.

    On success it prints what was ACTUALLY applied, including the ORT memory
    cap — never a bare "GPU memory capped", because on this path the torch cap
    the rulers apply does not bound the ONNX session at all.
    """
    providers_fn = getattr(model, "session_providers", None)
    if providers_fn is None:  # pragma: no cover - only a non-OnnxChessNet gets here
        raise SystemExit(
            "the ONNX net does not expose its session providers, so the device it "
            "runs on cannot be verified; refusing to report a device on trust",
        )
    active = list(providers_fn())
    if str(device).startswith("cuda") and CUDA_PROVIDER not in active:
        raise SystemExit(
            f"--onnx --device {device}: the onnxruntime session came up on {active} "
            f"— {CUDA_PROVIDER} was requested and DROPPED (ORT warns and falls back "
            "rather than failing). The run would report a CUDA number produced on "
            "CPU. Fix the GPU runtime or pass --device cpu. ⚑ Do not diagnose this "
            "from get_available_providers(): that is the wheel's COMPILE-TIME list "
            "and it can name a provider that does not start."
        )
    if CUDA_PROVIDER not in active:
        # No CUDA arena exists, so neither "capped" nor "uncapped" describes it.
        note = "no GPU memory allocated"
    elif gpu_mem_bytes is not None:
        note = (
            f"CUDA arena capped at {gpu_mem_bytes} bytes "
            f"({gpu_mem_bytes / 1024 ** 3:.2f} GiB) via gpu_mem_limit"
        )
    else:
        note = "CUDA arena UNCAPPED (pass --gpu-mem-fraction to bound it)"
    print(f"[{tag}] onnxruntime session on {active}; {note}")


def apply_gpu_mem_cap(
    *, net: NetSource, device: str, gpu_mem_fraction: float | None, tag: str,
) -> None:
    """Apply ``--gpu-mem-fraction`` to the TORCH allocator and say only that.

    The one-line summary of the bug this replaces: the rulers printed "GPU
    memory capped at fraction F" after calling
    ``torch.cuda.set_per_process_memory_fraction``, which bounds the torch
    caching allocator and nothing else. On ``--onnx`` the net is not a torch
    module at all, so that message described a cap over an allocator the net
    never touched while the ONNX session ran unbounded. The ORT half of the cap
    is reported by :func:`verify_onnx_session_device`, at the session, after it
    exists.

    A fraction on ``--device cpu`` is reported as IGNORED rather than passed
    over in silence — an accepted-and-dropped flag is the same defect one size
    down.
    """
    if gpu_mem_fraction is None:
        return
    # Range-check even on the CPU path: a typo'd fraction is a typo whether or
    # not this particular run would have used it, and reporting it IGNORED
    # would tell the user the wrong thing about why.
    fraction = validate_gpu_mem_fraction(gpu_mem_fraction)
    if not str(device).startswith("cuda"):
        print(
            f"[{tag}] --gpu-mem-fraction {gpu_mem_fraction} IGNORED: --device "
            f"{device} allocates no GPU memory",
        )
        return
    import torch

    idx = cuda_device_index(device)
    torch.cuda.set_per_process_memory_fraction(fraction, device=idx)
    where = (
        "the ONNX session's CUDA arena is capped separately, at load time"
        if net.is_onnx
        else "this is the allocator the model runs in"
    )
    print(
        f"[{tag}] TORCH GPU allocator capped at fraction {gpu_mem_fraction} on "
        f"cuda:{idx} ({where})",
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

    def load(
        self,
        *,
        device: str,
        gpu_mem_fraction: float | None = None,
        tag: str = "net-source",
    ) -> torch.nn.Module:
        """Build the net in eval mode. Raises rather than returning a stand-in.

        ``gpu_mem_fraction`` is the ruler's ``--gpu-mem-fraction``. On the ONNX
        path it is converted to ORT's ``gpu_mem_limit`` and handed to the CUDA
        provider — the ONLY way it reaches the session. The torch-side cap the
        callers also apply does not bound ORT; see :func:`apply_gpu_mem_cap`.
        On the checkpoint path the model IS torch, so the caller's cap is the
        whole story and this argument is unused.
        """
        if self.onnx is not None:
            from chess_anti_engine.onnx.load import OnnxChessNet

            mem_bytes = onnx_cuda_mem_limit(device, gpu_mem_fraction)
            model = OnnxChessNet(
                self.onnx.path,
                input_name=self.onnx.input_name,
                policy_output_name=self.onnx.policy_output,
                wdl_output_name=self.onnx.wdl_output,
                providers=onnx_providers_for_device(device, gpu_mem_bytes=mem_bytes),
            )
            # After the session exists, not before: this is the only point at
            # which what ORT DID can be distinguished from what it was asked for.
            verify_onnx_session_device(
                model, device, gpu_mem_bytes=mem_bytes, tag=tag,
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
