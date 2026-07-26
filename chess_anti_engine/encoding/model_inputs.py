"""Encode boards in the encoding a specific MODEL was built for.

The raw encoders (:func:`encode_position`, :func:`encode_positions_batch`,
``encode_cboard``) take ``input_history_encoding`` / ``input_extra_features``
as OPTIONAL keywords that silently default to the legacy v1 layout. That
default is a trap for anything feeding a model: production runs
``lc0_root_legacy_meta`` + ``v2_threats`` (175 planes), and most of those
planes differ from the legacy encoding of the same board — 98 of 175 on the
board M11 measured, position-dependent but never small. The model cannot
detect the mismatch, so it produces confidently wrong numbers instead of an
error (docs/rl_loop_audit.md M11).

Every model built by :func:`chess_anti_engine.model.build_model` carries its
encoding as runtime attributes. Model-facing callers should encode through
this module, which reads those attributes and REFUSES to guess when they are
absent, rather than passing the keywords by hand at each call site.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import chess
    import numpy as np

_ENCODING_ATTRS = ("input_history_encoding", "input_extra_features")


def model_encoding_kwargs(model: Any) -> dict[str, str]:
    """Return ``{input_history_encoding, input_extra_features}`` for ``model``.

    Raises ``ValueError`` when either attribute is missing: a model whose
    encoding is unknown must not be fed a guessed default.
    """
    values: dict[str, str] = {}
    missing: list[str] = []
    for attr in _ENCODING_ATTRS:
        value = getattr(model, attr, None)
        if value is None:
            missing.append(attr)
        else:
            values[attr] = str(value)
    if missing:
        raise ValueError(
            f"model {type(model).__name__} does not declare {', '.join(missing)}; "
            "cannot encode inputs for it without guessing the layout. Models built "
            "by chess_anti_engine.model.build_model always declare both."
        )
    return values


def encode_position_for_model(
    model: Any, board: chess.Board, *, add_features: bool = True,
) -> np.ndarray:
    """:func:`encode_position` in ``model``'s own input encoding."""
    from chess_anti_engine.encoding.encode import encode_position

    enc = model_encoding_kwargs(model)
    return encode_position(
        board,
        add_features=add_features,
        input_history_encoding=enc["input_history_encoding"],
        input_extra_features=enc["input_extra_features"],
    )


def encode_positions_batch_for_model(
    model: Any, boards: Sequence[chess.Board], *, add_features: bool = True,
) -> np.ndarray:
    """:func:`encode_positions_batch` in ``model``'s own input encoding."""
    from chess_anti_engine.encoding.encode import encode_positions_batch

    enc = model_encoding_kwargs(model)
    return encode_positions_batch(
        list(boards),
        add_features=add_features,
        input_history_encoding=enc["input_history_encoding"],
        input_extra_features=enc["input_extra_features"],
    )


def encode_cboard_for_model(model: Any, cb: Any) -> np.ndarray:
    """``encode_cboard`` in ``model``'s own input encoding.

    Imported lazily: the CBoard encoder pulls in the ``_lc0_ext`` C extension,
    which pure-Python callers of this module should not have to have built.
    """
    from chess_anti_engine.encoding.cboard_encode import encode_cboard

    enc = model_encoding_kwargs(model)
    return encode_cboard(
        cb,
        input_history_encoding=enc["input_history_encoding"],
        input_extra_features=enc["input_extra_features"],
    )


def model_input_plane_count(model: Any) -> int:
    """Input channel count for ``model``'s declared extra-feature version."""
    from chess_anti_engine.encoding.encode import input_plane_count

    return input_plane_count(model_encoding_kwargs(model)["input_extra_features"])
