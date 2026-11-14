/* Narrow, Python-free interface between _tb_ext.c and the vendored Fathom.
 *
 * Fathom's probing code is compiled in exactly one translation unit
 * (_tb_fathom.c, which #includes fathom/src/tbprobe.c unmodified) so that its
 * many file-scope names (WHITE, PAWN, popcount, ...) never meet the CBoard
 * headers _tb_ext.c includes. Everything crossing the boundary is plain
 * integers.
 *
 * Fathom's tablebase state is process-global; the callers in _tb_ext.c own the
 * locking (see the THREAD SAFETY note there). Nothing here locks.
 */
#ifndef DEEPFIN_TB_FATHOM_H
#define DEEPFIN_TB_FATHOM_H

#include <stdint.h>

typedef struct {
    uint64_t white, black;
    uint64_t kings, queens, rooks, bishops, knights, pawns;
    unsigned rule50;    /* halfmove clock */
    unsigned castling;  /* nonzero => not probeable (Syzygy has no castling) */
    unsigned ep;        /* en-passant square 0..63, 0 = none (a1 is never ep) */
    int turn;           /* 1 = white to move, 0 = black */
} DfTbPos;

/* One root move. ``dtz`` is the move's DTZ counted from the root, from the
 * mover's point of view, exactly as Fathom's root_probe_dtz computes it
 * before ranking; ``rank``/``score`` are root_probe_dtz's own outputs. */
typedef struct {
    int from_sq, to_sq;
    int promotes;  /* 0 none, 1 queen, 2 rook, 3 bishop, 4 knight (TB_PROMOTES_*) */
    int dtz;
    int rank;
    int score;
} DfTbRootMove;

#define DF_TB_MAX_ROOT_MOVES 193  /* TB_MAX_MOVES */

/* Probe outcome codes. */
#define DF_TB_OK 0
#define DF_TB_UNAVAILABLE 1   /* castling rights, too many men, or a missing table */
#define DF_TB_INVALID 2       /* not a legal position (Fathom's is_valid) */

/* (Re)initialise from a ':'-separated directory list. Returns 0 on success.
 * Caller must hold exclusive access (no probe in flight). */
int df_tb_init(const char *path);
/* Largest man count with a WDL table (Fathom's TB_LARGEST). */
unsigned df_tb_largest(void);
/* Release every mapping. Caller must hold exclusive access. */
void df_tb_free(void);

/* WDL from the side to move: -2 loss, -1 blessed loss, 0 draw, 1 cursed win,
 * 2 win. Assumes a reset 50-move counter (rule50 is ignored), the same
 * contract as python-chess ``Tablebase.probe_wdl``. */
int df_tb_probe_wdl(const DfTbPos *pos, int *out);
/* DTZ from the side to move in Fathom's probe_dtz convention (see
 * docs/tablebase_prober.md). Ignores rule50, like python-chess ``probe_dtz``. */
int df_tb_probe_dtz(const DfTbPos *pos, int *out);
/* Rank every legal root move with Fathom's root_probe_dtz (50-move aware via
 * pos->rule50 and use_rule50). *n_out receives the move count (0 at a
 * checkmate or stalemate root). */
int df_tb_probe_root(const DfTbPos *pos, int has_repeated, int use_rule50,
                     DfTbRootMove *out, int *n_out);

#endif
