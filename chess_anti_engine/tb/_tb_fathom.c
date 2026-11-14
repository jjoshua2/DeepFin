/* The one translation unit that compiles the vendored Fathom prober.
 *
 * fathom/src/tbprobe.c is included UNMODIFIED (it in turn includes tbchess.c
 * and stdendian.h); including it here rather than compiling it as its own
 * source gives this file access to Fathom's static probe_wdl/probe_dtz, which
 * its public header does not export, without patching the vendored code.
 * Provenance, licence and file hashes: fathom/VENDORED.md.
 *
 * No locking here: _tb_ext.c serialises init against probes and root probes
 * against each other. See _tb_fathom.h.
 */
#include "_tb_fathom.h"

#include <stdlib.h>

/* Fathom's own sources trip a few of the project's baseline warnings
 * (-Wshadow, -Wsign-compare, ...). They are upstream code we do not edit, so
 * the warnings are silenced for the included file only. */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
#include "fathom/src/tbprobe.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static void df_to_pos(const DfTbPos *in, Pos *out) {
    out->white = in->white;
    out->black = in->black;
    out->kings = in->kings;
    out->queens = in->queens;
    out->rooks = in->rooks;
    out->bishops = in->bishops;
    out->knights = in->knights;
    out->pawns = in->pawns;
    out->rule50 = (uint8_t)(in->rule50 > 255u ? 255u : in->rule50);
    out->ep = (uint8_t)in->ep;
    out->turn = in->turn ? true : false;
}

/* En-passant square handed to Fathom: 0 = none, -1 = refuse the position.
 *
 * ⚑ Fathom trusts pos->ep. Its ep capture removes whatever stands on ep∓8, so a
 * bogus square can delete a KING (a segfault inside the probe) or invent a
 * capture that changes the DTZ. Three cases, mirroring python-chess:
 *
 *  - no side-to-move pawn can capture onto the square (python-chess's
 *    generate_pseudo_legal_ep: ep empty, a capturer on rank 5/4 attacking it):
 *    python-chess's probe ignores the square, so it is dropped here too. This
 *    is the normal "double push with no capturer" case, and also any
 *    impossible square nobody can capture onto;
 *  - a capturer exists and the square is structurally possible
 *    (python-chess's Board._valid_ep_square: rank 6/3, an enemy pawn on ep∓8,
 *    ep±8 empty): a real ep capture, passed through;
 *  - a capturer exists but the square is impossible: python-chess would apply
 *    a nonsense capture (removing a knight, or the king, from ep∓8). Neither
 *    answer means anything, so the position is refused as illegal -- the same
 *    verdict as python-chess's is_valid() (STATUS_INVALID_EP_SQUARE). */
static int df_ep_square(const DfTbPos *in) {
    unsigned ep = in->ep;
    if (ep == 0) return 0;
    if (ep >= 64u) return -1;
    const uint64_t us = in->turn ? in->white : in->black;
    const uint64_t them = in->turn ? in->black : in->white;
    const uint64_t occ = in->white | in->black;
    const unsigned file = ep % 8;
    if (occ & (1ULL << ep)) return 0;  /* python-chess: no ep move at all */
    /* Squares from which a side-to-move pawn attacks ep, on its capture rank. */
    const uint64_t capture_rank = in->turn ? 0x000000FF00000000ULL : 0x00000000FF000000ULL;
    uint64_t from = 0;
    if (in->turn) {
        if (file > 0 && ep >= 9) from |= 1ULL << (ep - 9);
        if (file < 7 && ep >= 7) from |= 1ULL << (ep - 7);
    } else {
        if (file > 0 && ep + 7 < 64) from |= 1ULL << (ep + 7);
        if (file < 7 && ep + 9 < 64) from |= 1ULL << (ep + 9);
    }
    if (!(from & capture_rank & in->pawns & us)) return 0;
    const unsigned rank = ep / 8;
    if (rank != (in->turn ? 5u : 2u)) return -1;
    const unsigned pushed = in->turn ? ep - 8 : ep + 8;   /* the double-pushed pawn */
    const unsigned origin = in->turn ? ep + 8 : ep - 8;   /* where it came from */
    if (!(in->pawns & them & (1ULL << pushed))) return -1;
    if (occ & (1ULL << origin)) return -1;
    return (int)ep;
}

/* Common gate: invalid positions are refused before Fathom sees them (its
 * search-time entry points assume a legal position), castling rights and
 * positions above the loaded tables are "unavailable" exactly like
 * python-chess's KeyError. */
static int df_gate(const DfTbPos *in, Pos *pos) {
    df_to_pos(in, pos);
    int ep = df_ep_square(in);
    if (ep < 0) return DF_TB_INVALID;
    pos->ep = (uint8_t)ep;
    if (!is_valid(pos)) return DF_TB_INVALID;
    if (in->castling != 0) return DF_TB_UNAVAILABLE;
    if ((unsigned)popcount(pos->white | pos->black) > TB_LARGEST) return DF_TB_UNAVAILABLE;
    return DF_TB_OK;
}

int df_tb_init(const char *path) {
    return tb_init(path) ? 0 : -1;
}

unsigned df_tb_largest(void) {
    return TB_LARGEST;
}

void df_tb_free(void) {
    tb_free();
}

int df_tb_probe_wdl(const DfTbPos *in, int *out) {
    Pos pos;
    int gate = df_gate(in, &pos);
    if (gate != DF_TB_OK) return gate;
    int success = 0;
    int v = probe_wdl(&pos, &success);
    if (success == 0) return DF_TB_UNAVAILABLE;
    *out = v;
    return DF_TB_OK;
}

int df_tb_probe_dtz(const DfTbPos *in, int *out) {
    Pos pos;
    int gate = df_gate(in, &pos);
    if (gate != DF_TB_OK) return gate;
    int success = 0;
    int v = probe_dtz(&pos, &success);
    if (success == 0) return DF_TB_UNAVAILABLE;
    *out = v;
    return DF_TB_OK;
}

int df_tb_probe_root(const DfTbPos *in, int has_repeated, int use_rule50,
                     DfTbRootMove *out, int *n_out) {
    Pos pos;
    int gate = df_gate(in, &pos);
    if (gate != DF_TB_OK) return gate;
    *n_out = 0;

    /* Per-move DTZ from tb_probe_root's results array: |dtz| plus a WDL whose
     * side of 2 gives the sign. Ranks and scores from tb_probe_root_dtz. Both
     * are Fathom's own computations over gen_legal/gen_moves; they are joined
     * by move below, never recomputed here. */
    unsigned results[TB_MAX_MOVES];
    unsigned res = tb_probe_root_impl(pos.white, pos.black, pos.kings, pos.queens,
                                      pos.rooks, pos.bishops, pos.knights, pos.pawns,
                                      pos.rule50, pos.ep, pos.turn, results);
    if (res == TB_RESULT_FAILED) return DF_TB_UNAVAILABLE;

    struct TbRootMoves *rm = (struct TbRootMoves *)malloc(sizeof(struct TbRootMoves));
    if (rm == NULL) return -1;
    if (!root_probe_dtz(&pos, has_repeated != 0, use_rule50 != 0, rm)) {
        free(rm);
        return DF_TB_UNAVAILABLE;
    }
    unsigned n_results = 0;
    while (n_results < TB_MAX_MOVES && results[n_results] != TB_RESULT_FAILED) n_results++;
    if (rm->size != n_results || rm->size > DF_TB_MAX_ROOT_MOVES) {
        free(rm);
        return DF_TB_UNAVAILABLE;
    }
    for (unsigned i = 0; i < rm->size; i++) {
        TbMove mv = rm->moves[i].move;
        unsigned from = TB_MOVE_FROM(mv), to = TB_MOVE_TO(mv), promo = TB_MOVE_PROMOTES(mv);
        int found = -1;
        for (unsigned j = 0; j < n_results; j++) {
            unsigned r = results[j];
            if (TB_GET_FROM(r) == from && TB_GET_TO(r) == to && TB_GET_PROMOTES(r) == promo) {
                found = (int)j;
                break;
            }
        }
        if (found < 0) {
            free(rm);
            return DF_TB_UNAVAILABLE;
        }
        unsigned r = results[found];
        int mag = (int)TB_GET_DTZ(r);
        unsigned wdl = TB_GET_WDL(r);
        out[i].from_sq = (int)from;
        out[i].to_sq = (int)to;
        out[i].promotes = (int)promo;
        out[i].dtz = wdl > TB_DRAW ? mag : (wdl < TB_DRAW ? -mag : 0);
        out[i].rank = rm->moves[i].tbRank;
        out[i].score = rm->moves[i].tbScore;
    }
    *n_out = (int)rm->size;
    free(rm);
    return DF_TB_OK;
}
