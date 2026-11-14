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

/* Common gate: invalid positions are refused before Fathom sees them (its
 * search-time entry points assume a legal position), castling rights and
 * positions above the loaded tables are "unavailable" exactly like
 * python-chess's KeyError. */
static int df_gate(const DfTbPos *in, Pos *pos) {
    df_to_pos(in, pos);
    if (pos->ep >= 64u) return DF_TB_INVALID;
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
