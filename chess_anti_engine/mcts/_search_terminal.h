/*
 * _search_terminal.h — the search's terminal-position decision, in one place.
 *
 * These four things (the solved lattice, its flip across a ply, the terminal
 * solved status, and the search-time terminal test) used to live in the middle
 * of _mcts_tree.c. They moved here unchanged when the check resolver needed
 * them, because the alternative was a second implementation of "is this node
 * terminal, and what is it worth" — and two copies of a rule this load-bearing
 * drift. The tree and the resolver now answer that question with the SAME code,
 * including the LC0-style 2-fold-as-draw convention, which is what makes the
 * resolver's perpetual-check termination agree with the tree's.
 *
 * Everything here is pure: no statics, no mutable state, no allocation. So
 * unlike an evaluator this header IS safe to include from more than one
 * extension — see the capsule rule in _value_provider.h for the distinction.
 */

#ifndef CAE_SEARCH_TERMINAL_H
#define CAE_SEARCH_TERMINAL_H

#include <stdint.h>

#include "_cboard_impl.h"

/* Solved status, from the perspective of the side to move at that node. */
#define SOLVED_UNKNOWN  0
#define SOLVED_WIN      1
#define SOLVED_LOSS    -1
#define SOLVED_DRAW     2

/* Flip a solved status across one ply (parent <-> child STM swap).
 *   parent sees a child WIN  → parent considers it LOSS-for-self
 *   parent sees a child LOSS → parent considers it WIN-for-self
 *   DRAW and UNKNOWN are unchanged. */
static inline int8_t solved_flip(int8_t s) {
    if (s == SOLVED_WIN) return SOLVED_LOSS;
    if (s == SOLVED_LOSS) return SOLVED_WIN;
    return s;  /* DRAW or UNKNOWN */
}

/* Solved status for a known-terminal CBoard. Caller must have already
 * confirmed cboard_is_game_over(b). Companion to cboard_terminal_value:
 * checkmate ⇒ STM lost; everything else terminal (stalemate / 50-move /
 * repetition / insufficient material) ⇒ DRAW. */
static inline int8_t cboard_terminal_solved_status(const CBoard *b) {
    return cboard_is_checkmate(b) ? SOLVED_LOSS : SOLVED_DRAW;
}

/* Search-time terminal detection. Returns 1 and writes terminal Q + solved
 * status if the position should be treated as terminal during MCTS search:
 *   - true game-over (checkmate / stalemate / 3-fold / 50-move / insufficient)
 *   - LC0-style 2-fold-as-draw: any prior occurrence inside the search tree
 *     means the side-to-move can force the third repetition, so the position
 *     is draw-or-better for whichever side prefers it. Treating 2-fold as a
 *     hard draw lets the search prune perpetual-check / shuffling lines
 *     immediately instead of waiting for the third visit.
 * Q for 2-fold is 0.0 (draw); cboard_terminal_value already returns 0.0 for
 * any non-game-over position, but we set it explicitly here for clarity. */
static inline int cboard_search_terminal(const CBoard *b,
                                          double *out_q, int8_t *out_solved) {
    if (cboard_is_game_over(b)) {
        *out_q = (double)cboard_terminal_value(b);
        *out_solved = cboard_terminal_solved_status(b);
        return 1;
    }
    if (cboard_is_repetition(b)) {
        *out_q = 0.0;
        *out_solved = SOLVED_DRAW;
        return 1;
    }
    return 0;
}

#endif /* CAE_SEARCH_TERMINAL_H */
