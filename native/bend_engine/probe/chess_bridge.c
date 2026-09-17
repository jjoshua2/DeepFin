/*
 * DeepFin CBoard bridge for the Bend architecture probe.
 *
 * This translation unit deliberately includes the existing pure-C CBoard
 * implementation directly. No Python or NumPy API participates in the native
 * probe. The Bend program receives only opaque U32 handles and policy actions.
 *
 * PR 1 uses CBoard's portable ray-slider fallback. It is a correctness /
 * integration probe, not a movegen performance claim. A later benchmark can
 * compile the production PEXT/magic macro set once the boundary is proven.
 */

#include <stdint.h>
#include <string.h>

#include "chess_anti_engine/encoding/_cboard_impl.h"

#define PROBE_MAX_BOARDS 128u
#define PROBE_INVALID UINT32_MAX

static CBoard g_probe_boards[PROBE_MAX_BOARDS];
static uint32_t g_probe_board_count = 0;
static int g_probe_initialized = 0;

static void probe_ensure_init(void) {
  if (g_probe_initialized) return;
  cboard_init_all();
  g_probe_initialized = 1;
}

static void probe_set_position(
  CBoard *b,
  uint64_t pawns,
  uint64_t knights,
  uint64_t bishops,
  uint64_t rooks,
  uint64_t queens,
  uint64_t kings,
  uint64_t white_occ,
  uint64_t black_occ,
  int turn,
  uint8_t castling,
  int ep_square
) {
  memset(b, 0, sizeof(*b));
  cboard_reset_hist_ep(b);
  b->bb[PAWN] = pawns;
  b->bb[KNIGHT] = knights;
  b->bb[BISHOP] = bishops;
  b->bb[ROOK] = rooks;
  b->bb[QUEEN] = queens;
  b->bb[KING] = kings;
  b->occ[WHITE_C] = white_occ;
  b->occ[BLACK_C] = black_occ;
  b->turn = (int8_t)turn;
  b->castling = castling;
  b->ep_square = cboard_sanitize_ep(ep_square);
  b->halfmove_clock = 0;
  b->hash_stack_len = 0;
  b->hist_len = 0;
  b->hist_head = 0;
  b->ply = 0;
  b->hash = cboard_compute_hash(b);
}

static int probe_make_fixture(uint32_t fixture, CBoard *out) {
  probe_ensure_init();

  switch (fixture) {
    case 0: /* standard start position */
      probe_set_position(
        out,
        UINT64_C(0x00ff00000000ff00),
        UINT64_C(0x4200000000000042),
        UINT64_C(0x2400000000000024),
        UINT64_C(0x8100000000000081),
        UINT64_C(0x0800000000000008),
        UINT64_C(0x1000000000000010),
        UINT64_C(0x000000000000ffff),
        UINT64_C(0xffff000000000000),
        WHITE_C,
        WK_CASTLE | WQ_CASTLE | BK_CASTLE | BQ_CASTLE,
        -1
      );
      return 1;

    case 1: /* open castling lanes for both sides */
      probe_set_position(
        out,
        0, 0, 0,
        UINT64_C(0x8100000000000081),
        0,
        UINT64_C(0x1000000000000010),
        UINT64_C(0x0000000000000091),
        UINT64_C(0x9100000000000000),
        WHITE_C,
        WK_CASTLE | WQ_CASTLE | BK_CASTLE | BQ_CASTLE,
        -1
      );
      return 1;

    case 2: /* legal en-passant: white e5 x d6 ep */
      probe_set_position(
        out,
        UINT64_C(0x0000001800000000),
        0, 0, 0, 0,
        UINT64_C(0x1000000000000010),
        UINT64_C(0x0000001000000010),
        UINT64_C(0x1000000800000000),
        WHITE_C,
        0,
        43
      );
      return 1;

    case 3: /* white pawn on a7 can promote */
      probe_set_position(
        out,
        UINT64_C(0x0001000000000000),
        0, 0, 0, 0,
        UINT64_C(0x1000000000000010),
        UINT64_C(0x0001000000000010),
        UINT64_C(0x1000000000000000),
        WHITE_C,
        0,
        -1
      );
      return 1;

    case 4: /* white king e1 is checked by a black rook on e8 */
      probe_set_position(
        out,
        0, 0, 0,
        UINT64_C(0x1000000000000000),
        0,
        UINT64_C(0x0100000000000010),
        UINT64_C(0x0000000000000010),
        UINT64_C(0x1100000000000000),
        WHITE_C,
        0,
        -1
      );
      return 1;

    default:
      return 0;
  }
}

static int probe_valid_handle(uint32_t handle) {
  return handle < g_probe_board_count;
}

static uint32_t probe_new_fixture(uint32_t fixture) {
  if (g_probe_board_count >= PROBE_MAX_BOARDS) return PROBE_INVALID;

  CBoard board;
  if (!probe_make_fixture(fixture, &board)) return PROBE_INVALID;

  uint32_t handle = g_probe_board_count++;
  g_probe_boards[handle] = board;
  return handle;
}

static uint32_t probe_legal_count(uint32_t handle) {
  if (!probe_valid_handle(handle)) return PROBE_INVALID;
  int moves[CBOARD_MAX_LEGAL_MOVES];
  int count = cboard_legal_move_indices(&g_probe_boards[handle], moves, 1);
  return (uint32_t)count;
}

static uint32_t probe_legal_at(uint32_t handle, uint32_t slot) {
  if (!probe_valid_handle(handle)) return PROBE_INVALID;
  int moves[CBOARD_MAX_LEGAL_MOVES];
  int count = cboard_legal_move_indices(&g_probe_boards[handle], moves, 1);
  if (slot >= (uint32_t)count) return PROBE_INVALID;
  return (uint32_t)moves[slot];
}

static uint32_t probe_push(uint32_t handle, uint32_t action) {
  if (!probe_valid_handle(handle) || action >= 4672u) return PROBE_INVALID;
  if (g_probe_board_count >= PROBE_MAX_BOARDS) return PROBE_INVALID;

  int legal[CBOARD_MAX_LEGAL_MOVES];
  int count = cboard_legal_move_indices(&g_probe_boards[handle], legal, 1);
  int found = 0;
  for (int i = 0; i < count; i++) {
    if ((uint32_t)legal[i] == action) {
      found = 1;
      break;
    }
  }
  if (!found) return PROBE_INVALID;

  uint32_t child = g_probe_board_count++;
  g_probe_boards[child] = g_probe_boards[handle];
  cboard_push_index(&g_probe_boards[child], (int)action);
  return child;
}

static uint32_t probe_in_check(uint32_t handle) {
  if (!probe_valid_handle(handle)) return PROBE_INVALID;
  return (uint32_t)(cboard_in_check(&g_probe_boards[handle]) != 0);
}

static uint32_t probe_hash32(uint32_t handle) {
  if (!probe_valid_handle(handle)) return PROBE_INVALID;
  return (uint32_t)(g_probe_boards[handle].hash & UINT64_C(0xffffffff));
}

uint32_t deepfin_bend_probe_legal_moves(
  uint32_t handle, uint32_t *out, uint32_t capacity
) {
  if (!probe_valid_handle(handle) || out == NULL || capacity == 0) return 0;
  int moves[CBOARD_MAX_LEGAL_MOVES];
  int count = cboard_legal_move_indices(&g_probe_boards[handle], moves, 1);
  uint32_t n = (uint32_t)count;
  if (n > capacity) n = capacity;
  for (uint32_t i = 0; i < n; i++) out[i] = (uint32_t)moves[i];
  return n;
}

uint32_t deepfin_bend_probe_call(uint32_t op, uint32_t a, uint32_t b) {
  switch (op) {
    case 0: return probe_new_fixture(a);
    case 1: return probe_legal_count(a);
    case 2: return probe_legal_at(a, b);
    case 3: return probe_push(a, b);
    case 4: return probe_in_check(a);
    case 5: return probe_hash32(a);
    default: return PROBE_INVALID;
  }
}
