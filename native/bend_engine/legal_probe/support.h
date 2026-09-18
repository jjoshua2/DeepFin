#ifndef DEEPFIN_BEND_LEGAL_SUPPORT_H
#define DEEPFIN_BEND_LEGAL_SUPPORT_H
#include <stdint.h>
/* Private wire format. Eight piece/color bitboards, then turn/rights/ep.
 * White=1, KQkq=1/2/4/8, ep=64 is absent. All input tokens are hexadecimal.
 * Tables: masks[128], offsets[128], knight[64], king[64], pawns[128], attacks. */
enum { LEGAL_TABLE_WORDS = 131072, LEGAL_TABLE_START = 512 };
typedef struct {
    uint64_t bb[8];
    uint32_t turn, rights, ep, depth, mode, seed;
} LegalInput;
int legal_read_input(LegalInput *input);
void legal_make_tables(uint64_t *out);
#endif
