/* External test oracle ONLY; not linked into deepfin-bend. */
#include "../legal_probe/support.h"
#include <inttypes.h>
#include <stdio.h>
static uint64_t table[LEGAL_TABLE_WORDS];
int main(void) {
    legal_make_tables(table);
    for (unsigned i = 0; i < 108160; ++i)
        printf("%" PRIu64 " %" PRIu64 "\n", table[i] >> 32, table[i] & UINT64_C(0xffffffff));
    return 0;
}
