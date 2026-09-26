/* Private synchronous test transport, not UCI or a production NN interface.
 * F32 values are raw IEEE-754 words. Semantic checks remain in Bend. */
#include <errno.h>
#include <stdint.h>
#include <string.h>

static unsigned read_record(char tag[16], u32 *out, unsigned max, int eof_ok) {
    char line[4096];
    fflush(stdout);
    if (!fgets(line, sizeof(line), stdin)) {
        if (eof_ok && feof(stdin)) return UINT32_MAX;
        fputs("unexpected evaluator EOF\n", stderr); exit(2);
    }
    if (!strchr(line, '\n')) { fputs("unterminated transport record\n", stderr); exit(2); }
    char *token = strtok(line, " \t\r\n");
    if (!token || strlen(token) >= 16) { fputs("wrong transport record\n", stderr); exit(2); }
    strcpy(tag, token);
    unsigned n = 0;
    while ((token = strtok(NULL, " \t\r\n"))) {
        if (n == max || !*token || strlen(token) > 8 || strspn(token, "0123456789abcdefABCDEF") != strlen(token)) {
            fputs("invalid transport word\n", stderr); exit(2);
        }
        errno = 0;
        unsigned long long x = strtoull(token, NULL, 16);
        if (errno || x > UINT32_MAX) { fputs("transport word overflow\n", stderr); exit(2); }
        out[n++] = (u32)x;
    }
    return n;
}

static unsigned read_words(const char *expected, u32 *out, unsigned max, int eof_ok) {
    char tag[16] = {0};
    unsigned n = read_record(tag, out, max, eof_ok);
    if (n != UINT32_MAX && strcmp(expected, tag)) {
        fputs("wrong transport record\n", stderr); exit(2);
    }
    return n;
}

static Term command_read_run(Env e, Term *f, IoWork *w) {
    (void)f; (void)w;
    u32 x[4] = {0}, fields[7] = {0};
    char tag[16] = {0};
    unsigned n = read_record(tag, x, 4, 1);
    if (n == UINT32_MAX) {
        /* Clean EOF at the command boundary means quit, as before. */
    } else if (!strcmp(tag, "config")) {
        if (n != 4 || (!x[0] && (x[1] || x[2] || x[3])) ||
            (x[0] && (!x[1] || x[1] > 256 || !x[2] || x[2] > 4096 || !x[3] || x[3] > 32))) {
            fputs("invalid session config\n", stderr); exit(2);
        }
        for (unsigned i = 0; i < 4; i++) fields[i+2] = x[i];
    } else if (!strcmp(tag, "advance")) {
        if (n != 3 || !x[1]) { fputs("invalid advance command\n", stderr); exit(2); }
        fields[0] = 1; fields[1] = x[0]; fields[2] = x[1]; fields[6] = x[2];
    } else {
        fputs("wrong transport record\n", stderr); exit(2);
    }
    if (cid_arity(CID_COMMAND) != 7) err_fail("Command ABI changed");
    Loc loc = heap_alloc(e, cls_fit(7));
    if (err_seen(e.mem)) err_fail("command allocation failed");
    for (unsigned i = 0; i < 7; i++) e.mem[loc+i] = fields[i];
    return term_ctr(CID_COMMAND, loc);
}

static Term reply_read_run(Env e, Term *f, IoWork *w) {
    (void)f; (void)w;
    u32 x[264];
    unsigned n = read_words("reply", x, 264, 0);
    if (n < 8 || x[7] > 256 || n != 8 + x[7]) { fputs("invalid reply length\n", stderr); exit(2); }
    if (cid_arity(CID_SEARCH_REPLY) != 9) err_fail("Reply ABI changed");
    Term zero[1] = {0};
    Term policy = blk_new(e, false, 8, 1, 1, zero);
    if (err_seen(e.mem)) err_fail("policy allocation failed");
    for (unsigned i = 0; i < x[7]; i++) blk_write(e.mem, false, term_loc(policy), i, x[8+i]);
    Loc loc = heap_alloc(e, cls_fit(9));
    if (err_seen(e.mem)) err_fail("reply allocation failed");
    for (unsigned i = 0; i < 8; i++) e.mem[loc+i] = x[i];
    e.mem[loc+8] = policy;
    return term_ctr(CID_SEARCH_REPLY, loc);
}

static void __attribute__((constructor)) session_effects_use(void) {
    io_eff(CID_COMMAND_READ, command_read_run, 0);
    io_eff(CID_REPLY_READ, reply_read_run, 0);
}
