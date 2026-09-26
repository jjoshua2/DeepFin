/* Generic bounded ASCII line polling only. No UCI keywords, chess state,
 * evaluator, move selection, process spawning or application scheduler here.
 * Single event-loop owner; a partial line never blocks a search step. */
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

static Term wire_packet(Env e, u32 kind, const char *text, size_t len) {
    if (cid_arity(CID_PACKET) != 2) err_fail("Packet ABI changed");
    Term string = io_str(e, text, len);
    Loc loc = heap_alloc(e, cls_fit(2));
    if (err_seen(e.mem)) err_fail("packet allocation failed");
    e.mem[loc] = kind;
    e.mem[loc + 1] = string;
    return term_ctr(CID_PACKET, loc);
}
static Term wire_poll_run(Env e, Term *f, IoWork *w) {
    (void)f; (void)w;
    static char line[4096];
    static size_t len;
    static int initialized, bad, ended;
    if (!initialized) {
        int flags = fcntl(STDIN_FILENO, F_GETFL);
        if (flags < 0 || fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK) < 0)
            err_fail("cannot make stdin nonblocking");
        initialized = 1;
    }
    fflush(stdout);
    if (ended) return wire_packet(e, 2, "", 0);
    for (unsigned budget = 0; budget < 8192; ++budget) {
        unsigned char c;
        ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) return wire_packet(e, 0, "", 0);
            err_fail("stdin read failed");
        }
        if (n == 0) {
            ended = 1;
            return wire_packet(e, len || bad ? 3 : 2, "", 0);
        }
        if (c == '\n') {
            Term packet = wire_packet(e, bad ? 3 : 1, bad ? "" : line, bad ? 0 : len);
            len = 0; bad = 0;
            return packet;
        }
        if ((c < 32 && c != '\t' && c != '\r') || c > 126 || len == sizeof(line)) bad = 1;
        if (!bad) line[len++] = (char)c;
    }
    return wire_packet(e, 0, "", 0);
}
static void __attribute__((constructor)) wire_use(void) {
    io_eff(CID_WIRE_POLL, wire_poll_run, 0);
}
