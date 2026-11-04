# Logical supply CPU readout

The distinct V3 CPU gate passed three methods on 2026-10-21 at 13:47 UTC.
Native elapsed time was 1.1504119 seconds. Independent evidence review accepted
the result with no findings. This is a scheduling and ownership result, not a
native-computation, model, memory-fit or performance qualification.

The controls exercised the actual dispatcher, driver and pool interfaces with
explicit fake computation and writer/ACK seams:

- FIFO/scatter dispatch returned batches of 320, 320 and 17 rows, with 640 owned
  logical row credits and refusal of excess credit.
- Five 64-lane pools supplied two disjoint modeled producer completions of 160
  rows. Both completions were deliberately ready before the owner advanced.
- Raw-writer receipt, owner ACK, refill and STOP behavior retained the modeled
  ownership sequence. The ACK seam was not the canonical durable checkpoint.

The driver control used one-byte synthetic trajectory reservations. It cannot
establish actual trajectory memory fit. Controlled completion and the test-only
fill wait do not establish real producer readiness or recommend a production
waiting policy. Five pools provide a 320-actor reservoir; 640 is the total game
quota or row-credit ceiling, not 640 simultaneously ready actors.

V1 and V2 remain failed attempts with zero method credit. V1 refused the
controller's canonical-path check before target qualification. V2 refused an
optional-import fallback before method qualification. V3 corrected the
controller behavior without widening the dependency allowlist or resource caps.
The complete raw streams, source versions, one-use records and reviews remain
in the retained local evidence capsule. No failed receipt was overwritten.

Evidence identities:

- V3 actual receipt: `1b1bb21762be2f4eaea08029203709c9b01df97a1b8594c76493062aaf4bfb7d`.
- Independent actual review: `84411c38148bc22d946fd2a254057098375b8aa2453ee7e4b18e85b66d00ef91`.
- Local 42-member evidence/contingency capsule:
  `1e72d291195e73964db575823a5c8f0be35a5c1135f857c7d74f5640670af775`.
  All member hashes were independently read back by the task owner.

The success/partial, startup-failure and timeout branches are prepared source
plans, not execution admission. No candidate resource hold remains from this
gate. Productive ordinary generation has priority through its existing owner;
any candidate GPU experiment needs an explicit coordinated yield and its own
current source, control, custody and resource qualification.

The next candidate gates are native two-producer conformance and an explicit
lightweight validation contract. Increasing actor supply does not itself remove
the previously measured outer validation overhead.
