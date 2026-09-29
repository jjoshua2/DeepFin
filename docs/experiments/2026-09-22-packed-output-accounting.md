# Bound output accounting during the storage comparison

The storage retry completed its external arm in 2,481.91 seconds but exhausted
its 5,392-second pair execution budget before starting NVMe. The traceback shows
post-arm source authentication inside an output-file stat. Each of 70,606 source
members called a guard that walked roughly 91,755 generated output/cache files.
Repeated accounting, rather than a measured storage comparison, consumed the
remaining time. The original failed run stays preserved.

This correction keeps STOP, deadlines, available RAM and disk-space floors on
every guard invocation. Source authentication still hashes every member/archive;
it no longer walks unrelated output files for each source member. Full output
accounting runs at phase boundaries and before success. While an owned child
writes, it runs at most once per 15 seconds measured from the previous scan's
end. A scan checks cheap guards throughout. The 10GiB cumulative output cap is
unchanged; as with any sampled resource guard, growth can occur between checks.

Twenty-one focused pair tests passed with two CPU threads on cores14/15 at
nice19. The new regressions demonstrate that 70,606 repeated calls cause only
one output walk before the interval expires, immediate STOP/RAM/deadline
failures remain active, periodic checks reject over-budget growth, and the final
check performs fresh exact accounting regardless of cadence. Scoped Ruff passes.
No frozen runtime, registered plan, queue or active training is changed.

This fix alone does not make another GPU retry ready: the first completed arm
already accounted for 8.219GB including prior/preparation bytes, and another
similar arm would exceed the current 10GiB cap. A later reviewed plan must resolve
that budget explicitly. No cap increase or automatic retry is included here.

Independent parent review approved the final diff and independently passed all
25 pair/probe tests. Scoped typechecking reported zero errors or warnings. A
bounded host check measured 200 cheap guard calls in 0.0544s, distinguishing
the retained checks from the removed repeated full-tree walks; this is not a
measurement of the corrected complete GPU benchmark.
