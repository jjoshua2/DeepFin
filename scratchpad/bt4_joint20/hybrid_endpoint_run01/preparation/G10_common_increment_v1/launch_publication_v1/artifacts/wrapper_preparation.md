# Fixed incremental wrapper — prepared, not frozen or launched

`run_common.py` is a separate adaptation of the completed v3 wrapper. It admits
only the registered two sources and w00-00032..00063, keeping original sources,
value/policy selectors and v3 outputs unchanged. The actual command sequence is
**derive → snapshot → adapt → rank → qualify** for each source. Both selected-raw
consumers receive the same `--source-shards` path; there is no old prefix-only
`source_check`, extra preliminary census, target mixing, teacher inference or
training stage. The reusable selector performs its normal payload verification;
this wrapper adds selected raw metadata checks before/after the operation.

Actual support omissions come from the derivative ledger, then independently
validated raw rank observations. Independent raw no-result counts, rank eligibility,
injective full-history row joins, the emitted-reference bitmap, zero envelope drops
and exact cardinality establish the survivor complement. The copied qualifier
retains ordinary BT4/rank admission and unchanged-derived-storage checks. No exact
number of survivors is predeclared. Caps remain64 support/source and2% missing
results/source; other malformed rows remain fatal.

Freeze is intentionally unavailable without a separately qualified final checkout,
exact merged commit, interpreter and runtime receipt. The runtime receipt must have
`status:qualified`, matching `checkout`, `commit`, `python`, its existing transitive
`pins`, and `features.closed_shard_selection:true` plus
`features.support_exclusion_requires_result:true`. These attest qualified behavior
in pinned runtime files; they are not a request for new policy semantics. Freeze
pins registration, selections, original source manifests, closed receipts, selected
teacher attrs, metadata snapshots, wrapper/bootstrap, tool binaries and runtime.
`launch.json` is written exclusively. **No such file has been created.**

The eventual parent launch must use GPUhidden, numeric threads2, nice19/ionice3 and
CPUs2–3. The owned worker uses the same explicit environment and priority. A GNU
`timeout` in its own process group supplies the remaining7200-second whole-operation
allowance including30 seconds for termination, so coordinator loss does not remove
the deadline. The wrapper polls STOP/free-space every5 seconds and samples its
8GiB output/cache cap every60 seconds and at admission/stage boundaries/final
completion. The outer monitor also samples guards. Limits can be exceeded between
samples; they are not quotas. The150GiB reserve and parent/operation STOP paths
abort without retry; failure evidence and partial outputs remain.

Owned-group cleanup covers failed-stage descendants, including a descendant that
ignores TERM after its leader exits. No unrelated group is signaled. An exclusive
worker log/started receipt prevents adopting or repeating an attempt. Default
invocation does not run. Internal snapshot/qualify modes require the exact launch
plan hash; there is no arbitrary stage command interface.

`qualify_wrapper.py` used temporary metadata and owned CPU fixture subprocesses.
Seven cases passed in `wrapper_qualification_v2.json`: fixed-registration/value
selector checks, exclusion caps, independent omitted-row accounting, selected
metadata drift versus unrelated growth, actual worker command construction,
60-second size sampling, and real process-group cleanup preserving an unrelated
fixture. The earlier six-case receipt is retained as historical evidence. These
are wrapper checks, not an operational dataset qualification or runtime freeze.
