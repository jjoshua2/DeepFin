# Actual fast and prepare occurrence counts

The consumer checks full-Ply multiplicity of the actual Chess.filter_fast and
Chess.filter_prepare results. Duplicate candidate occurrences are retained in
the count; filter_fast also includes an arbitrary duplicate-containing tail.
Prepare starts its implementation's filter branch with an empty tail. Both
operations preserve the exact input table. These are count/table contracts,
without a fast/full equivalence or chess safety claim.

## Reproduce from a clean published checkout

Check out this PR's published head. No base-plus-source overlay is needed.
The qualifier requires clean tracked files and verifies every qualified source
against its HEAD Git blob. The entire HEAD delta from the fixed PR1018 base
must be limited to this directory's four published files. Every reused proof,
production dependency, and qualification support file remains pinned to the
exact base Git blob at 066fef8640770398daf6e27af804f2374856c066, whose tree is
c13eb9ba7d2fee34e36a530d49071c15f7467511.

For a shallow clone, first make that base object available:

    git fetch --depth=1 origin 066fef8640770398daf6e27af804f2374856c066

Supply Bun 1.4.2 and the verified Bend 2.0.21+U64 checkout at
aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae. Supply its 84-file checker manifest;
the required SHA256 fingerprint is
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
The qualifier verifies the compiler pin and all manifest file hashes before
running the checker. From the repository root:

    python3 -m native.bend_engine.standalone.proofs.legal_fast_count.qualify_fast_count \
      /path/to/pinned-checker \
      --checker-manifest /path/to/checker-tree.json \
      --report /outside/checkout/qualification-001.json \
      --evidence-dir /outside/checkout/checks-001

Both output paths must be fresh. Store run output outside the checkout.
The checker gets the first up to two CPUs in the caller's allowed affinity,
including when a scheduler or taskset restricts the caller to CPUs other than
0 and 1. The receipt and every checker command record that exact selected set.
The caller's original affinity is restored on exit.

Positive allowance: 86400 seconds. Each of the seven declaration-local semantic
controls gets 120 seconds. Checker runs retain the 6 GiB address-space cap,
16 MiB output cap, compiler/source identity checks and strict rejection classifier.
Parser, import, timeout, signal and resource failures do not qualify as semantic
control rejections.

The earlier base-overlay development mode remains supported and is explicitly
identified as such in its receipt. Published-head receipts instead record the
actual clean HEAD/tree and qualified source Git blobs. CI evidence must distinguish
the branch head from GitHub's synthetic PR merge commit; they are separate
identities even when their source trees match.
