# Using saved G10 adaptive-search values

Status: opt-in implementation and independent review passed; final repository validation passed. No real-corpus rewrite or training launch.

The saved 531,412-row diagnostic published in PR #632 found substantive value-target differences. Its apparent coverage gap was almost entirely complete single-legal-move d10 positions. This implementation reuses the existing corpus deriver to test the saved search evidence without additional Stockfish inference.

Use `--sf-value-selector g10-adaptive-final-or-d9-v1` only with uniform-d9, phase0 policy observation, latest-phase value observation and the search value scheme. The default remains unchanged. The existing explicit `--value-depth 12` full-width contract remains unchanged.

The selector reconstructs the recorded G10 rank-based gate and the exact preceding ranked top-8/top-4 move order. Complete single-move d10 is valid when the gate explicitly stopped for fewer than two moves. Valid extended observations use d12; other valid stops use d10. If a later observation fails these checks but the baseline is valid, preserve the original composite latest-phase d9 value and count the fallback reason. Existing registered policy-support exclusions run first and must match between control and candidate. Among retained rows, invalid baseline observations, incompatible experiment identity and malformed raw schema remain fatal.

Production uses ranked emissions, not a fresh sort of centipawn scores. Accordingly, complete unique ranked blocks with nonmonotone scores remain usable when the recorded rank-based gate and narrowing order agree. The earlier diagnostic conservatively excluded such anomalies; none were observed in its bank. This implementation does not silently reorder them.

The selector changes the search WDL target through the existing value-only path. A distinct value-source identity and selector metadata prevent accidental mixing with historical value targets. Tests exercise real one- and two-worker derivation, exact d9 fallback, row provenance, byte-identical non-value arrays and the actual mixed-value training-admission guard.

Before committing training compute, qualify the selector on a frozen common G10 cohort and compare both recipes on identical rows and schedules. Neural mixtures must preserve matched teacher coverage and fixed weights across the SF-anchor contrast. The original20M Ceres bank is not G10 coverage. This tool supplies a candidate target recipe; it establishes neither improved accuracy nor playing strength.

Independent review SHA256: `8aa815b96390cd8cb01f875757fee9096b0af41e2a9434af47c643bf9f706b38`. Review found and closed an ordering bug that intercepted existing policy-support exclusions before the candidate could preserve the control cohort. The regression covers real one- and two-worker derivation.

Validation: focused selector/observation/provenance tests and strengthened one/two-worker writer regressions passed. Ruff and Vulture passed. Whole-repository basedpyright passed with zero errors and warnings using the explicitly selected tested interpreter outside the sandbox; earlier sandbox runs failed interpreter discovery and are not counted as passes.
