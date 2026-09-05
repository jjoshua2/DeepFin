# Development and production revisions

`main` is the complete development line: reusable tools, fixes and supported features
belong there. A live branch is a pinned deployment revision that can lag behind main
until the next experiment is ready to adopt changes. It is not a second development
line or the only home of production tools.

## Normal flow

Start feature and fix branches from current `main` and target their PRs at `main`.
A temporary stacked PR can target its dependency branch; the stack's destination is
still main. Keep run outputs separate from implementation, and register experiments
before using the tools to spend their compute budget.

A production emergency may need a patch in the live checkout first. Preserve that
patch and its test, then open the corresponding main-targeted PR in the same work
session. Link the main PR in the incident or experiment record so the fix cannot be
mistaken for complete while it exists only in production. Check for a newer equivalent
fix on main before porting it.

When a new experiment is ready, choose and record the main revision to deploy, check
config/schema and checkpoint compatibility, and use [operations](operations.md) for
graceful adoption. Keep the prior revision and recovery artifacts available. Merging
a feature into main is not a request to restart production or to switch its checkout.

## Keeping divergence visible

After fetching, compare published branches without touching a running checkout:

```bash
git rev-list --left-right --count origin/main...origin/ops/live-20260725
git log --right-only --cherry-pick --oneline origin/main...origin/ops/live-20260725
```

Replace the live ref with the actual deployment branch. The first count includes
merge commits; the second helps distinguish unported patches from commits already
represented by equivalent changes. Every reusable live-only change should have a main
PR or an explicit disposition. Deployment-only state belongs in a named configuration
or experiment record, not a hidden source-code fork.

For a reconciliation merge, preserve both ancestries and resolve conflicts by behavior:
newer main fixes must survive, and complete live toolchains must bring their dependencies
and tests. Use a merge commit when landing that reconciliation, rather than squashing
away the live parent. Then the published live revision becomes an ancestor of main;
production can remain on that revision until adoption is authorized.

## September 2026 reconciliation

The reconciliation starts from main `ac0dff588` and published live `d529afefc`.
The running checkout at `a38988814` is deliberately left in place. Main's newer search,
UCI, worker and server fixes are retained alongside the live corpus, replay, training
and match tools. The reviewed instruction and Grok workflows, G10 policy, varying-horizon instruments
and empty-batch relation fix are incorporated too. Four local diagnostic edits are
preserved without altering the original checkout: live-config authority in the control
driver, temperature-matched prior ranking, calibration controls and support-geometry
reporting. Their preservation does not claim a new experiment result.

`configs/pbt2_small.yaml` remains the supported main template; the published live
configuration is preserved at [snapshots/live_20260904.yaml](../configs/snapshots/live_20260904.yaml).
The portable snapshot contains a different supported model bundle and experiment
doses; repository paths are relative and its worker login must be supplied. Keeping it available does not silently promote every setting into a new run.

Research code, preregistrations and ledger readouts remain available. Generated binary
and log files newly present only on live remain recoverable from `d529afefc` and the
original artifact directories, rather than being copied into main's working tree.
See [toolchains](toolchains.md) for the supported entry points.
