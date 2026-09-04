---
name: independent-review
description: Review a proposed diff independently or reconcile review findings on a pull request, including comments, review bodies and inline threads.
---

# Independent review

Establish the base and candidate revisions and inspect the actual diff and relevant
source. Give an independent reviewer access to that evidence, not just the author's
explanation. Inherit the selected model unless there is a task-specific reason to
choose another. If independent review cannot be obtained, label the self-review.

Follow changed behavior to its consumers. For configuration and distributed work,
check that values reach the executing path and that the proposed observation can
fail when the effect is absent. Use focused checks that distinguish the failure;
there is no requirement to mutate every test or run every suite.

When reconciling a GitHub PR, retrieve all three review surfaces with pagination:

```bash
gh api --paginate repos/OWNER/REPO/issues/NUMBER/comments
gh api --paginate repos/OWNER/REPO/pulls/NUMBER/reviews
gh api --paginate repos/OWNER/REPO/pulls/NUMBER/comments
```

Read the bodies and inline findings. A blank `reviewDecision` does not mean unreviewed;
a template review body does not mean zero findings. Tie reviews, fixes and checks to
revisions. Distinguish whether review occurred, whether findings were addressed, and
whether validation still applies after base/head changes.

Report actionable findings with evidence and severity, or an explicit no-finding result
with coverage limits. Verify alleged fixes rather than trusting commit titles or
resolved-thread labels. Do not edit while acting as the independent reviewer. Posting,
triggering external reviewers or using a paid service requires authorization beyond
merely reading a PR; do not add those actions as a routine review gate.
