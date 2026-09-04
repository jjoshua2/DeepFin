---
name: experiment-readout
description: Analyze banked experiment data, preregister a compute-backed comparison, or record a research readout. Use for experiment decisions, not ordinary code fixes.
---

# Experiment readout

Start from the project's experiment record and available artifacts. Find the closest
prior hypothesis, controls, verdicts and unresolved confounds. Read relevant sections,
not entire historical logs. Identify what the user wants decided and what evidence
would distinguish the competing explanations.

Prefer analysis of banked observations before new compute. Check their provenance,
sampling unit and comparability: code/build, settings, checkpoint, data version and
measurement definition. Preserve source-qualified identities when joining or clustering.
A missing stamp is unknown, not proof that two runs match.

For a new comparison, preregister the baseline and a simpler control where useful,
the deciding metric and threshold, uncertainty method, compute budget and stopping
rule. Choose a diagnostic that can distinguish a plausible failure mechanism when its
cost is justified. Distinguish cheap screening from evidence of the final objective;
consult the project's evaluation protocol for appropriate stages. Explain any exception
to a usual screen before seeing the result. Do not turn a fixed seed count or arena size
from an old experiment into a universal gate.

Bank observations at the lowest useful level, with settings and cluster keys. Correct
estimators against the same observations instead of rerolling the intervention.
Scale only within the authorized scope and budget; identify what new uncertainty the
next stage resolves before spending more compute.

Record the readout, uncertainty, provenance, confounds and next decision against the
precommitted rule. A failed or unfinished job is not a clean negative result. Separate
claims about the current checkpoint/horizon from claims about the general idea.
