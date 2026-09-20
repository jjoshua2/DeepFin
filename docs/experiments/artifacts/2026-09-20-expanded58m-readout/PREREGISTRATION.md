# Completed 58M continuation against its immediate 50M predecessor

Status: STAGED; not queued or launched.

Question: did the completed 58M continuation improve playing strength over the exact 50M checkpoint it resumed? This contrast changes both training exposure and data coverage. It cannot isolate the contribution of new positions.

Candidate: runs/expanded58m_pressure_v1/checkpoint.pt. Reference: runs/expanded50548069_v50_from35m_epoch4_seed105_recovery_v3/checkpoint.pt. The 58M completion receipt binds the reference as its starting checkpoint with SHA 43786945252e746a6e646a21663f798e1dff224e8c48c6dac757d8398a4ee015. Candidate SHA c3259d8d8237271e83e2f7ae1250cc60c074a1c50e8071dad7475eecadc6a92b.

One fixed 256-game match (128 opening pairs) at 400 simulations, fresh seed 2026092001. Reuse the validated epoch1 readout runner and frozen arena runtime 82298a5d4010e7097f473712e3720de229522427. Keep the same opening book and 16-ply starts, matched simulations, training search shape, candidate/reference policy prior temperature 1.0, play temperature 0.1, maximum 300 plies, compile enabled, 128 concurrent games, 4096 evaluation batch, and Syzygy disabled. The maximum arena runtime is 2400s, runner 2450s, outer 2490s. No automatic sequential extension.

Report the candidate score, Elo point estimate and nominal 95% opening-paired interval computed from the 128-pair pentanomial, together with full game bank and truncation status. A positive estimate is useful development evidence even if its interval crosses zero; do not require confidence exclusion to justify discussing it. Report invalid/incomplete execution separately from a negative result. No teacher recipe, training continuation or promotion is authorized by this match alone.

External output: /home/josh/chess-artifacts/runs/expanded58m_vs50m_20260920. Controls, model/summary/terminal pins and queue proposal are alongside this document. Proposed position: after the reviewed BT4 batch benchmark and before final factorial preparation, only while preparation remains queued. CPU preparation sidecar can continue during the arena; GPU ownership remains serialized by the existing lease.

The opening book exceeds the operator's8MiB direct-pin limit. Its SHA remains in the pinned plan and the unchanged runner hashes it before launch; only the redundant direct descriptor pin is omitted.
