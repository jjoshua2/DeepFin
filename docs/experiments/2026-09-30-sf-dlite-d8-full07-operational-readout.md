# D-lite d8 labeling: full06 telemetry stop and full07 continuation

The full06 CPU labeling attempt stopped after 1,052.312 seconds when the shared cgroup physical I/O meter saw its device set change. The launcher failed closed before its 100 GiB I/O, 24 GiB aggregate PSS, 30 GiB output, or four-hour wall limits were reached. It left 235 fsynced label blocks covering 374,577 selected rows. Those blocks remain source evidence only: full06 has zero label, target, training, and strength credit.

The meter compared an exact device set at every sample. A device was present on a later read with 4,096 write bytes, consistent with a newly appearing cgroup device; full06 did not bank its baseline device map, so the exact transition cannot be reconstructed. The successor meter charges every new device's full read and write counters from zero while still stopping on a disappeared device or any counter reversal. It banks the baseline and each sampled per-device map. A CPU fixture checked additive charging and both refusal cases.

Full07 is a fresh 2.5-million-row six-worker attempt. It uses the same selected roster, fixed-depth-8 Stockfish profile, [prospectively declared first complete non-bound depth-8 scalar target](https://github.com/jjoshua2/DeepFin/pull/968), calibration, and unchanged policy/target mixing. It adopts no failed full06 labels. Each completed block is fsynced and source-bound; an active unsealed block has a 600-second hard limit. A restart under the identical full07 worker, authorization, and selection identity may reuse sealed blocks only after validation. Full07 remains unadmitted until its completed launch terminal and independent all-label audit session pass. No model training or strength claim follows from a running label job.

| Evidence | SHA-256 |
| --- | --- |
| full06 failed launcher receipt | `e8c870eaacf8abd6a939fef39f463afabb7e0a2fbf9f4d47b3e92c9e72129538` |
| full07 root authorization | `b4806340c7f3a5a61c8eedd0ea4156f99928b494274780ae7877fb072c484dc1` |
| unchanged full07 scientific worker | `31a7730cac1f9b6b39f01b938ccf6772c895d5a2727f89eb5a969aacec24cf83` |
| full07 additive-I/O launcher | `71f14583a2f505eb435f28a410838bcab93f5bb4862948559aa794115e78ac3f` |
| full07 detached launch receipt | `d23c1b9e249d040fe925e8f4798d1329058f6ac0b3ea6158ac209f7716edb61d` |
| full07 launch start receipt | `dc8d1822180774dc10a97ccb04ddc443ad25931f4d8fea7ec18e10e201409deb` |
