# Independent review: actual clear_lsb/popcount step

Result: **No findings** in ClearStep.bend or its importing consumer.

The review independently traced the proof bridges. Base defines Word.sub through fixed-width Word.adc; the checked Borrow.sub_one and Borrow.u64_sub lemmas connect that representation to actual U64.sub; Step.and_word connects actual U64 AND to Word.and; and Base defines U64.clear_lsb(a) as U64.and(a,U64.sub(a,U64.one())) and U64.popcount(a) through Word.count(64n,U64.to_word(a)). ClearStep composes those equalities with its structural count induction. The theorem is unconditional for every U64, with no added premise, axiom, hole, or unsafe shortcut.

The final pinned consumer check independently returned exactly All terms check. The final source receipt SHA-256 is 2729441f8fdbf389d7fd18868221edc2359464c0ade128fa811991117eda258f; all 56 source identities match the worktree. The ClearStep source identity is c6228e39b7d3c2a3059f1d61c7ad96e4e2fff885e6123623d76bcc05fa23dc8a, and the importing consumer identity is 84b3159aae3d15e79768a2b97d4fee5d68bdb83154a3014af7fa778697f1f064. The inherited destination receipt SHA-256 is 66a868fddd31d66b8b0a851621f21e32caff798804a80597e00339d5608e8e04.

The semantic control changes the importing consumer's clear_lsb expression to U64.clear_bit(a,1n). The pinned checker rejects that mutation at use_clear_lsb_popcount_step with the expected/observed count-step mismatch. This checks consumer-theorem connectivity; it does not mutate the Base implementation.

The reviewed scope is one general clear_lsb/popcount count-step lemma. It does not prove ctz membership or the full multi-bit Chess.bit_squares inventory.
