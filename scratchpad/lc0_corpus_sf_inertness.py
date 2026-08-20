"""Corpus-wide proof that arm F's floor term is inert on the lc0 control corpus.

`m_sf_policy_floor` is `masked_mean(sf_floor, sf_p0_regret_base)`
(`train/losses.py:1953`), and `masked_mean` clamps its denominator
(`losses.py:84`), so an empty mask contributes exactly 0.0 -- not NaN. The
inertness claim therefore reduces to: does ANY shard in the frozen corpus carry
an SF label array?

⚑ This exists because the claim was first made from ONE shard of 9,653. A
representative shard is not a corpus. This reads the array NAMES of every shard
via zarr group metadata -- directory listings, no row data -- so it is cheap
enough to have no excuse for not being run.

Read-only. No GPU, no Stockfish, no network.
"""

from __future__ import annotations

import collections
import glob
import json
import os

ROOTS = ("data/lc0_rows", "data/lc0_rows_heldout")
OUT = "scratchpad/lc0_corpus_sf_inertness.json"


def main() -> None:
    sets: collections.Counter[tuple[str, ...]] = collections.Counter()
    per_root: dict[str, int] = {}
    sf_bearing: list[str] = []

    for root in ROOTS:
        n = 0
        for shard in sorted(glob.glob(f"{root}/*/shard_*.zarr")):
            try:
                names = tuple(sorted(
                    d for d in os.listdir(shard)
                    if os.path.isdir(os.path.join(shard, d))
                ))
            except OSError:
                continue
            n += 1
            sets[names] += 1
            if any("sf" in name.lower() for name in names):
                sf_bearing.append(shard)
        per_root[root] = n

    total = sum(per_root.values())
    print(f"shards scanned: {total}  {per_root}")
    print(f"distinct array-name sets: {len(sets)}")
    for names, cnt in sets.most_common():
        sf = [n for n in names if "sf" in n.lower()]
        print(f"  {cnt:>6} shards | {len(names)} arrays | SF-related: {sf or 'NONE'}")
    print(f"\nshards carrying ANY sf* array: {len(sf_bearing)}")
    print("=> sf_p0_regret_base is EMPTY corpus-wide; the floor term contributes 0.0"
          if not sf_bearing else f"=> NOT INERT: {sf_bearing[:3]}")

    with open(OUT, "w") as fh:
        json.dump({
            "shards_scanned": total,
            "per_root": per_root,
            "distinct_name_sets": len(sets),
            "name_sets": [{"n_shards": c, "arrays": list(k)} for k, c in sets.most_common()],
            "shards_with_any_sf_array": len(sf_bearing),
        }, fh, indent=1)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
