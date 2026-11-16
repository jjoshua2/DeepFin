"""Collect a prospective literal HOLDOUT cohort; never an accepted TRAIN bank.

No bank override exists. The separate public schema fixes the family, while the
real game/actor/history/raw/SF/TB/custody executor is shared with strict TRAIN.
"""
from pathlib import Path

from scripts import core_finite_collect as collection


def main() -> int:
    collection.require(Path(__file__).resolve() == Path(collection.__file__).resolve().with_name("core_finite_holdout_collect.py"),
                       "foreign HOLDOUT collector entrypoint")
    return collection.holdout_main()


if __name__ == "__main__":
    raise SystemExit(main())
