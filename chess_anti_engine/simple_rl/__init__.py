"""A plain, resumable self-play RL loop: split openings, collect, train, evaluate.

Driven by ``scripts/simple_rl_loop.py``; see ``docs/simple_rl_loop.md``. Each
stage writes a ``manifest.json`` LAST, and a stage directory without one is
incomplete and is redone from scratch on the next invocation.
"""
