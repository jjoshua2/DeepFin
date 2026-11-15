"""C Syzygy tablebase prober (vendored Fathom) -- see docs/tablebase_prober.md.

One production caller: the UCI engine's root DTZ shortcut
(``chess_anti_engine.uci.tb_root``). Every other caller (search leaves,
selfplay, training) still uses ``chess_anti_engine.tablebase`` (python-chess),
which is also the test oracle for this one.
"""
