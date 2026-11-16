"""C Syzygy tablebase prober (vendored Fathom) -- see docs/tablebase_prober.md.

Production callers: the UCI engine only -- its root DTZ shortcut
(``chess_anti_engine.uci.tb_root``) and its search-leaf / root-eval WDL probe
(``chess_anti_engine.uci.tb_leaf``). Every other caller (selfplay and arena
leaves, adjudication, training labels) still uses
``chess_anti_engine.tablebase`` (python-chess), which is also the test oracle
for this one.
"""
