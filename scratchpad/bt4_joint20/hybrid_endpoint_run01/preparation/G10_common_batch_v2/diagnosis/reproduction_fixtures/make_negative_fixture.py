from pathlib import Path
import chess
from tests.test_derive_corpus_targets import history_row,full_width_phase,ramp
from tests.test_derive_parallel import write_split_corpus
rows=[]
for i in range(2):
 row=history_row(game_id=i,result=1.)
 moves=sorted(m.uci() for m in chess.Board(row['fen']).legal_moves)
 row['phases']=[full_width_phase(row['fen'],{5:ramp(row['fen'],moves[0])})]
 rows.append(row)
p=write_split_corpus(Path('/tmp/g10-spawn-diagnosis-v1'),rows,[1,1]);print(p)
