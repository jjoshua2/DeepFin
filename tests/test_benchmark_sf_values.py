"""Real UCI pipe tests for history-preserving scalar-versus-full-width cost rows."""
import hashlib,json
from pathlib import Path
import pytest
from scripts import benchmark_sf_values as tool

def test_worker_uses_history_fresh_tables_and_requested_widths(tmp_path):
    from tests.test_sf_policy_rewrite import raw_row
    row=raw_row(game_id=7)
    sample={'source':'frozen/shard.zst','source_row_index':42,'row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'row':row}
    data=tmp_path/'sample.jsonl';data.write_text(json.dumps(sample)+'\n')
    engine=tmp_path/'fake_sf'
    engine.write_text('''#!/usr/bin/python3
import sys,chess
board=chess.Board();width=1
for command in sys.stdin:
 command=command.strip()
 if command=='uci':print('id name ProtocolFixture\\nuciok',flush=True)
 elif command=='isready':print('readyok',flush=True)
 elif command.startswith('setoption name MultiPV value '):width=int(command.split()[-1])
 elif command.startswith('position fen '):
  value=command[len('position fen '):];fen,sep,moves=value.partition(' moves ');board=chess.Board(fen)
  if sep:
   for move in moves.split():board.push_uci(move)
 elif command.startswith('go depth '):
  depth=int(command.split()[-1]);moves=list(board.legal_moves)[:width]
  for rank,move in enumerate(moves,1):print(f'info depth {depth} multipv {rank} score cp {100-rank} nodes {rank*100} pv {move.uci()}',flush=True)
  print('bestmove '+moves[0].uci(),flush=True)
 elif command=='quit':break
''');engine.chmod(0o755)
    out=tmp_path/'out';out.mkdir()
    tool.worker({'rows':1,'sample':str(data),'out':str(out),'stockfish':str(engine),'syzygy_path':'','arms':[['root_d8',8,1],['root_d10',10,1],['all_d8',8,'all']]})
    records=[json.loads(x) for x in (out/'labels.jsonl').read_text().splitlines()]
    assert [r['arm'] for r in records]==['root_d8','root_d10','all_d8']
    assert [r['depth'] for r in records]==[8,10,8]
    assert records[0]['multipv']==records[1]['multipv']==1
    assert records[2]['multipv']>1
    for r in records:
        assert 'ucinewgame' in r['commands']
        assert r['position_command']==r['commands'][r['commands'].index(r['position_command'])]
        assert row['history_root_fen'] in r['position_command']
        assert ' '.join(row['history_uci']) in r['position_command']
        assert r['row_sha256']==sample['row_sha256']
        assert len(r['pv'])==r['multipv']
