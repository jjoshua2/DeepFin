from pathlib import Path
from tests.test_derive_corpus_targets import history_row
from tests.test_derive_parallel import write_split_corpus
print(write_split_corpus(Path('/tmp/g10-spawn-diagnosis-v1/positive'),[history_row(game_id=i,result=1.) for i in range(4)],[2,2]))
