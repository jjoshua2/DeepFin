"""Test-only observer around the unchanged actual private operation startup hook."""
import importlib.util
import json
import os
from pathlib import Path

p=Path('/home/josh/projects/chess/scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_common_batch_v1/python_bootstrap/sitecustomize.py')
spec=importlib.util.spec_from_file_location('qualified_private_hook',p)
hook=importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
import torch
from numcodecs import blosc
record={'pid':os.getpid(),'ppid':os.getppid(),'torch':torch.get_num_threads(),'blosc':blosc.get_nthreads(),
        'cmdline':Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),'actual_hook':str(p)}
fd=os.open('/tmp/g10-spawn-diagnosis-v1/positive_caps.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
os.write(fd,(json.dumps(record)+'\n').encode());os.close(fd)
if record['torch']!=2 or record['blosc']!=2:os._exit(78)
