"""One export/oracle/readout sequence; failures are not retried."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--plan', type=Path, required=True)
a = p.parse_args()
plan = json.loads(a.plan.read_text())
root = Path(plan['work_root'])
transport = str(Path(plan['author_root'])/'tools/ceres_cpu_oracle/transport.py')
out = root/'results'
subprocess.run([plan['python'], transport, 'export', '--plan', str(a.plan), '--out', str(out)], check=True)
subprocess.run([str(root/'sdk/dotnet'), str(root/'harness/bin/Release/net10.0/Oracle.dll'),
                str(out/'transport.json'), str(out/'oracle.json')], check=True)
subprocess.run([plan['python'], transport, 'compare', '--plan', str(a.plan), '--out', str(out)], check=True)
