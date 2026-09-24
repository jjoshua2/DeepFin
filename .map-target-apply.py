"""Temporary source preparation; never included in the feature diff."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

work = Path(sys.argv[1])
source = Path(__file__).parent
module = 'native/bend_engine/u64_map_probe/'


def replace(text, old, new, count=1):
    assert text.count(old) == count, old
    return text.replace(old, new)


new_files = {
    module+'cpu_target.py': '80db6b3851e2be9d5679465c9554ea547f6a1752e18b20fc0603bd15511007b7',
    'tests/test_bend_map_target.py': 'd9d86dc3db743f08f29dfcda5ae89011ba06273f77d291d7e0d6eb762ddae2ba',
}
for name, expected in new_files.items():
    data = (source/name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    assert not (work/name).exists()
    (work/name).write_bytes(data)

p = work/module/'run_probe.py'
s = p.read_text()
assert hashlib.sha256(s.encode()).hexdigest() == '174e8ab7b511080fc3eeab7eb278c45303ec8db0069a13b465fc44f8bf4ed3d4'
s = replace(s, 'from typing import NamedTuple\n', 'from typing import NamedTuple\n\nfrom . import cpu_target\n')
s = replace(s, "'native': ['-march=native']", 'cpu_target.TARGET_NAME: list(cpu_target.TARGET_FLAGS)')
s = replace(s, "        report['cc'] = command([args.cc, '--version'], 'cc')", "        report['cc'] = command([args.cc, '--version'], 'cc')\n        report['cpu_target'] = cpu_target.qualify(args.cc, output, command)\n        report['mode_flags'] = MODES")
s = replace(s, "('U64Map.bend', 'main.bend', 'run_probe.py')", "('U64Map.bend', 'main.bend', 'run_probe.py', 'cpu_target.py')")
p.write_text(s)

p = work/module/'benchmark.py'
s = p.read_text()
assert hashlib.sha256(s.encode()).hexdigest() == 'cceddf8c7b14e4b2c4697e2a735c4c53a6c30bd3e707bc6198db0877535a9d66'
s = replace(s, 'from typing import Any\n', 'from typing import Any\n\nfrom . import cpu_target\n')
s = replace(s, "        report['cc'] = command([args.cc, '--version'], 'cc')", "        report['cc'] = command([args.cc, '--version'], 'cc')\n        report['cpu_target'] = cpu_target.qualify(args.cc, output, command)")
s = replace(s, "('U64Map.bend', 'ScanMap.bend', 'benchmark.bend', 'benchmark.py')", "('U64Map.bend', 'ScanMap.bend', 'benchmark.bend', 'benchmark.py', 'cpu_target.py')")
s = replace(s, "('native', ['-march=native'])", '(cpu_target.TARGET_NAME, list(cpu_target.TARGET_FLAGS))', 2)
s = replace(s, "mode: str = 'native'", 'mode: str = cpu_target.TARGET_NAME')
s = replace(s, "for mode in ('native', 'ubsan'):", "for mode in (cpu_target.TARGET_NAME, 'ubsan'):")
p.write_text(s)

p = work/module/'README.md'
p.write_text(p.read_text()+'''\n\n## Explicit CPU target after the hosted feature-selection failure\n\nRun 36013240881 at 25599740 passed generic and portable cases, then the Clang 18\n`-march=native` build emitted an invalid AVX10 feature-combination diagnostic.\nThe harness rejected stderr despite compiler exit zero. It did not run the\naccelerated/UBSan cases or mutations in that attempt; the failed run remains\nfailed. Artifact 10814175152 has ZIP SHA-256\n`b6d0fb743fec7153d5c02874ac14f95b6179e349eccffce7643a739111e433c5`.\n\nBoth current map drivers now use the explicit `bmi2-popcnt` target:\n`-march=x86-64 -mpopcnt -mbmi2`. This deliberately replaces host-wide automatic\nfeature selection, rather than suppressing its warning or claiming all of that\nhost's features are covered. A baseline x86-64 executable first checks BMI2 and\nPOPCNT support. Unsupported/unknown hosts fail, not skip or silently downgrade.\nA second executable checks the target macros and performs PEXT/POPCNT on volatile\nhigh-bit data before map code is run. All compiler diagnostics and runtime errors\nstill fail their commands. The reports record target flags and capability results.\n\nThe original four correctness modes are now generic, forced-portable, explicit\nBMI2/POPCNT and UBSan. The comparison driver uses explicit BMI2/POPCNT and UBSan.\nThis is a test/benchmark build change, not a Bend/compiler/map implementation\nchange. Earlier performance records retain their original `-march=native` source\nand build identities. Their ratios are not measurements of this new target; no\nperformance panel was rerun or relabeled as part of this repair.\n\nCI also runs the benchmark's dictionary/driver checks without `--measure`, so\nboth map implementations remain covered without imposing a speed threshold.\n''')
paths = [*new_files, module+'run_probe.py', module+'benchmark.py', module+'README.md']
subprocess.run(['git', '-C', str(work), 'add', '--', *paths], check=True)
subprocess.run(['git', '-C', str(work), 'diff', '--cached', '--check'], check=True)
manifest = {name: hashlib.sha256((work/name).read_bytes()).hexdigest() for name in paths}
(Path(os.environ['RUNNER_TEMP'])/'target-sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
