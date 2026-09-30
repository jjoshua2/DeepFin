"""CPU-only payload-parent symlink qualification; never touches production data."""
import tempfile,json,importlib.util,sys
from pathlib import Path
import pytest
import zarr
from tests.test_bootstrap_factorial_targets import _cohort
from tests.test_ceres_derived_sidecar import setup,Session
from scripts import bootstrap_factorial_targets as targets
from chess_anti_engine.replay import target_overlay as storage
from chess_anti_engine.replay.shard import load_shard_arrays
spec=importlib.util.spec_from_file_location('prep','/home/josh/projects/chess/scratchpad/bt4_joint20/factorial58_20260919/preparation/run.py');prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
with tempfile.TemporaryDirectory(prefix='factorial-storage-fixture-') as name,pytest.MonkeyPatch.context() as mp:
 root=Path(name);m=_cohort(root,mp);external=root/'external';external.mkdir();legacy=root/'checkout';legacy.mkdir();(legacy/'outputs').symlink_to(external,target_is_directory=True)
 out=legacy/'outputs/cohort00';r=targets.build_cohort(m,out);prep.verify_cohort(m,out,{'rows':32,'shards':1})
 for arm in targets.ARMS:
  q=root/f'{arm}.json';storage.qualify_target_roots([out/arm],q);paths=storage.shard_paths(out/arm)
  storage.qualified_paths({'path':str(q),'sha256':targets.policy.shared.file_sha256(q)},paths)
  arrays,_=load_shard_arrays(paths[0],allow_target_overlay=True);assert len(arrays['x'])==32
 print('PASS_TARGET_PARENT_BUILD_RESUME_QUALIFY_CONSUME')
with tempfile.TemporaryDirectory(prefix='ceres-storage-fixture-') as name,pytest.MonkeyPatch.context() as mp:
 root=Path(name);a=setup(root,mp);tool=targets.policy.ceres
 Path(a.out).rmdir();external=root/'external';external.mkdir();parent=root/'bank';parent.symlink_to(external,target_is_directory=True)
 a.out=str(parent/'shard000000');Path(a.out).mkdir();a.retain_value2=True
 mp.setattr(tool,'open_teacher',lambda _:Session(root));tool.produce(a)
 shard=Path(a.out)/'shard_000000.zarr';binding=dict(zarr.open_group(str(shard),mode='r').attrs)['binding'];tool.verify_cached(shard,binding)
 mp.setattr(tool,'open_teacher',lambda _:pytest.fail('cache opened teacher'));tool.produce(a)
 assert (external/'shard000000/shard_000000.zarr').is_dir()
 print('PASS_CERES_REAL_PRODUCER_PARENT_SYMLINK_VERIFY_AND_CACHE')
# Exercise actual collection driver and Ceres completion discovery; fake only inference.
spec=importlib.util.spec_from_file_location('driver_test','/tmp/deepfin-ceres-audited-source/tests/test_ceres_collection_batches.py');dt=importlib.util.module_from_spec(spec);spec.loader.exec_module(dt)
with tempfile.TemporaryDirectory(prefix='ceres-driver-link-') as name:
 root=Path(name);external=root/'external';external.mkdir();(root/'outs').symlink_to(external,target_is_directory=True)
 worker=dt.write_script(root/'worker.py',dt.FAKE_CERES);p,h,_=dt.make_plan(root,[dt.ceres_chunk(root,'c16',worker,'ok')],minimum_free_gib=150,pause_between_chunks_seconds=0)
 assert dt.tool.main(['--plan',str(p),'--expected-plan-sha256',h,'--execute'])==0
 result=json.loads((root/'state/manifest.json').read_text());assert result['status']=='COMPLETE'
 assert (external/'c16').is_dir()
 print('PASS_DRIVER_PARENT_SYMLINK_VALIDATE_LAUNCH_DISCOVER_COMPLETE')
