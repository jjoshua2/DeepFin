import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('binding',Path(__file__).with_name('bind.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class DonorTests(unittest.TestCase):
 def fixture(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name);ck=p/'checkpoint.pt';ck.write_bytes(b'full-state-checkpoint');s=p/'summary.json';c={'step_end':275896};s.write_text(json.dumps({'continuation':c,'sampling':{'complete':True}}));r=p/'complete.json';r.write_text(json.dumps({'status':'PASS_V50_CONTINUED_EPOCHS2_4','plan_sha256':'plan','summary_sha256':m.sha(s),'continuation':c,'checkpoints':[m.ref(ck)]}));terminal=p/'terminal.json';terminal.write_text('{"returncode":0}');return {'donor_terminal':str(terminal),'donor_complete':str(r),'donor_checkpoint':str(ck),'donor_plan_sha256':'plan','expected_donor_step':275896}
 def test_success_binds_actual_final_hash(self):
  p=self.fixture();self.assertEqual(m.successful_donor(p)['checkpoint']['sha256'],m.sha(p['donor_checkpoint']))
 def test_failed_process_rejected(self):
  p=self.fixture();Path(p['donor_terminal']).write_text('{"returncode":1}')
  with self.assertRaisesRegex(ValueError,'process failed'):m.successful_donor(p)
 def test_modified_checkpoint_rejected(self):
  p=self.fixture();Path(p['donor_checkpoint']).write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'hash differs'):m.successful_donor(p)
 def test_intermediate_is_not_final(self):
  p=self.fixture();p['donor_checkpoint']=str(Path(p['donor_checkpoint']).with_name('checkpoint_epoch1.pt'))
  with self.assertRaisesRegex(ValueError,'final donor checkpoint missing'):m.successful_donor(p)
if __name__=='__main__':unittest.main()
