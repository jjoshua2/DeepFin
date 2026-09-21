import {test, expect} from 'bun:test';
import {createHash} from 'node:crypto';
import {mkdtempSync, writeFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {contract} from './contract.js';

function fixture(changes={}, bytes='test-package') {
  const dir=mkdtempSync(join(tmpdir(),'bend-model-contract-'));
  const path=join(dir,'model.pt2');
  writeFileSync(path, bytes);
  const m={format:'deepfin-tuple-policy-wdl-checkpoint-v3',
    torch_version:'2.14.0+cpu', sha256:createHash('sha256').update(bytes).digest('hex'),
    checkpoint_sha256:'1'.repeat(64), weights_key:'model', channels:175, batch:4,
    policy_width:1858, row_independent:true, device:'cpu', device_index:0, dtype:'float32',
    input_history_encoding:'lc0_root_legacy_meta', input_extra_features:'v2_threats',
    history_rep_fix:true, ...changes};
  writeFileSync(join(dir,'model.json'),JSON.stringify(m));
  return {path,dir};
}
for (const [name,changes] of Object.entries({
  format:{format:'other'}, device:{device:'cuda'}, dtype:{dtype:'bfloat16'}, index:{device_index:1},
  channels:{channels:146}, policy:{policy_width:4672}, batch:{batch:3}, boolbatch:{batch:true},
  history:{input_history_encoding:'legacy'}, features:{input_extra_features:'v3'},
  fix:{history_rep_fix:false}, independent:{row_independent:false},
  packagehash:{sha256:'0'.repeat(64)}, checkpoint:{checkpoint_sha256:''},
  state:{weights_key:'anything'}, version:{torch_version:'2.14.0+cu130'},
})) test(`reject ${name}`,()=>{
  const f=fixture(changes);
  try {expect(()=>contract(f.path)).toThrow();} finally {rmSync(f.dir,{recursive:true});}
});
for (const history of ['lc0_root','lc0_root_legacy_meta'])
 for (const features of ['v1','v2_threats']) test(`bind ${history}/${features}`,()=>{
  const f=fixture({input_history_encoding:history,input_extra_features:features,channels:features==='v1'?146:175});
  try {
    const c=contract(f.path);
    expect(c.header).toContain('#define MODEL_CHANNELS '+c.manifest.channels);
    expect(c.header).toContain('#define MODEL_SHA256 "'+c.manifest.sha256+'"');
    expect(c.header).toContain('#define MODEL_LAYOUT '+(history==='lc0_root'?0:1));
    expect(c.header).toContain('#define MODEL_FEATURES '+(features==='v1'?0:1));
  } finally {rmSync(f.dir,{recursive:true});}
 });
test('modified package rejected',()=>{
 const f=fixture(); try {writeFileSync(f.path,'changed');expect(()=>contract(f.path)).toThrow();}
 finally {rmSync(f.dir,{recursive:true});}
});
