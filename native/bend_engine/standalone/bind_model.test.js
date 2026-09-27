import { describe, expect, test } from 'bun:test';
import { validateBinding } from './bind_model.js';
const hash = 'a'.repeat(64);
const baseline = () => ({format: 'deepfin-tuple-policy-wdl-checkpoint-v3', sha256: hash,
  checkpoint_sha256: 'b'.repeat(64), weights_key: 'model', batch: 1, policy_width: 1858,
  row_independent: true, device: 'cpu', device_index: 0, dtype: 'float32',
  history_rep_fix: true, input_history_encoding: 'lc0_root_legacy_meta',
  input_extra_features: 'v2_threats', channels: 175, torch_version: '2.14.0+cpu',
  resolved_model_config: {input_history_encoding:'lc0_root_legacy_meta', input_extra_features:'v2_threats',
    history_rep_fix:true, policy_encoding:'lc0_1858'}});
describe('strict standalone native model binding', () => {
  test('four supported profile encodings', () => {
    for (const [extra, c, offset] of [['v1',146,1],['v2_threats',175,3]]) {
      for (const [h, delta] of [['lc0_root',0],['lc0_root_legacy_meta',1]]) {
        const m=baseline(); m.input_extra_features=extra; m.channels=c; m.input_history_encoding=h;
        m.resolved_model_config.input_extra_features=extra; m.resolved_model_config.input_history_encoding=h;
        expect(validateBinding(m, hash).profile).toBe(offset+delta);
      }
    }
  });
  const invalid={format:'other', sha256:'c'.repeat(64), checkpoint_sha256:'x', weights_key:'random',
    batch:4, policy_width:4672, row_independent:false, device:'cuda', device_index:1, dtype:'bfloat16',
    history_rep_fix:false, input_history_encoding:'legacy', input_extra_features:'v3', channels:146,
    torch_version:'2.14.0+cu130'};
  for (const [k,v] of Object.entries(invalid)) test('reject '+k, () => {
    expect(() => validateBinding({...baseline(),[k]:v},hash)).toThrow();
  });
  for (const k of ['input_history_encoding','input_extra_features','history_rep_fix','policy_encoding']) {
    test('reject inconsistent resolved '+k,()=> {
      const m=baseline();m.resolved_model_config[k]='mismatch';expect(()=>validateBinding(m,hash)).toThrow();
    });
  }
  test('reject missing manifest/config and non-integer dimensions',()=>{
    for (const m of [null, [], {...baseline(),resolved_model_config:null}, {...baseline(),batch:true}])
      expect(()=>validateBinding(m,hash)).toThrow();
  });
});
