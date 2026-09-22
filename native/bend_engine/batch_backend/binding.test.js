import { describe, expect, test } from 'bun:test';
import { bindingHeader, validateBinding, validateBatchBinding } from '../standalone/bind_model.js';
const hash = 'a'.repeat(64);
const baseline = () => ({format: 'deepfin-tuple-policy-wdl-checkpoint-v3', sha256: hash,
  checkpoint_sha256: 'b'.repeat(64), weights_key: 'model', batch: 4, policy_width: 1858,
  row_independent: true, device: 'cpu', device_index: 0, dtype: 'float32',
  history_rep_fix: true, input_history_encoding: 'lc0_root_legacy_meta',
  input_extra_features: 'v2_threats', channels: 175, torch_version: '2.14.0+cpu',
  resolved_model_config: {input_history_encoding:'lc0_root_legacy_meta', input_extra_features:'v2_threats',
    history_rep_fix:true, policy_encoding:'lc0_1858'}});
describe('explicit native batch binding', () => {
  for (const batch of [1,2,4,8,16]) test(`fixed batch ${batch}`, () => {
    const m = {...baseline(), batch};
    const bound = validateBatchBinding(m, hash);
    expect(bound.batch).toBe(batch);
    expect(bindingHeader(bound)).toContain(`#define DEEPFIN_MODEL_BATCH ${batch}\n`);
    expect(Object.isFrozen(bound)).toBe(true);
    if (batch > 1) expect(() => validateBinding(m, hash)).toThrow();
    else expect(bound).toEqual(validateBinding(m, hash));
  });
  for (const batch of [0,-1,3,32,1.5,true,'4',null,undefined]) test(`reject batch ${batch}`, () => {
    expect(() => validateBatchBinding({...baseline(),batch}, hash)).toThrow();
  });
  test('does not mutate the original manifest', () => {
    const m = baseline(), serialized = JSON.stringify(m);
    validateBatchBinding(m, hash);
    expect(JSON.stringify(m)).toBe(serialized);
  });
  test('four root input profiles', () => {
    for (const [extra,channels] of [['v1',146],['v2_threats',175]])
      for (const history of ['lc0_root','lc0_root_legacy_meta']) {
        const m = baseline();
        Object.assign(m, {input_extra_features:extra,channels,input_history_encoding:history});
        Object.assign(m.resolved_model_config, {input_extra_features:extra,input_history_encoding:history});
        expect(validateBatchBinding(m,hash).channels).toBe(channels);
      }
  });
  const invalid = {format:'other', sha256:'c'.repeat(64), checkpoint_sha256:'x', weights_key:'random',
    policy_width:4672, row_independent:false, device:'cuda', device_index:1, dtype:'bfloat16',
    history_rep_fix:false, input_history_encoding:'legacy', input_extra_features:'v3', channels:146,
    torch_version:'2.14.0+cu130',resolved_model_config:null};
  for (const [key,value] of Object.entries(invalid)) test(`reject ${key}`, () => {
    expect(() => validateBatchBinding({...baseline(),[key]:value}, hash)).toThrow();
  });
  test('reject malformed manifest and contradictory encoding', () => {
    for (const m of [null, [], undefined, 'x', {batch:4},
      {...baseline(),resolved_model_config:{...baseline().resolved_model_config,history_rep_fix:false}}])
      expect(() => validateBatchBinding(m,hash)).toThrow();
  });
});
