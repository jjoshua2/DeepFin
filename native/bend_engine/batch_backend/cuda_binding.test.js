import { describe, expect, test } from 'bun:test';
import { bindingHeader, validateBinding, validateBatchBinding, validateCudaBatchBinding } from '../standalone/bind_model.js';
const hash = 'a'.repeat(64);
const baseline = () => ({format: 'deepfin-tuple-policy-wdl-checkpoint-v3', sha256: hash,
  checkpoint_sha256: 'b'.repeat(64), weights_key: 'model', batch: 4, policy_width: 1858,
  row_independent: true, device: 'cuda', device_index: 0, dtype: 'bfloat16',
  history_rep_fix: true, input_history_encoding: 'lc0_root_legacy_meta',
  input_extra_features: 'v2_threats', channels: 175, torch_version: '2.14.0+cu130',
  resolved_model_config: {input_history_encoding:'lc0_root_legacy_meta', input_extra_features:'v2_threats',
    history_rep_fix:true, policy_encoding:'lc0_1858'}});
describe('explicit CUDA/BF16 backend binding', () => {
  for (const batch of [1,2,4,8,16]) for (const index of [0,1,127]) test(`batch ${batch} index ${index}`, () => {
    const m = {...baseline(),batch,device_index:index}, before = JSON.stringify(m);
    const b = validateCudaBatchBinding(m,hash);
    expect(Object.isFrozen(b)).toBe(true);
    expect(b).toMatchObject({device:'cuda',dtype:'bfloat16',batch,device_index:index,cuda_runtime_version:13000});
    const h = bindingHeader(b);
    expect(h).toContain('#define DEEPFIN_MODEL_CUDA 1\n');
    expect(h).toContain(`#define DEEPFIN_MODEL_DEVICE_INDEX ${index}\n`);
    expect(h).toContain('#define DEEPFIN_MODEL_CUDA_RUNTIME 13000\n');
    expect(JSON.stringify(m)).toBe(before);
    expect(() => validateBinding(m,hash)).toThrow();
    expect(() => validateBatchBinding(m,hash)).toThrow();
  });
  for (const index of [-1,128,true,1.5,'0',NaN,Infinity,null,undefined]) test(`bad index ${index}`, () => {
    expect(() => validateCudaBatchBinding({...baseline(),device_index:index},hash)).toThrow();
  });
  for (const version of ['2.14.0','2.14.0+cpu','2.14.0+cu13','2.14.0+cu130extra','2.14.0.dev1+cu130','2.14.0+cu000']) test(`bad version ${version}`, () => {
    expect(() => validateCudaBatchBinding({...baseline(),torch_version:version},hash)).toThrow();
  });
  test('CUDA minor version is preserved', () => {
    expect(validateCudaBatchBinding({...baseline(),torch_version:'2.14.0+cu128'},hash).cuda_runtime_version).toBe(12080);
  });
  const invalid = {format:'other',sha256:'c'.repeat(64),checkpoint_sha256:'x',batch:3,weights_key:'random',
    policy_width:4672,row_independent:false,device:'cpu',dtype:'float32',history_rep_fix:false,
    input_history_encoding:'legacy',input_extra_features:'v3',channels:146,resolved_model_config:null};
  for (const [key,value] of Object.entries(invalid)) test(`reject ${key}`, () => {
    expect(() => validateCudaBatchBinding({...baseline(),[key]:value},hash)).toThrow();
  });
  test('preserve CPU headers', () => {
    const m = {...baseline(),device:'cpu',dtype:'float32',torch_version:'2.14.0+cpu',batch:1};
    const h = bindingHeader(validateBinding(m,hash));
    expect(h).toContain('#define DEEPFIN_MODEL_CUDA 0\n');
    expect(h).toContain('#define DEEPFIN_MODEL_CUDA_RUNTIME 0\n');
    expect(() => validateCudaBatchBinding(m,hash)).toThrow();
  });
  test('reject malformed manifests and contradictory encoding', () => {
    for (const m of [null,undefined,[],1,'x',{},
      {...baseline(),resolved_model_config:{...baseline().resolved_model_config,history_rep_fix:false}}])
      expect(() => validateCudaBatchBinding(m,hash)).toThrow();
  });
});
