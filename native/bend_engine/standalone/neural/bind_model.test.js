import {test, expect} from 'bun:test';
import {mkdtempSync, writeFileSync, rmSync, symlinkSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {validate, inspect, header} from './bind_model.js';
const baseline = () => ({format: 'deepfin-tuple-policy-wdl-checkpoint-v3', sha256: 'a'.repeat(64),
  checkpoint_sha256: 'b'.repeat(64), weights_key: 'model', device: 'cpu', device_index: 0,
  dtype: 'float32', row_independent: true, history_rep_fix: true, policy_width: 1858,
  batch: 4, channels: 175, input_history_encoding: 'lc0_root_legacy_meta',
  input_extra_features: 'v2_threats', torch_version: '2.14.0+cpu', arch: {}, resolved_model_config: {}});
test('exact supported model generates a fixed header', () => {
  const m = baseline();
  expect(validate(m)).toBe(m);
  expect(header(m)).toContain('#define DF_MODEL_CHANNELS 175u');
  expect(header(m)).toContain('#define DF_TORCH_RELEASE "2.14.0"');
});
for (const [key, value] of Object.entries({format: 'old', sha256: 'wrong', checkpoint_sha256: null,
  weights_key: 'guess', device: 'cuda', device_index: 1, dtype: 'bfloat16', row_independent: false,
  history_rep_fix: false, policy_width: 4672, batch: true, channels: 146,
  input_history_encoding: 'legacy', input_extra_features: 'v3', torch_version: '2.14.0+cu130', arch: null})) {
  test('reject ' + key, () => expect(() => validate({...baseline(), [key]: value})).toThrow());
}
test('actual package identity and symlink checks', () => {
  const d = mkdtempSync(join(tmpdir(), 'bend-binding-'));
  try {
    const p = join(d, 'model.pt2'); const m = baseline();
    writeFileSync(p, 'package-fixture');
    m.sha256 = createHash('sha256').update('package-fixture').digest('hex');
    writeFileSync(join(d, 'model.json'), JSON.stringify(m));
    expect(inspect(p).sha256).toBe(m.sha256);
    writeFileSync(p, 'changed');
    expect(() => inspect(p)).toThrow('SHA256');
    symlinkSync(p, join(d, 'link.pt2'));
    expect(() => inspect(join(d, 'link.pt2'))).toThrow('non-symlink');
  } finally { rmSync(d, {recursive: true, force: true}); }
});
