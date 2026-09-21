// Build-time packaging only. No Bun/JS is part of the running neural engine.
import {createHash} from 'node:crypto';
import {readFileSync, statSync} from 'node:fs';
import {resolve, extname} from 'node:path';

export function contract(packagePath) {
  const path = resolve(packagePath);
  if (extname(path) !== '.pt2') throw new Error('expected an immutable .pt2 package');
  const manifestPath = path.slice(0, -4) + '.json';
  const m = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (m === null || typeof m !== 'object' || Array.isArray(m)
      || m.format !== 'deepfin-tuple-policy-wdl-checkpoint-v3')
    throw new Error('expected the existing v3 checkpoint exporter manifest');
  const layout = ['lc0_root', 'lc0_root_legacy_meta'].indexOf(m.input_history_encoding);
  const features = ['v1', 'v2_threats'].indexOf(m.input_extra_features);
  if (layout < 0 || features < 0 || m.history_rep_fix !== true
      || m.channels !== (features === 0 ? 146 : 175) || m.policy_width !== 1858
      || ![1, 2, 4, 8, 16].includes(m.batch) || m.row_independent !== true
      || m.device !== 'cpu' || m.device_index !== 0 || m.dtype !== 'float32')
    throw new Error('unsupported encoding, tensor shape, or CPU/F32 contract');
  if (typeof m.torch_version !== 'string' || !/^\d+\.\d+\.\d+(\+cpu)?$/.test(m.torch_version))
    throw new Error('expected a stable CPU LibTorch export version');
  for (const key of ['sha256', 'checkpoint_sha256'])
    if (typeof m[key] !== 'string' || !/^[a-f0-9]{64}$/.test(m[key]))
      throw new Error('invalid package/checkpoint identity');
  if (!['model', 'swa_model'].includes(m.weights_key)) throw new Error('invalid selected weight state');
  const before = statSync(path, {bigint: true});
  const digest = createHash('sha256').update(readFileSync(path)).digest('hex');
  const after = statSync(path, {bigint: true});
  if (digest !== m.sha256 || before.ino !== after.ino || before.size !== after.size || before.mtimeNs !== after.mtimeNs)
    throw new Error('package SHA256 mismatch or package changed');
  return {manifest: m, header: [
    '#pragma once',
    `#define MODEL_LAYOUT ${layout}`,
    `#define MODEL_FEATURES ${features}`,
    `#define MODEL_CHANNELS ${m.channels}`,
    `#define MODEL_BATCH ${m.batch}`,
    `#define MODEL_SHA256 "${digest}"`,
    `#define MODEL_TORCH_VERSION "${m.torch_version.split('+')[0]}"`,
    '',
  ].join('\n')};
}
if (import.meta.main) {
  if (process.argv.length !== 3) throw new Error('usage: contract.js PACKAGE.pt2');
  process.stdout.write(contract(process.argv[2]).header);
}
