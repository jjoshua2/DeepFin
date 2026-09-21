// Build-time package binding. Never deployed as an engine controller/interpreter.
import {createHash} from 'node:crypto';
import {lstatSync, readFileSync, writeFileSync} from 'node:fs';
import {resolve, dirname, basename, extname, join} from 'node:path';

const sha = x => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const integer = (x, choices) => Number.isSafeInteger(x) && choices.includes(x);
export function validate(m) {
  if (!m || typeof m !== 'object' || Array.isArray(m)
      || m.format !== 'deepfin-tuple-policy-wdl-checkpoint-v3'
      || !sha(m.sha256) || !sha(m.checkpoint_sha256)
      || !['model', 'swa_model'].includes(m.weights_key)
      || m.device !== 'cpu' || m.device_index !== 0 || m.dtype !== 'float32'
      || m.row_independent !== true || m.history_rep_fix !== true
      || m.policy_width !== 1858 || !integer(m.batch, [1, 2, 4, 8, 16])
      || !['lc0_root', 'lc0_root_legacy_meta'].includes(m.input_history_encoding)
      || !['v1', 'v2_threats'].includes(m.input_extra_features)
      || m.channels !== (m.input_extra_features === 'v1' ? 146 : 175)
      || typeof m.torch_version !== 'string'
      || !/^\d+\.\d+\.\d+(\+cpu)?$/.test(m.torch_version)) {
    throw new Error('unsupported or incomplete CPU model manifest');
  }
  if (!m.arch || typeof m.arch !== 'object' || !m.resolved_model_config
      || typeof m.resolved_model_config !== 'object') throw new Error('missing checkpoint architecture provenance');
  return m;
}
export function inspect(packagePath) {
  const p = resolve(packagePath);
  const sidecar = join(dirname(p), basename(p, extname(p)) + '.json');
  for (const file of [p, sidecar]) {
    const st = lstatSync(file);
    if (!st.isFile() || st.isSymbolicLink()) throw new Error('package and sidecar must be regular non-symlink files');
  }
  const m = validate(JSON.parse(readFileSync(sidecar, 'utf8')));
  if (createHash('sha256').update(readFileSync(p)).digest('hex') !== m.sha256)
    throw new Error('model package SHA256 mismatch');
  return m;
}
export function header(m) {
  validate(m);
  return '#pragma once\n' + [
    ['DF_PACKAGE_SHA256', JSON.stringify(m.sha256)],
    ['DF_TORCH_RELEASE', JSON.stringify(m.torch_version.split('+')[0])],
    ['DF_MODEL_HISTORY', m.input_history_encoding === 'lc0_root' ? '0u' : '1u'],
    ['DF_MODEL_FEATURES', m.input_extra_features === 'v1' ? '0u' : '1u'],
    ['DF_MODEL_CHANNELS', m.channels + 'u'], ['DF_MODEL_BATCH', m.batch + 'u'],
  ].map(([k, v]) => '#define ' + k + ' ' + v + '\n').join('');
}
if (import.meta.main) {
  const [packagePath, target, ...extra] = process.argv.slice(2);
  if (!packagePath || extra.length) throw new Error('expected PACKAGE [NEW_CONFIG_DIRECTORY]');
  const m = inspect(packagePath);
  if (target) {
    // Caller creates a fresh private build directory. Never overwrite source files.
    writeFileSync(join(target, 'leaf_model_config.h'), header(m), {flag: 'wx'});
    writeFileSync(join(target, 'model-binding.json'), JSON.stringify(m, null, 2) + '\n', {flag: 'wx'});
  }
  console.log(JSON.stringify({package_sha256: m.sha256, checkpoint_sha256: m.checkpoint_sha256,
    torch_version: m.torch_version, channels: m.channels, batch: m.batch}));
}
