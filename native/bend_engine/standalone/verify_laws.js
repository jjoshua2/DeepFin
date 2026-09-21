// Build-time proof gate only. The deployed engine needs neither Bun nor proofs.
// Full finite-domain checking is intentional; no compiled-C result substitutes
// for a proof. The separate opt-in runtime qualification covers native lowering.
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {spawnSync} from "node:child_process";
import {PIN, verifyCompiler} from "./verify_compiler.js";

const here = import.meta.dirname;
const success = "All terms check.";
const names = ["index_ignores_off_mask", "index_full_width", "index_recovers_masked_input",
  "reflection_preserves_conditions", "fill_matches_lookup", "every_visited_slot_is_safe", "canonical_tables_checked",
  "every_canonical_segment_is_safe", "tables_match_lookup", "lookup_uses_checked_address"];
const proofFiles = ["LAWS.bend", "PROOF.bend", "proofs/Bits.bend", "proofs/Core.bend",
  "proofs/Spec.bend", "proofs/Lookup.bend", "proofs/Fast.bend", "proofs/u64/LAWS.bend", "proofs/u64/PROOF.bend", "proofs/u64/Words.bend"];
const sha = data => createHash("sha256").update(data).digest("hex");

export function clean(result) {
  assert.equal(result.error, undefined, "checker execution error: " + result.error);
  assert.equal(result.signal, null, "checker terminated: " + result.signal);
  assert.equal(result.status, 0, result.text);
  assert.equal(result.text, success, result.text);
}
function invoke(compiler, file, timeout = 60000) {
  const r = spawnSync(process.execPath, [path.join(compiler, "bend2/main.ts"), file, "--check-only"], {
    encoding: "utf8", timeout, maxBuffer: 16 << 20,
    env: {...process.env, BEND_NO_TELEMETRY: "1"},
  });
  return {error: r.error, signal: r.signal, status: r.status, text: (r.stdout + r.stderr).trim()};
}
export function policy(dir) {
  const allowed = new Set([...proofFiles, "Tables.bend", "../bitboard_probe/Sliders.bend"]
    .map(file => path.resolve(dir, file)));
  for (const file of allowed) {
    assert.ok(fs.lstatSync(file).isFile(), "proof source must be a regular file: " + file);
    const code = fs.readFileSync(file, "utf8").split("\n").map(l => l.split("#")[0]).join("\n");
    assert.doesNotMatch(code, /@unsafe|\?/, file + ": unsafe dependency or proof hole");
    for (const [, dependency] of code.matchAll(/^\s*import\s+(\S+)/gm)) {
      assert.ok(dependency === "Base" || (dependency.startsWith(".")
        && allowed.has(path.resolve(path.dirname(file), dependency))), "unexpected proof import: " + dependency);
    }
  }
  const laws = fs.readFileSync(path.join(dir, "LAWS.bend"), "utf8");
  const proof = fs.readFileSync(path.join(dir, "PROOF.bend"), "utf8");
  assert.deepEqual([...laws.matchAll(/^law ([A-Za-z0-9_]+):/gm)].map(m => m[1]), names,
    "the approved laws must not silently disappear");
  assert.match(proof, /^import \.\/LAWS\.bend as Laws$/m, "PROOF.bend must import LAWS.bend");
  assert.deepEqual([...proof.matchAll(/^def Laws\.([A-Za-z0-9_]+)\(/gm)].map(m => m[1]), names,
    "every law needs exactly one proof, in declaration order");
}
export function dependencies(dir, compiler) {
  verifyCompiler(compiler);
  const deps = JSON.parse(fs.readFileSync(path.join(dir, "proofs/dependencies.json"), "utf8"));
  assert.equal(deps.repository, PIN.repository);
  assert.equal(deps.revision, PIN.revision, "proof dependency/compiler revisions differ");
  assert.deepEqual(Object.keys(deps.files).sort(), ["u64/LAWS.bend", "u64/LICENSE", "u64/PROOF.bend", "u64/Words.bend"]);
  for (const [file, info] of Object.entries(deps.files)) {
    assert.equal(info.source, file === "u64/LICENSE" ? "LICENSE" : "demos/proof_u64/" + path.basename(file));
    const local = fs.readFileSync(path.join(dir, "proofs", file));
    assert.equal(sha(local), info.sha256, "vendored proof changed: " + file);
    // Read the immutable Git object, not a potentially modified source file.
    const r = spawnSync("git", ["-C", compiler, "show", PIN.revision + ":" + info.source], {
      timeout: 10000, maxBuffer: 2 << 20,
    });
    assert.equal(r.error, undefined);
    assert.equal(r.status, 0, "cannot read the pinned proof dependency: " + info.source);
    assert.equal(sha(r.stdout), info.sha256, "proof is not from the pinned compiler commit: " + file);
  }
}
function replace(file, before, after) {
  const text = fs.readFileSync(file, "utf8");
  assert.ok(text.includes(before), "mutation anchor missing: " + before);
  fs.writeFileSync(file, text.replace(before, after));
}
function controls(compiler, temp) {
  const results = [];
  function variant(name, edit, target = "Check.bend", pattern = /expected[\s\S]*observed/) {
    const root = path.join(temp, name);
    const dir = path.join(root, "standalone");
    fs.cpSync(here, dir, {recursive: true});
    fs.cpSync(path.join(here, "../bitboard_probe"), path.join(root, "bitboard_probe"), {recursive: true});
    fs.writeFileSync(path.join(dir, "Check.bend"), "import Base\nimport ./proofs/Spec.bend as Spec\n"
      + "law check: {Spec.walk(4n, U64.from_u32(3), U64.zero(), 0, 512, 512, 4) == True{} : Bool}\n"
      + "def check(): {==}\n");
    clean(invoke(compiler, path.join(dir, "Check.bend")));
    edit(dir);
    const result = invoke(compiler, path.join(dir, target));
    assert.equal(result.error, undefined, name + ": checker did not complete");
    assert.equal(result.signal, null, name + ": checker terminated");
    assert.equal(result.status, 1, name + ": broken implementation was accepted\n" + result.text);
    assert.match(result.text, pattern, name + ": rejected for the wrong reason");
    results.push({control: name, rejected_by: "checker", status: result.status});
  }
  variant("pext-high-word", d => replace(path.join(d, "../bitboard_probe/Sliders.bend"),
    "U64.low(U64.pext(occupied, mask))", "U64.high(U64.pext(occupied, mask))"), "proofs/Core.bend");
  variant("wrong-subset-order", d => replace(path.join(d, "Tables.bend"),
    "U64.sub(subset, mask)", "U64.add(subset, mask)"), "proofs/Core.bend");
  variant("shifted-lookup-address", d => replace(path.join(d, "../bitboard_probe/Sliders.bend"),
    "U32.add(offset, index)", "U32.inc(U32.add(offset, index))"), "proofs/Lookup.bend");
  variant("shifted-production-write", d => replace(path.join(d, "Tables.bend"),
    "Array.set(U64, a, at, slider(", "Array.set(U64, a, U32.inc(at), slider("), "proofs/Core.bend");
  variant("wrong-production-segment-offset", d => replace(path.join(d, "Tables.bend"),
    "tables(p, U32.inc(key), U32.add(at, size),", "tables(p, U32.inc(key), U32.inc(U32.add(at, size)),"), "proofs/Core.bend");
  variant("wrong-production-metadata", d => replace(path.join(d, "Tables.bend"),
    "Array.set(U64, a, key, mask)", "Array.set(U64, a, key, U64.zero())"), "proofs/Core.bend");
  variant("unrestricted-mask-truncation", d => fs.writeFileSync(path.join(d, "Check.bend"),
    "import Base\nimport ../bitboard_probe/Sliders.bend as Sliders\n"
    + "law false_claim: {U64.from_u32(Sliders.pext_index(U64.bit(63n), U64.not(U64.zero()))) == U64.bit(63n) : U64}\n"
    + "def false_claim(): {==}\n"));

  variant("corrupted-reflection-extractor", d => replace(path.join(d, "proofs/Fast.bend"),
    "Con{b, selected(p, t, m)}", "Con{Bool.not(b), selected(p, t, m)}"), "proofs/Fast.bend");
  variant("dropped-reflection-index-bound", d => replace(path.join(d, "proofs/Spec.bend"),
    "def step_ok(mask: U64, subset: U64, i: U32, at: U32, offset: U32, size: U32) -> Bool:\n  step_values(Fast.pext(subset, mask), i, at, offset, size)",
    "def step_ok(mask: U64, subset: U64, i: U32, at: U32, offset: U32, size: U32) -> Bool:\n  True{}"), "proofs/Core.bend");

  for (const name of ["missing-proof", "missing-laws-import", "proof-hole", "unsafe-proof"]) {
    const dir = path.join(temp, name, "standalone");
    fs.cpSync(here, dir, {recursive: true});
    fs.cpSync(path.join(here, "../bitboard_probe"), path.join(temp, name, "bitboard_probe"), {recursive: true});
    if (name === "missing-proof") replace(path.join(dir, "PROOF.bend"), "def Laws.index_full_width(", "def accidentally_removed(");
    if (name === "missing-laws-import") replace(path.join(dir, "PROOF.bend"), "import ./LAWS.bend as Laws", "# removed");
    if (name === "proof-hole") replace(path.join(dir, "PROOF.bend"), "  Bits.zero_high(U64.pext(occupied, mask), high_zero)", "  ?TODO");
    if (name === "unsafe-proof") replace(path.join(dir, "PROOF.bend"), "def Laws.index_full_width(", "@unsafe\ndef Laws.index_full_width(");
    assert.throws(() => policy(dir));
    results.push({control: name, rejected_by: "proof-policy"});
  }
  const unsafe = path.join(temp, "Unsafe.bend");
  fs.writeFileSync(unsafe, "import Base\nlaw claim: {0 == 0 : U32}\n@unsafe\ndef claim(): {==}\n");
  const warned = invoke(compiler, unsafe);
  assert.equal(warned.status, 0);
  assert.match(warned.text, /unsafe or foreign/);
  assert.throws(() => clean(warned));
  results.push({control: "status-zero-unsafe-warning", rejected_by: "clean-verdict", raw_status: 0});

  const cycle = spawnSync("git", ["-C", compiler, "show", PIN.revision + ":tests/check/template_inst_cycle.bend"],
    {encoding: "utf8", timeout: 10000, maxBuffer: 1 << 20});
  assert.equal(cycle.error, undefined);
  assert.equal(cycle.status, 0, "cannot read pinned cyclic-instance regression");
  const cycleFile = path.join(temp, "Cycle.bend");
  fs.writeFileSync(cycleFile, cycle.stdout);
  const cyclic = invoke(compiler, cycleFile);
  assert.equal(cyclic.error, undefined);
  assert.equal(cyclic.signal, null);
  assert.equal(cyclic.status, 1, "cyclic template instance must not prove a false equality");
  assert.match(cyclic.text, /expected : a decreasing self-call/);
  results.push({control: "cyclic-template-false-proof", rejected_by: "checker", status: 1});

  const dir = path.join(temp, "vendor-drift", "standalone");
  fs.cpSync(here, dir, {recursive: true});
  fs.appendFileSync(path.join(dir, "proofs/u64/Words.bend"), "\n# changed dependency\n");
  assert.throws(() => dependencies(dir, compiler), /vendored proof changed/);
  results.push({control: "modified-proof-dependency", rejected_by: "pin-integrity"});
  const deps = JSON.parse(fs.readFileSync(path.join(dir, "proofs/dependencies.json"), "utf8"));
  deps.revision = "0".repeat(40);
  fs.writeFileSync(path.join(dir, "proofs/dependencies.json"), JSON.stringify(deps));
  assert.throws(() => dependencies(dir, compiler), /revisions differ/);
  results.push({control: "mismatched-proof-revision", rejected_by: "pin-integrity"});
  return results;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  assert.ok(args.length >= 1, "usage: bun verify_laws.js COMPILER [--controls-only] [--report FILE]");
  const compiler = path.resolve(args.shift());
  let onlyControls = false, reportPath;
  while (args.length) {
    const arg = args.shift();
    if (arg === "--controls-only") onlyControls = true;
    else if (arg === "--report" && args.length) reportPath = args.shift();
    else throw new Error("unknown/incomplete argument: " + arg);
  }
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "deepfin-slider-laws-"));
  const start = performance.now();
  try {
    dependencies(here, compiler);
    policy(here);
    // The universal bridge and all mutation controls are cheap and run first.
    clean(invoke(compiler, path.join(here, "proofs/Core.bend")));
    const negative = controls(compiler, temp);
    console.error("Compiler/proof pins, universal bridge, and " + negative.length + " rejection controls pass.");
    if (!onlyControls) {
      console.error("Checking all ten laws, including every canonical occupancy subset (no native oracle).");
      clean(invoke(compiler, path.join(here, "PROOF.bend"), 30 * 60 * 1000));
    }
    const report = {status: "PASS", scope: onlyControls ? "controls-only; full laws NOT checked" : "complete-source-proof-gate",
      compiler_revision: PIN.revision, source_sha256: PIN.source_sha256,
      laws: onlyControls ? [] : names, canonical_segments: onlyControls ? null : 128,
      canonical_subsets: onlyControls ? null : 107648, imported_u64_contracts: 16,
      negative_controls: negative, elapsed_seconds: (performance.now() - start) / 1000};
    const json = JSON.stringify(report, null, 2) + "\n";
    if (reportPath) fs.writeFileSync(reportPath, json);
    console.log(json);
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
}
