// Opt-in build contracts: bun test native/bend_engine/standalone/verify_compiler.test.js
// Synthetic source fixtures only. No C compilation, perft, engine or neural work.
import {afterEach, expect, test} from "bun:test";
import {mkdtempSync, mkdirSync, readFileSync, renameSync, rmSync, symlinkSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fingerprint, PIN, verifyCompiler} from "./verify_compiler.js";

const roots = [];
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, {recursive: true, force: true}); });
function fixture() {
  const root = mkdtempSync(join(tmpdir(), "bend-pin-test-"));
  roots.push(root);
  mkdirSync(join(root, "bend2/effs/nested"), {recursive: true});
  for (const file of ["base.bend", "bend.ts", "comp.ts", "main.ts", "effs/a.c", "effs/nested/a.js"]) {
    writeFileSync(join(root, "bend2/" + file), file + "\n");
  }
  return root;
}

test("fetch and source verifier share the exact documented U64 pin", () => {
  expect(PIN.revision).toBe("fd1df81707fd758f749a9570ccb5b12b1bb2fea3");
  expect(PIN.upstream_version).toBe("2.0.20");
  expect(PIN.source_files).toBe(84);
  expect(PIN.source_sha256).toBe("88f7505294c77f8187396aaeefd4d1845a6194d7e64b0ab9a982455bf2d8d38b");
});
test("fingerprint is independent of checkout location", () => {
  expect(fingerprint(fixture())).toEqual(fingerprint(fixture()));
});
test("nested effects and core inputs all count", () => {
  expect(fingerprint(fixture()).source_files).toBe(6);
});
test("core mutation changes fingerprint", () => {
  const root = fixture(), before = fingerprint(root);
  writeFileSync(join(root, "bend2/comp.ts"), "changed");
  expect(fingerprint(root).source_sha256).not.toBe(before.source_sha256);
});
test("effect mutation changes fingerprint", () => {
  const root = fixture(), before = fingerprint(root);
  writeFileSync(join(root, "bend2/effs/nested/a.js"), "changed");
  expect(fingerprint(root).source_sha256).not.toBe(before.source_sha256);
});
test("extra effects are not silently ignored", () => {
  const root = fixture(), before = fingerprint(root);
  writeFileSync(join(root, "bend2/effs/extra.c"), "extra");
  expect(fingerprint(root).source_files).toBe(7);
  expect(fingerprint(root).source_sha256).not.toBe(before.source_sha256);
});
test("renaming unchanged effect bytes changes the fingerprint", () => {
  const root = fixture(), before = fingerprint(root);
  renameSync(join(root, "bend2/effs/a.c"), join(root, "bend2/effs/b.c"));
  expect(fingerprint(root).source_sha256).not.toBe(before.source_sha256);
});
test("missing core input is rejected", () => {
  const root = fixture();
  rmSync(join(root, "bend2/main.ts"));
  expect(() => fingerprint(root)).toThrow();
});
test("core symlinks are rejected", () => {
  const root = fixture();
  rmSync(join(root, "bend2/main.ts"));
  symlinkSync("comp.ts", join(root, "bend2/main.ts"));
  expect(() => fingerprint(root)).toThrow("compiler symlink");
});
test("effect directory symlinks are rejected", () => {
  const root = fixture();
  symlinkSync("nested", join(root, "bend2/effs/alias"));
  expect(() => fingerprint(root)).toThrow("compiler symlink");
});
test("mismatched source fails without mutating the checkout", () => {
  const root = fixture(), before = fingerprint(root);
  expect(() => verifyCompiler(root)).toThrow("use a fresh compiler directory");
  expect(fingerprint(root)).toEqual(before);
});
test("build derives revision and revision-keyed cache from the pin reader", () => {
  const build = readFileSync(new URL("./build.sh", import.meta.url), "utf8");
  expect(build).toContain('"$here/verify_compiler.js" --revision');
  expect(build).toContain('bend_standalone_toolchain/$revision/source');
  expect(build).not.toContain("d9b9bce9ce4c02583ca1a548cfd239b368c3fc77");
});
