// Build-time source integrity only. No JavaScript/Bun is deployed with the engine.
import {createHash} from "node:crypto";
import {lstatSync, readFileSync, readdirSync} from "node:fs";
import {join} from "node:path";

// One pin owns fetch revision, cache identity and source verification together.
export const PIN = Object.freeze(JSON.parse(readFileSync(new URL("./toolchain.json", import.meta.url), "utf8")));
if (PIN.repository !== "https://github.com/jjoshua2/bend.git"
    || !/^[a-f0-9]{40}$/.test(PIN.revision)
    || !/^[a-f0-9]{40}$/.test(PIN.upstream_revision)
    || !/^\d+\.\d+\.\d+$/.test(PIN.upstream_version)
    || !Number.isSafeInteger(PIN.source_files) || PIN.source_files < 4
    || !/^[a-f0-9]{64}$/.test(PIN.source_sha256)) {
  throw new Error("invalid standalone compiler pin");
}

export function fingerprint(root) {
  function files(path) {
    const stat = lstatSync(join(root, path));
    if (stat.isSymbolicLink()) throw new Error("unexpected compiler symlink: " + path);
    if (stat.isFile()) return [path];
    if (!stat.isDirectory()) throw new Error("unexpected compiler file type: " + path);
    return readdirSync(join(root, path)).flatMap(name => files(path + "/" + name));
  }
  const paths = ["bend2/base.bend", "bend2/bend.ts", "bend2/comp.ts", "bend2/main.ts", "bend2/effs"]
    .flatMap(files).sort();
  const hash = createHash("sha256");
  for (const path of paths) {
    hash.update(path + "\0");
    hash.update(createHash("sha256").update(readFileSync(join(root, path))).digest());
  }
  return {source_files: paths.length, source_sha256: hash.digest("hex")};
}

export function verifyCompiler(root) {
  const actual = fingerprint(root);
  if (actual.source_files !== PIN.source_files || actual.source_sha256 !== PIN.source_sha256) {
    throw new Error("compiler source mismatch: expected " + PIN.revision + " (" + PIN.source_sha256
      + "), found " + actual.source_sha256 + " in " + actual.source_files
      + " files; use a fresh compiler directory, do not reset an existing checkout");
  }
  return actual;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  if (args.length !== 1) throw new Error("expected compiler directory, --revision or --repository");
  if (args[0] === "--revision") console.log(PIN.revision);
  else if (args[0] === "--repository") console.log(PIN.repository);
  else {
    const actual = verifyCompiler(args[0]);
    console.log("verified compiler " + PIN.revision + " (Bend " + PIN.upstream_version
      + " + U64), " + actual.source_files + " files, source " + actual.source_sha256);
  }
}
