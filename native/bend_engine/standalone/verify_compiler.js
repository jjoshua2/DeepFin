// Build-time source integrity check. This is never deployed with the engine.
import {createHash} from "node:crypto";
import {readFileSync, readdirSync} from "node:fs";
import {join} from "node:path";
const root = process.argv[2];
if (!root) throw new Error("expected compiler source directory");
function files(path) {
  return readdirSync(join(root, path), {withFileTypes: true}).flatMap(e => {
    const name = path + "/" + e.name;
    if (e.isSymbolicLink()) throw new Error("unexpected compiler symlink: " + name);
    return e.isDirectory() ? files(name) : [name];
  });
}
const hash = createHash("sha256");
for (const p of ["bend2/base.bend", "bend2/bend.ts", "bend2/comp.ts", "bend2/main.ts", ...files("bend2/effs")].sort()) {
  hash.update(p + "\0");
  hash.update(createHash("sha256").update(readFileSync(join(root, p))).digest());
}
const actual = hash.digest("hex");
if (actual !== "9aad7eaa045f981f8d8ba444dfa807e69f8d769d8934d03854544eb090e3ba5a") throw new Error("compiler source mismatch: " + actual);
console.log("verified compiler source " + actual);
