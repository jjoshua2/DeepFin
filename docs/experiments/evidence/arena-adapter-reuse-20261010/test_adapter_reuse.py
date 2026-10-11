"""Stdlib-only differential contracts; no live source imports or canonical leases."""
import argparse
import ast
import builtins
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "short-e-evaluation-owner-v4/arena_phase_adapter_pair_bound_v4.py"
FRESH = ROOT / "fresh_epoch1_arena_phase_adapter_20261010.py"


def load(path, raw=None):
    module = types.ModuleType("isolated_adapter_" + path.stem)
    module.__file__ = str(path)
    exec(compile(path.read_bytes() if raw is None else raw, str(path), "exec"), module.__dict__)
    return module


def original(name):
    expected = {
        "fresh_epoch1_arena_phase_adapter_20261010.py": "2f01463b53e5c7ba8e71677ff3a470680f86ec6e7decbe28360184a8086c216a",
        "short-e-evaluation-owner-v4/arena_phase_adapter_pair_bound_v4.py": "70777653e0a6579b138e5b69cb4139cad09bbaecdda710242f7c0ce8b08a19fa",
    }
    raw = (ROOT / "baseline" / (name + ".txt")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected[name]:
        raise AssertionError("immutable baseline fixture changed: " + name)
    return raw


def arguments(**changes):
    args = argparse.Namespace(
        mode="prepare", manifest_sha256="m" * 64, manifest_path="manifest.json",
        output_root=".", panel_sha256="p" * 64, candidate_sha256="c" * 64,
        control_sha256="r" * 64, attempt_number=1, adapter_source=CORE,
        candidate_name="selected_e_three_pass_candidate",
        reference_name="selected_e_three_pass_control",
        pair_bound_wrapper=Path("/mnt/c/Users/jjosh/Documents/Codex/2026-10-08/task/short_e_pair_bound_recovery_v1.py"),
    )
    vars(args).update(changes)
    return args


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.core = load(CORE)
        self.fresh = load(FRESH)

    def test_only_four_shared_functions_change(self):
        before = ast.parse(original("short-e-evaluation-owner-v4/arena_phase_adapter_pair_bound_v4.py"))
        after = ast.parse(CORE.read_bytes())
        bodies = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = bodies(before), bodies(after)
        self.assertEqual(old.keys(), new.keys())
        self.assertEqual({k for k in old if old[k] != new[k]},
                         {"main", "verify_authority", "arena_argv", "mode_attempt"})

    def test_both_variants_generate_byte_identical_legacy_commands(self):
        refs = {k: Path("/private-fixture") / k for k in ("control", "candidate", "panel")}
        for name, candidate, reference in [
            ("short-e-evaluation-owner-v4/arena_phase_adapter_pair_bound_v4.py", "selected_e_three_pass_candidate", "selected_e_three_pass_control"),
            ("fresh_epoch1_arena_phase_adapter_20261010.py", "fresh104363327_epoch1", "THREE_frozen206922"),
        ]:
            legacy = load(ROOT / name, original(name))
            args = arguments(candidate_name=candidate, reference_name=reference)
            actual = self.core.arena_argv(args, refs, Path("/private-output"), Path("/private-seal"), "s" * 64)
            expected = legacy.arena_argv(args, refs, Path("/private-output"), Path("/private-seal"), "s" * 64)
            self.assertEqual(actual, expected)

    def test_main_routes_defaults_and_explicit_bindings_without_new_cli_flags(self):
        for changes in ({}, dict(adapter_source=FRESH, candidate_name="fresh104363327_epoch1",
                                reference_name="THREE_frozen206922", pair_bound_wrapper=Path("/private-shim"))):
            args = arguments()
            with patch.object(self.core.argparse.ArgumentParser, "parse_args", return_value=args), \
                 patch.object(self.core, "mode_prepare", return_value=7) as prepare:
                self.assertEqual(self.core.main(**changes), 7)
                realized = prepare.call_args.args[0]
                self.assertEqual(realized.adapter_source, changes.get("adapter_source", CORE))
                self.assertEqual(realized.candidate_name, changes.get("candidate_name", "selected_e_three_pass_candidate"))
                self.assertEqual(realized.pair_bound_wrapper, changes.get("pair_bound_wrapper", arguments().pair_bound_wrapper))
        argv = ["adapter"]
        for action in self.core.parser()._actions:
            if action.required:
                argv.extend([action.option_strings[0], "prepare" if action.dest == "mode" else "private-fixture"])
        error = io.StringIO()
        with patch("sys.argv", argv + ["--adapter-source", "elsewhere"]), redirect_stderr(error), \
             patch.object(self.core, "mode_prepare", side_effect=AssertionError("workload")):
            with self.assertRaises(SystemExit) as refused:
                self.core.main()
        self.assertEqual(refused.exception.code, 2)
        self.assertIn("unrecognized arguments: --adapter-source elsewhere", error.getvalue())

    def test_fresh_wrapper_executes_pinned_bytes_and_routes_fresh_identity(self):
        observed = {}
        def execute(code, namespace):
            builtins.exec(code, namespace)
            def prepare(args):
                observed.update(vars(args))
                return 9
            namespace["mode_prepare"] = prepare
        with patch.object(self.core.argparse.ArgumentParser, "parse_args", return_value=arguments()), \
             patch.object(self.fresh, "exec", execute, create=True):
            self.assertEqual(self.fresh.main(), 9)
        self.assertEqual(observed["adapter_source"], FRESH)
        self.assertEqual(observed["reference_name"], "THREE_frozen206922")
        self.assertEqual(observed["candidate_name"], "fresh104363327_epoch1")
        self.assertEqual(str(observed["pair_bound_wrapper"]), "/mnt/c/Users/jjosh/Documents/Codex/2026-10-08/task/fresh_epoch1_pair_bound_20261010.py")

    def test_changed_shared_source_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "untrusted.py"
            source.write_text('raise AssertionError("executed changed source")')
            with patch.object(self.fresh, "SHARED_SOURCE", source):
                with self.assertRaisesRegex(SystemExit, "shared adapter source changed"):
                    self.fresh.main()

    def test_attempt_uses_bound_shim_and_keeps_attempt_limit(self):
        with tempfile.TemporaryDirectory() as td:
            args = arguments(output_root=td, pair_bound_wrapper=Path("/private-fresh-shim"))
            manifest = {"attempt_argv": ["--fixture"], "overlay_files": {}}
            with patch.object(self.core, "verify_manifest", return_value=(manifest, {})), \
                 patch.object(self.core, "checked_overlay", return_value={}), \
                 patch.object(self.core, "verify_qualified_source"), \
                 patch.object(self.core, "run_child", return_value=3) as child:
                self.assertEqual(self.core.mode_attempt(args), 3)
                self.assertEqual(child.call_args.args[0], [self.core.sys.executable, "-B", "/private-fresh-shim", "--fixture"])
                for n in (None, 0, 26):
                    args.attempt_number = n
                    with self.assertRaises(self.core.AdapterError):
                        self.core.mode_attempt(args)
                self.assertEqual(child.call_count, 1)

    def test_incomplete_attempt_keeps_descriptors_and_create_only_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(source_seal_path="seal", source_seal_sha256="s",
                                               panel_sha256="p", control_sha256="r", candidate_sha256="c")))
            args = arguments(output_root=td, manifest_path=str(manifest))
            result = root / "attempt-result.json"
            with patch.object(self.core, "inherited_fds", return_value=(81, 82)), \
                 patch.object(self.core, "set_runtime_env", return_value={}), \
                 patch.object(self.core.subprocess, "run", return_value=types.SimpleNamespace(returncode=3)) as launch, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(self.core.run_child(["inert"], args=args, output_root=root,
                                                    result_path=result, mode="attempt"), 3)
                self.assertEqual(launch.call_args.kwargs["pass_fds"], (81, 82))
                self.assertEqual(json.loads(result.read_text())["status"], "VALID_INCOMPLETE_ATTEMPT_EXIT_3")
                saved = result.read_bytes()
                with self.assertRaises(self.core.AdapterError):
                    self.core.run_child(["inert"], args=args, output_root=root, result_path=result, mode="attempt")
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(result.read_bytes(), saved)

    def test_finalizer_does_not_accept_incomplete_attempt_code(self):
        with patch.object(self.core, "inherited_fds", return_value=(81, 82)), \
             patch.object(self.core, "set_runtime_env", return_value={}), \
             patch.object(self.core.subprocess, "run", return_value=types.SimpleNamespace(returncode=3, stderr="incomplete")):
            for mode in ("progress", "final"):
                with self.assertRaises(self.core.AdapterError):
                    self.core.run_finalizer(["inert"], args=arguments(), output_root=Path("."), mode=mode, manifest={}, refs={})

    def test_missing_inherited_lease_descriptor_refused(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(self.core.AdapterError):
                self.core.inherited_fds()

    def test_authority_binds_executing_wrapper_and_preserves_owner_leases(self):
        core = self.core
        with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
            root = Path(td)
            config, catalog = root / "config", root / "catalog"
            config.write_text("fixture-config"); catalog.write_text("fixture-catalog")
            stack.enter_context(patch.object(core, "FROZEN_CONFIG", config))
            gpu, disk = root / "private-gpu", root / "private-io"
            gpu.touch(); disk.touch()
            owner = dict(pid=42, start_ticks="100", boot_id="boot")
            args = arguments(adapter_source=FRESH, tablebase_catalog=str(catalog))
            docs = {
                "operation": {"arena": {"adapter_ref": {"path": str(FRESH), "sha256": core.file_sha(FRESH)},
                    "tablebase_catalog_ref": {"path": str(catalog), "sha256": core.file_sha(catalog)},
                    "live_config_ref": {"path": str(config), "sha256": core.file_sha(config)},
                    "verification_seconds": 60, "max_sessions": 25, "whole_attempt_seconds": 1650,
                    "phase_wall_seconds": 43200, "pairs": 2606, "games": 5212}},
                "authorization": {}, "budget": {},
                "lane_release": {"schema": "deepfin_three_pass_runtime_lane_release_v1", "status": "PASS_PARENT_REVIEWED_THREE_PASS_LANE_RELEASE",
                    "owner_stop_confirmed": True, "nonce": "private", "science_owner_identity": owner,
                    "canonical_lease_identities": {str(gpu): [2096, 81348917], str(disk): [2096, 18503149]}},
                "operation_baseline": {"schema": "deepfin_three_pass_science_slot_baseline_v1", "owner_identity": owner,
                    "scope_started_monotonic": time.monotonic() - 1, "scope_deadline_monotonic": time.monotonic() + 100},
            }
            refs = {k: root / (k + ".json") for k in docs}
            # Authority digests are compared with supplied values; actual byte refs
            # are independently enforced by validate_request_files/verify_manifest.
            digests = {k: hashlib.sha256(k.encode()).hexdigest() for k in docs}
            for k, digest in digests.items():
                setattr(args, k + "_sha256", digest)
            ref = lambda k: dict(path=str(refs[k]), sha256=digests[k])
            docs["operation"]["budget_ref"] = ref("budget")
            docs["authorization"].update(operation_sha256=digests["operation"], budget_sha256=digests["budget"])
            docs["lane_release"].update(operation_ref=ref("operation"), whole_operation_authorization_ref=ref("authorization"))
            docs["operation_baseline"].update(operation_sha256=digests["operation"], authorization_sha256=digests["authorization"], lane_release_ref=ref("lane_release"))
            def publish():
                for k, value in docs.items(): refs[k].write_text(json.dumps(value))
            publish()
            real_stat, real_read = Path.stat, Path.read_text
            def private_stat(p, *a, **kw):
                if p == gpu: return types.SimpleNamespace(st_dev=2096, st_ino=81348917)
                if p == disk: return types.SimpleNamespace(st_dev=2096, st_ino=18503149)
                return real_stat(p, *a, **kw)
            stack.enter_context(patch.object(Path, "stat", private_stat))
            stack.enter_context(patch.object(Path, "read_text", lambda p, *a, **kw: "boot" if str(p) == "/proc/sys/kernel/random/boot_id" else real_read(p, *a, **kw)))
            stack.enter_context(patch.object(core.os, "getppid", return_value=42))
            stack.enter_context(patch.object(core, "_linux_start_ticks", return_value="100"))
            stack.enter_context(patch.dict(os.environ, DEEPFIN_INHERITED_GPU_LOCK_FD="81", DEEPFIN_INHERITED_IO_LOCK_FD="82"))
            stat_fd = stack.enter_context(patch.object(core.os, "fstat", side_effect=lambda fd: types.SimpleNamespace(st_dev=2096, st_ino=81348917 if fd == 81 else 18503149)))
            stack.enter_context(patch.object(core.fcntl, "fcntl", return_value=os.O_RDONLY))
            core.verify_authority(args, refs)
            args.adapter_source = CORE
            with self.assertRaisesRegex(core.AdapterError, "adapter source"):
                core.verify_authority(args, refs)
            args.adapter_source = FRESH
            with patch.object(core.os, "getppid", return_value=43):
                with self.assertRaisesRegex(core.AdapterError, "parent"):
                    core.verify_authority(args, refs)
            with patch.object(core, "_linux_start_ticks", return_value="101"):
                with self.assertRaisesRegex(core.AdapterError, "start ticks"):
                    core.verify_authority(args, refs)
            stat_fd.side_effect = lambda fd: types.SimpleNamespace(st_dev=0, st_ino=0)
            with self.assertRaisesRegex(core.AdapterError, "canonical lease"):
                core.verify_authority(args, refs)

    def test_manifest_byte_change_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            p.write_bytes(b"original")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            p.write_bytes(b"changed")
            with self.assertRaises(self.core.AdapterError):
                self.core.verify_ref(str(p), digest, "prepared manifest")


if __name__ == "__main__":
    unittest.main()
