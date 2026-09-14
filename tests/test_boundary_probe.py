"""Deterministic unit tests for the diagnostic-only infra boundary probe.

Runs with the standard library only:

    python3 -m unittest discover -s tests -t .

The suite covers the required points: a scrubbed child environment, a /proc
scanner that yields booleans and counts only, synthetic process-environment
fixtures detected without exposing values, tolerated permission denials,
cleanup after allowed worktree and store writes, ``NOT_PRESENT`` for a missing
store, a stable ``diagnostic_only`` result schema, and the allowlist entry.
"""

from __future__ import annotations

import dataclasses
import errno
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = REPO_ROOT / "science" / "cb16_science" / "boundary_probe.py"
ALLOWLIST_PATH = REPO_ROOT / "config" / "cb16_science_allowlist.json"
SECRET = "ghp_not_a_real_secret_value_000000"
REQUIRED_KEYS = {
    "schema",
    "diagnostic_only",
    "worktree_write",
    "store_write",
    "child_env_has_github_token",
    "parent_or_peer_proc_exposes_github_token_key",
    "proc_scan_status",
}


def load_probe():
    """Load the entrypoint by path: the repository has no installed package."""

    spec = importlib.util.spec_from_file_location("cb16_boundary_probe_under_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe()


class ProbeTestCase(unittest.TestCase):
    """Throwaway worktree, store and synthetic /proc fixtures."""

    _proc_counter = itertools.count()

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cb16-probe-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.worktree = self.tmp / "worktree"
        self.worktree.mkdir()
        self.store = self.tmp / "store"
        self.store.mkdir()

    def make_proc_root(self, name: str = "proc") -> Path:
        root = self.tmp / f"{name}-{next(self._proc_counter)}"
        root.mkdir()
        return root

    @staticmethod
    def write_environ(root: Path, pid: str, payload: bytes) -> Path:
        directory = root / pid
        directory.mkdir(parents=True, exist_ok=True)
        environ = directory / "environ"
        environ.write_bytes(payload)
        return environ

    @staticmethod
    def make_directory_unwritable(directory: Path) -> bool:
        """Make ``directory`` read-only; True when permissions are enforced."""

        directory.chmod(0o500)
        canary = directory / ".canary"
        try:
            canary.write_text("x", encoding="utf-8")
        except OSError:
            return True
        canary.unlink()
        directory.chmod(0o700)
        return False

    @staticmethod
    def make_directory_unlistable(directory: Path) -> bool:
        """Remove read permission so listing fails; True when enforced."""

        directory.chmod(0o300)
        try:
            os.listdir(directory)
        except OSError:
            return True
        directory.chmod(0o700)
        return False

    def probe_env(self, **overrides) -> dict:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(REPO_ROOT / "science"),
            "CB16_RESULT_DIR": str(self.tmp / "results"),
            "CB16_STORE": str(self.store),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.update(overrides)
        return env

    def run_entrypoint(self, env: dict):
        return subprocess.run(
            [sys.executable, "-m", "cb16_science.boundary_probe"],
            cwd=str(self.worktree),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def probe_leftovers(self, directory: Path) -> list:
        return sorted(path.name for path in directory.iterdir())


# ---------------------------------------------------------------------------
# 1 + 5: child environment and worktree write boundary
# ---------------------------------------------------------------------------


class WorktreeWriteProbeTests(ProbeTestCase):
    def test_status_is_always_from_the_known_set(self):
        result = probe.probe_worktree_write(self.worktree)
        self.assertIn(result.status, probe.WRITE_STATUSES)
        self.assertTrue(result.cleaned_up)

    def test_allowed_write_is_reported_and_cleaned_up(self):
        result = probe.probe_worktree_write(self.worktree)
        self.assertEqual(result.status, probe.WRITE_ALLOWED)
        self.assertTrue(result.cleaned_up)
        self.assertEqual(self.probe_leftovers(self.worktree), [])

    def test_blocked_write_is_reported_and_creates_nothing(self):
        if not self.make_directory_unwritable(self.worktree):
            self.skipTest("directory permissions are not enforced here")
        self.addCleanup(self.worktree.chmod, 0o700)
        result = probe.probe_worktree_write(self.worktree)
        self.assertEqual(result.status, probe.WRITE_BLOCKED)
        self.assertTrue(result.cleaned_up)

    def test_missing_worktree_is_an_error_not_a_boundary_refusal(self):
        result = probe.probe_worktree_write(self.tmp / "does-not-exist")
        self.assertEqual(result.status, probe.WRITE_ERROR)
        self.assertTrue(result.cleaned_up)

    def test_existing_file_is_never_truncated_or_removed(self):
        target = self.worktree / "pre-existing.txt"
        target.write_text("keep me\n", encoding="utf-8")
        result = probe.probe_file_write(target)
        self.assertEqual(result.status, probe.WRITE_ERROR)
        self.assertEqual(target.read_text(encoding="utf-8"), "keep me\n")

    def test_errno_classification_is_deterministic(self):
        for number in (errno.EACCES, errno.EPERM, errno.EROFS):
            with self.subTest(errno=number):
                self.assertEqual(
                    probe.classify_write_error(OSError(number, "x")), probe.WRITE_BLOCKED
                )
        self.assertEqual(probe.classify_write_error(OSError(errno.ENOSPC, "x")), probe.WRITE_ERROR)
        self.assertEqual(
            probe.classify_write_error(OSError(errno.ENOENT, "x")), probe.WRITE_ERROR
        )

    def test_child_environment_has_no_github_token(self):
        payload = probe.build_payload(
            worktree=self.worktree,
            store=str(self.store),
            child_env={},
            proc_root=self.make_proc_root(),
        )
        self.assertIs(payload["child_env_has_github_token"], False)

    def test_token_in_the_own_environment_is_detected_without_exposing_it(self):
        payload = probe.build_payload(
            worktree=self.worktree,
            store=str(self.store),
            child_env={"PATH": "/usr/bin", probe.GITHUB_TOKEN_KEY: SECRET},
            proc_root=self.make_proc_root(),
        )
        self.assertIs(payload["child_env_has_github_token"], True)
        self.assertNotIn(SECRET, json.dumps(payload))


# ---------------------------------------------------------------------------
# 2 + 3 + 4: /proc scanning, synthetic fixtures, permission denials
# ---------------------------------------------------------------------------


class ProcScanTests(ProbeTestCase):
    def test_synthetic_environ_with_the_key_is_detected_without_values(self):
        root = self.make_proc_root()
        payload = b"PATH=/usr/bin\0" + probe.GITHUB_TOKEN_MARKER + SECRET.encode() + b"\0"
        self.write_environ(root, "1234", payload)
        result = probe.scan_proc_environ(root)
        self.assertEqual(result.status, probe.PROC_OK)
        self.assertEqual(result.readable_environments, 1)
        self.assertEqual(result.token_key_matches, 1)
        self.assertEqual(result.denied_environments, 0)
        self.assertNotIn(SECRET, json.dumps(dataclasses.asdict(result)))

    def test_a_readable_environ_without_the_key_is_not_a_match(self):
        root = self.make_proc_root()
        self.write_environ(root, "7", b"PATH=/usr/bin\0HOME=/root\0")
        result = probe.scan_proc_environ(root)
        self.assertEqual(result.status, probe.PROC_OK)
        self.assertEqual(result.readable_environments, 1)
        self.assertEqual(result.token_key_matches, 0)

    def test_non_numeric_proc_entries_are_never_followed(self):
        root = self.make_proc_root()
        for name in ("self", "thread-self", "net", "sys"):
            self.write_environ(root, name, probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        self.write_environ(root, "11", b"PATH=/usr/bin\0")
        result = probe.scan_proc_environ(root)
        self.assertEqual(result.readable_environments, 1)
        self.assertEqual(result.token_key_matches, 0)

    def test_symlinked_numeric_entries_are_not_followed(self):
        root = self.make_proc_root()
        outside = self.tmp / "outside"
        self.write_environ(outside.parent, "outside", probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        (root / "4242").symlink_to(outside, target_is_directory=True)
        result = probe.scan_proc_environ(root)
        self.assertEqual(result.readable_environments, 0)
        self.assertEqual(result.token_key_matches, 0)

    def test_permission_denied_environments_are_tolerated(self):
        root = self.make_proc_root()
        self.write_environ(root, "1", probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        self.write_environ(root, "2", b"PATH=/usr/bin\0")

        def denying_reader(path: Path, limit: int) -> bytes:
            raise PermissionError(errno.EACCES, "denied", str(path))

        result = probe.scan_proc_environ(root, environ_reader=denying_reader)
        self.assertEqual(result.status, probe.PROC_DENIED)
        self.assertEqual(result.readable_environments, 0)
        self.assertEqual(result.denied_environments, 2)
        self.assertEqual(result.token_key_matches, 0)

    def test_a_denied_file_does_not_hide_a_readable_match(self):
        root = self.make_proc_root()
        self.write_environ(root, "1", b"PATH=/usr/bin\0")
        self.write_environ(root, "2", probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        real_reader = probe.read_environ_bounded
        first = root / "1" / "environ"

        def reader(path: Path, limit: int) -> bytes:
            if path == first:
                raise PermissionError(errno.EACCES, "denied", str(path))
            return real_reader(path, limit)

        result = probe.scan_proc_environ(root, environ_reader=reader)
        self.assertEqual(result.status, probe.PROC_OK)
        self.assertEqual(result.readable_environments, 1)
        self.assertEqual(result.token_key_matches, 1)
        self.assertEqual(result.denied_environments, 1)

    def test_unreadable_proc_root_reports_denied_not_a_match(self):
        root = self.make_proc_root()
        self.write_environ(root, "1", probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        if not self.make_directory_unlistable(root):
            self.skipTest("directory permissions are not enforced here")
        self.addCleanup(root.chmod, 0o700)
        result = probe.scan_proc_environ(root)
        self.assertEqual(result.status, probe.PROC_DENIED)
        self.assertEqual(result.token_key_matches, 0)

    def test_a_proc_root_that_is_not_a_directory_is_an_error(self):
        not_a_directory = self.tmp / "proc-file"
        not_a_directory.write_text("not proc\n", encoding="utf-8")
        result = probe.scan_proc_environ(not_a_directory)
        self.assertEqual(result.status, probe.PROC_ERROR)
        self.assertEqual(result.token_key_matches, 0)

    def test_the_number_of_files_read_is_bounded(self):
        root = self.make_proc_root()
        for pid in ("1", "2", "3", "4"):
            self.write_environ(root, pid, probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        result = probe.scan_proc_environ(root, max_entries=2)
        self.assertEqual(result.readable_environments, 2)
        self.assertEqual(result.token_key_matches, 2)

    def test_the_bytes_read_per_file_are_bounded(self):
        root = self.make_proc_root()
        environ = self.write_environ(root, "5", b"A" * 64 + probe.GITHUB_TOKEN_MARKER)
        self.assertEqual(len(probe.read_environ_bounded(environ, 8)), 8)
        self.assertEqual(probe.scan_proc_environ(root, max_environ_bytes=32).token_key_matches, 0)
        self.assertEqual(probe.scan_proc_environ(root, max_environ_bytes=4096).token_key_matches, 1)

    def test_scan_result_carries_only_status_and_counts(self):
        root = self.make_proc_root()
        self.write_environ(root, "9", probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        result = probe.scan_proc_environ(root)
        self.assertEqual(
            sorted(dataclasses.asdict(result)),
            ["denied_environments", "readable_environments", "status", "token_key_matches"],
        )
        self.assertNotIn(SECRET, json.dumps(dataclasses.asdict(result)))
        self.assertIsInstance(result.readable_environments, int)
        self.assertIsInstance(result.token_key_matches, int)
        self.assertIsInstance(result.denied_environments, int)

    def test_the_real_proc_root_yields_a_well_formed_result(self):
        result = probe.scan_proc_environ()
        self.assertIn(result.status, probe.PROC_STATUSES)
        self.assertGreaterEqual(result.readable_environments, 0)
        self.assertGreaterEqual(result.token_key_matches, 0)
        self.assertGreaterEqual(result.denied_environments, 0)
        self.assertLessEqual(result.token_key_matches, result.readable_environments)


# ---------------------------------------------------------------------------
# 6 + 7: store write boundary
# ---------------------------------------------------------------------------


class StoreWriteProbeTests(ProbeTestCase):
    def test_allowed_store_write_is_reported_and_cleaned_up(self):
        keep = self.store / "keep.txt"
        keep.write_text("pre-existing\n", encoding="utf-8")
        result = probe.probe_store_write(str(self.store))
        self.assertEqual(result.status, probe.WRITE_ALLOWED)
        self.assertTrue(result.cleaned_up)
        self.assertEqual(self.probe_leftovers(self.store), ["keep.txt"])
        self.assertEqual(keep.read_text(encoding="utf-8"), "pre-existing\n")
        self.assertFalse((self.store / probe.PROBE_DIR_NAME).exists())

    def test_missing_store_reports_not_present(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                result = probe.probe_store_write(value)
                self.assertEqual(result.status, probe.WRITE_NOT_PRESENT)
                self.assertTrue(result.cleaned_up)

    def test_store_path_that_does_not_exist_reports_not_present_without_creating_it(self):
        missing = self.tmp / "absent" / "store"
        result = probe.probe_store_write(str(missing))
        self.assertEqual(result.status, probe.WRITE_NOT_PRESENT)
        self.assertTrue(result.cleaned_up)
        self.assertFalse(missing.exists())

    def test_blocked_store_write_is_reported(self):
        if not self.make_directory_unwritable(self.store):
            self.skipTest("directory permissions are not enforced here")
        self.addCleanup(self.store.chmod, 0o700)
        result = probe.probe_store_write(str(self.store))
        self.assertEqual(result.status, probe.WRITE_BLOCKED)
        self.assertTrue(result.cleaned_up)

    def test_a_non_empty_probe_directory_is_left_in_place(self):
        probe_dir = self.store / probe.PROBE_DIR_NAME
        probe_dir.mkdir()
        other = probe_dir / "other.txt"
        other.write_text("someone else\n", encoding="utf-8")
        result = probe.probe_store_write(str(self.store))
        self.assertEqual(result.status, probe.WRITE_ALLOWED)
        self.assertTrue(result.cleaned_up)
        self.assertEqual(self.probe_leftovers(probe_dir), ["other.txt"])
        self.assertEqual(other.read_text(encoding="utf-8"), "someone else\n")


# ---------------------------------------------------------------------------
# 8: stable output schema
# ---------------------------------------------------------------------------


class PayloadSchemaTests(ProbeTestCase):
    def build(self, **overrides) -> dict:
        arguments = {
            "worktree": self.worktree,
            "store": str(self.store),
            "child_env": {},
            "proc_root": self.make_proc_root(),
        }
        arguments.update(overrides)
        return probe.build_payload(**arguments)

    def test_schema_is_stable_and_marks_diagnostic_only(self):
        payload = self.build()
        self.assertEqual(payload["schema"], "cb16.infra.boundary_probe.v1")
        self.assertIs(payload["diagnostic_only"], True)
        self.assertEqual(payload["schema"], probe.SCHEMA)
        self.assertTrue(REQUIRED_KEYS.issubset(payload))
        self.assertIn(payload["worktree_write"], probe.WRITE_STATUSES)
        self.assertIn(payload["store_write"], probe.WRITE_STATUSES)
        self.assertIn(payload["proc_scan_status"], probe.PROC_STATUSES)
        self.assertIsInstance(payload["child_env_has_github_token"], bool)
        self.assertIsInstance(payload["parent_or_peer_proc_exposes_github_token_key"], bool)

    def test_payload_is_deterministic_for_the_same_facts(self):
        root = self.make_proc_root()
        self.write_environ(root, "3", probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        first = probe.build_payload(
            worktree=self.worktree, store=str(self.store), child_env={}, proc_root=root
        )
        second = probe.build_payload(
            worktree=self.worktree, store=str(self.store), child_env={}, proc_root=root
        )
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )

    def test_parent_visibility_flag_follows_the_match_count(self):
        root = self.make_proc_root()
        self.write_environ(root, "8", probe.GITHUB_TOKEN_MARKER + SECRET.encode())
        payload = self.build(proc_root=root)
        self.assertIs(payload["parent_or_peer_proc_exposes_github_token_key"], True)
        self.assertEqual(payload["proc_environ_token_key_matches"], 1)
        self.assertNotIn(SECRET, json.dumps(payload))

    def test_report_explains_results_without_claiming_a_fix(self):
        payload = self.build()
        report = probe.render_report(payload)
        self.assertIn("Diagnostic only", report)
        self.assertIn("no host-side fix", report)
        self.assertIn("did not apply, and does not claim, any host-side fix", report)
        self.assertNotIn(SECRET, report)
        self.assertNotIn(str(self.tmp), report)

    def test_summary_line_is_secret_and_path_free(self):
        payload = self.build(child_env={probe.GITHUB_TOKEN_KEY: SECRET})
        line = probe.summary_line(payload)
        self.assertNotIn(SECRET, line)
        self.assertNotIn(str(self.tmp), line)
        self.assertIn("worktree_write=", line)


# ---------------------------------------------------------------------------
# Entrypoint execution and allowlist wiring
# ---------------------------------------------------------------------------


class EntrypointTests(ProbeTestCase):
    def test_entrypoint_writes_diagnostic_artifacts(self):
        env = self.probe_env()
        completed = self.run_entrypoint(env)
        self.assertEqual(completed.returncode, probe.EXIT_OK, completed.stderr)

        payload = json.loads((self.tmp / "results" / "RESULT.json").read_text(encoding="utf-8"))
        self.assertTrue(REQUIRED_KEYS.issubset(payload))
        self.assertEqual(payload["schema"], probe.SCHEMA)
        self.assertIs(payload["diagnostic_only"], True)
        self.assertIs(payload["child_env_has_github_token"], False)
        self.assertIsInstance(payload["readable_proc_environments"], int)

        report = (self.tmp / "results" / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("CB16 Infra Boundary Probe", report)
        self.assertNotIn(str(self.worktree), report)
        self.assertNotIn(str(self.store), report)

        # Nothing may be left behind in the worktree or in the store.
        self.assertEqual(self.probe_leftovers(self.worktree), [])
        self.assertEqual(self.probe_leftovers(self.store), [])
        self.assertIn("BUILD_REPORT", completed.stdout)

    def test_entrypoint_fails_closed_when_its_own_environment_carries_the_token(self):
        env = self.probe_env(**{probe.GITHUB_TOKEN_KEY: SECRET})
        completed = self.run_entrypoint(env)
        self.assertEqual(completed.returncode, probe.EXIT_CREDENTIAL_BOUNDARY_VIOLATION)

        result_text = (self.tmp / "results" / "RESULT.json").read_text(encoding="utf-8")
        report_text = (self.tmp / "results" / "REPORT.md").read_text(encoding="utf-8")
        payload = json.loads(result_text)
        self.assertIs(payload["child_env_has_github_token"], True)
        self.assertIn("BOUNDARY VIOLATION", report_text)
        for text in (result_text, report_text, completed.stdout, completed.stderr):
            self.assertNotIn(SECRET, text)
        self.assertEqual(self.probe_leftovers(self.worktree), [])
        self.assertEqual(self.probe_leftovers(self.store), [])

    def test_entrypoint_has_no_shell_network_or_dsh_route(self):
        source = PROBE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "import subprocess",
            "import socket",
            "import urllib",
            "os.system",
            "os.popen",
            "shell=True",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertFalse([name for name in vars(probe) if "dsh" in name.lower()])

    def test_allowlist_entry_resolves_to_this_module(self):
        allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
        entry = allowlist["entrypoints"][probe.RESULT_COMMAND]
        self.assertIs(entry.get("diagnostic_only"), True)
        self.assertFalse(entry.get("dry_run_only"))
        self.assertEqual(sorted(entry["produces"]), ["REPORT.md", "RESULT.json"])
        self.assertEqual(entry["env"]["PYTHONPATH"], "science")
        for token in entry["argv"]:
            self.assertFalse(token.startswith("/"))
            self.assertNotIn("..", Path(token).parts)
        module = entry["argv"][-1]
        self.assertEqual(PROBE_PATH, REPO_ROOT / "science" / (module.replace(".", "/") + ".py"))

    def test_allowlist_entry_resolves_through_the_existing_science_lane(self):
        spec = importlib.util.spec_from_file_location(
            "cb16_dispatch_for_boundary_probe_test", REPO_ROOT / "scripts" / "cb16_dispatch.py"
        )
        dispatcher = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = dispatcher
        spec.loader.exec_module(dispatcher)

        allowlist = dispatcher.load_allowlist(ALLOWLIST_PATH)
        task = dispatcher.validate_metadata(
            {
                "mode": "science",
                "commit_sha": "0" * 40,
                "experiment_spec": "docs/tasks/INFRA_BOUNDARY_PROBE_R0.md",
                "result_command": probe.RESULT_COMMAND,
            },
            "science",
            allowlist=allowlist,
        )
        self.assertEqual(task.result_command, probe.RESULT_COMMAND)
        argv = dispatcher.resolve_allowlisted_argv(
            allowlist["entrypoints"][probe.RESULT_COMMAND], REPO_ROOT
        )
        self.assertEqual(argv, ["python3", "-m", "cb16_science.boundary_probe"])


if __name__ == "__main__":
    unittest.main()
