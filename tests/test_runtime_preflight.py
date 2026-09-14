"""Deterministic tests for the ``cb16.runtime-preflight@v1`` entrypoint.

Covers the required evidence checks: bounded runtime facts, a missing PyTorch
reported rather than fatal, canonical-root and manifest booleans, the BTCUSDT
archive discovery/open probe, artifact writing, and that no secret environment
value or non-canonical host path reaches an artifact.

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = REPO_ROOT / "config" / "cb16_science_allowlist.json"
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))


def _load_support():
    path = Path(__file__).resolve().parent / "cb16_test_support.py"
    spec = importlib.util.spec_from_file_location("cb16_test_support", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


support = _load_support()
fixtures = support.fixtures

from cb16_science import runtime_preflight as preflight  # noqa: E402

SECRET_TOKEN = "ghp_not_a_real_secret_value_000000"
SECRET_KEY = "aws_not_a_real_secret_access_key_111111"


class PayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cb16-preflight-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_payload_has_the_required_bounded_sections(self):
        payload = preflight.build_payload(env={})
        self.assertEqual(payload["schema"], preflight.SCHEMA)
        self.assertEqual(payload["command"], preflight.RESULT_COMMAND)
        self.assertEqual(payload["status"], "OK")
        self.assertIs(payload["diagnostic_only"], True)
        self.assertIn("version", payload["python"])
        for section in ("installed", "version", "import_ok"):
            self.assertIn(section, payload["numpy"])
        for section in ("installed", "version", "cpu_import_ok", "cuda_available"):
            self.assertIn(section, payload["torch"])
        self.assertIsInstance(payload["torch"]["cuda_available"], bool)
        self.assertEqual(
            [entry["path"] for entry in payload["canonical_roots"]],
            ["/cb16/raw", "/cb16/frozen", "/cb16/brain_assets"],
        )
        for entry in payload["canonical_roots"]:
            self.assertIsInstance(entry["exists"], bool)
            self.assertIsInstance(entry["is_dir"], bool)
            self.assertIsInstance(entry["readable"], bool)

    def test_missing_torch_is_reported_not_fatal(self):
        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("No module named 'torch'")
            return importlib.import_module(name)

        with mock.patch.object(preflight.importlib, "import_module", side_effect=fake_import):
            section = preflight.torch_section()
        self.assertIs(section["installed"], False)
        self.assertIsNone(section["version"])
        self.assertIs(section["cpu_import_ok"], False)
        self.assertIs(section["cuda_available"], False)
        self.assertEqual(section["import_error_class"], "ModuleNotFoundError")

    def test_missing_numpy_is_reported_not_fatal(self):
        def fake_import(name, *args, **kwargs):
            if name == "numpy":
                raise ModuleNotFoundError("No module named 'numpy'")
            return importlib.import_module(name)

        with mock.patch.object(preflight.importlib, "import_module", side_effect=fake_import):
            section = preflight.numpy_section()
        self.assertIs(section["installed"], False)
        self.assertIs(section["import_ok"], False)

    def test_numpy_is_reported_on_this_host(self):
        section = preflight.numpy_section()
        self.assertIs(section["installed"], True)
        self.assertIs(section["import_ok"], True)
        self.assertTrue(section["version"])

    def test_manifest_inputs_are_booleans_with_canonical_paths_only(self):
        local = self.tmp / "local-input"
        local.mkdir()
        manifest = self.tmp / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "cb16.data.manifest.v1",
                    "read_only": {
                        "local_operator_path": {"path": str(local)},
                        "canonical_probe": {"path": "/cb16/cb16-preflight-test-absent"},
                    },
                }
            ),
            encoding="utf-8",
        )
        section = preflight.manifest_inputs_section({"CB16_DATA_MANIFEST": str(manifest)})
        self.assertIs(section["manifest_found"], True)
        self.assertEqual(section["manifest_source"], "env:CB16_DATA_MANIFEST")
        entries = {entry["logical_name"]: entry for entry in section["inputs"]}
        self.assertIs(entries["local_operator_path"]["exists"], True)
        self.assertIsNone(entries["local_operator_path"]["canonical_path"])
        self.assertEqual(
            entries["canonical_probe"]["canonical_path"], "/cb16/cb16-preflight-test-absent"
        )
        self.assertIs(entries["canonical_probe"]["exists"], False)
        rendered = json.dumps(section)
        self.assertNotIn(str(self.tmp), rendered)

    def test_absent_env_manifest_falls_back_to_the_repository_default(self):
        section = preflight.manifest_inputs_section(
            {"CB16_DATA_MANIFEST": str(self.tmp / "absent.json")}
        )
        self.assertIs(section["manifest_found"], True)
        self.assertEqual(section["manifest_source"], "repo_default")
        self.assertTrue(section["inputs"])

    def test_a_host_without_any_manifest_reports_absence_not_an_error(self):
        with mock.patch.object(
            preflight, "DEFAULT_MANIFEST_PATH", self.tmp / "absent-default.json"
        ):
            section = preflight.manifest_inputs_section(
                {"CB16_DATA_MANIFEST": str(self.tmp / "absent.json")}
            )
        self.assertIs(section["manifest_found"], False)
        self.assertEqual(section["manifest_source"], "absent")
        self.assertEqual(section["inputs"], [])

    def test_archive_probe_discovers_and_opens_a_fixture_archive(self):
        root = self.tmp / "klines"
        fixtures.write_month_archive(root, "BTCUSDT", "2020-01")
        section = preflight.probe_archives(root, "BTCUSDT", ("2020-01", "2020-02", "2020-03"))
        self.assertEqual(section["archive_count"], 1)
        self.assertEqual(section["basenames"], ["BTCUSDT-1m-2020-01.zip"])
        self.assertEqual(section["missing_months"], ["2020-02", "2020-03"])
        probe = section["probe"]
        self.assertIs(probe["opened"], True)
        self.assertIs(probe["member_is_csv"], True)
        self.assertIs(probe["header_detected"], False)
        self.assertIs(probe["valid_ohlcv_row_parsed"], True)
        self.assertGreater(probe["rows_inspected"], 0)

    def test_archive_probe_reports_a_missing_tree_without_raising(self):
        section = preflight.probe_archives(
            self.tmp / "absent", "BTCUSDT", ("2020-01", "2020-02", "2020-03")
        )
        self.assertEqual(section["archive_count"], 0)
        self.assertEqual(section["missing_months"], ["2020-01", "2020-02", "2020-03"])
        self.assertIsNone(section["probe"])

    def test_payload_contains_no_secret_environment_values(self):
        payload = preflight.build_payload(
            env={
                "CB16_GITHUB_TOKEN": SECRET_TOKEN,
                "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
                "HOME": "/home/cb16-secret-user",
            }
        )
        rendered = json.dumps(payload)
        self.assertNotIn(SECRET_TOKEN, rendered)
        self.assertNotIn(SECRET_KEY, rendered)
        self.assertNotIn("cb16-secret-user", rendered)

    def test_report_contains_only_canonical_paths(self):
        root = self.tmp / "klines"
        fixtures.write_month_archive(root, "BTCUSDT", "2020-01")
        payload = preflight.build_payload(env={"CB16_DATA_ROOT": str(root)})
        report = preflight.render_report(payload)
        self.assertNotIn(str(self.tmp), report)
        self.assertIn("A GPU is not required", report)
        self.assertIn("CB16 Runtime Preflight", report)


class EntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cb16-preflight-run-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "klines"
        fixtures.write_month_archive(self.root, "BTCUSDT", "2020-01")
        self.result_dir = self.tmp / "results"

    def run_entrypoint(self, **extra):
        env = support.entrypoint_env(self.result_dir, data_root=self.root, extra=extra)
        return subprocess.run(
            [sys.executable, "-m", "cb16_science.runtime_preflight"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_entrypoint_writes_result_and_report(self):
        completed = self.run_entrypoint()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("BUILD_REPORT", completed.stdout)
        self.assertEqual(
            sorted(path.name for path in self.result_dir.iterdir()),
            ["REPORT.md", "RESULT.json"],
        )
        payload = json.loads((self.result_dir / "RESULT.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], preflight.SCHEMA)
        self.assertEqual(payload["btcusdt_archives"]["archive_count"], 1)
        self.assertIs(payload["btcusdt_archives"]["probe"]["opened"], True)
        self.assertIs(payload["btcusdt_archives"]["probe"]["valid_ohlcv_row_parsed"], True)
        report = (self.result_dir / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("BTCUSDT-1m-2020-01.zip", report)

    def test_entrypoint_emits_no_secret_environment_values(self):
        # HOME is deliberately inherited: on this host NumPy lives in the user
        # site directory, and a preflight that changed HOME would be testing a
        # different failure than the one under test.
        completed = self.run_entrypoint(
            CB16_GITHUB_TOKEN=SECRET_TOKEN,
            AWS_SECRET_ACCESS_KEY=SECRET_KEY,
            CB16_OPERATOR_PRIVATE_PATH="/home/cb16-secret-user/private",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        texts = [
            completed.stdout,
            completed.stderr,
            (self.result_dir / "RESULT.json").read_text(encoding="utf-8"),
            (self.result_dir / "REPORT.md").read_text(encoding="utf-8"),
        ]
        for text in texts:
            for secret in (SECRET_TOKEN, SECRET_KEY, "cb16-secret-user"):
                with self.subTest(secret=secret[:8]):
                    self.assertNotIn(secret, text)
        # The data fixture lives outside /cb16 and must stay out of the report.
        self.assertNotIn(str(self.root), "\n".join(texts))

    def test_entrypoint_does_not_write_into_the_fixture_tree(self):
        before = sorted(path.name for path in (self.root / "BTCUSDT").iterdir())
        self.run_entrypoint()
        after = sorted(path.name for path in (self.root / "BTCUSDT").iterdir())
        self.assertEqual(before, after)


class AllowlistTests(unittest.TestCase):
    def test_allowlist_entry_resolves_to_this_module(self):
        allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
        entry = allowlist["entrypoints"][preflight.RESULT_COMMAND]
        self.assertIs(entry.get("diagnostic_only"), True)
        self.assertFalse(entry.get("dry_run_only"))
        self.assertEqual(entry["env"]["PYTHONPATH"], "science")
        self.assertEqual(sorted(entry["produces"]), ["REPORT.md", "RESULT.json"])
        module = entry["argv"][-1]
        self.assertEqual(
            Path(preflight.__file__).resolve(),
            REPO_ROOT / "science" / (module.replace(".", "/") + ".py"),
        )


class SourceBoundaryTests(unittest.TestCase):
    def test_preflight_never_dumps_the_environment(self):
        source = (REPO_ROOT / "science" / "cb16_science" / "runtime_preflight.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("os.environ.copy", "dict(os.environ)", "environ.items()", "print(os.environ"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
