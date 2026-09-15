"""Deterministic tests for the ``cb16.normalization-ablation@v1`` experiment.

Covers the required evidence checks: the five candidate IDs are present in the
result schema, the three declared artifacts are written, the spec freezes the
contract values plus the resolved archive basenames and commit SHA, candidates
with invalid windows stay candidate-local, the interpretation table states the
frozen invertibility labels (N0/N1 true, N2/N3/N4 false), no winner is
promoted, and both new Science commands resolve through the repository
allowlist.

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

import numpy as np

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

from cb16_science.normalization import diagnostics, experiment  # noqa: E402
from cb16_science.normalization.contract import (  # noqa: E402
    CANDIDATES,
    CANDIDATE_IDS,
    CONTEXT_LENGTH,
    EPS_STD,
    EPS_V,
    INTERVAL_MONTHS,
    MAX_EVALUATION_WINDOWS,
    NO_WINNER_STATEMENT,
    PROJECTION_SEED,
    PROJECTION_SHAPE,
    RESULT_COMMAND,
    SCHEMA_RESULT,
    SCHEMA_SPEC,
    STATUS_INPUT_UNAVAILABLE,
    STATUS_OK,
)
from cb16_science.normalization.data import window_matrix  # noqa: E402
from cb16_science.normalization.reporting import (  # noqa: E402
    REPORT_FILENAME,
    RESULT_FILENAME,
    SPEC_FILENAME,
)

COMMIT_SHA = "a" * 40
INTERPRETATION_HEADING = "## Interpretation fields (not a winner)"


def _interpretation_rows(report: str):
    """Parse the REPORT.md interpretation table into per-candidate cell maps."""

    lines = report.splitlines()
    if INTERPRETATION_HEADING not in lines:
        raise AssertionError("report has no interpretation section")
    rows = {}
    for line in lines[lines.index(INTERPRETATION_HEADING) :]:
        if not line.startswith("| `"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = {
            "scale_invariant": cells[1],
            "ohlc_order": cells[2],
            "relative_vol_amplitude": cells[3],
            "invertible": cells[4],
            "removed": cells[5],
        }
    return rows


class SmallExperimentTests(unittest.TestCase):
    """In-process runs on a one-month fixture with a reduced window budget."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cb16-experiment-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        fixtures.build_klines_root(self.tmp / "klines", months=("2020-01",))
        self.root = self.tmp / "klines"
        self.result_dir = self.tmp / "results"

    def run_small(self):
        return experiment.run_experiment(
            result_dir=self.result_dir,
            klines_root=self.root,
            months=("2020-01",),
            max_windows=64,
            env={"CB16_COMMIT_SHA": COMMIT_SHA},
        )

    def test_experiment_writes_exactly_the_declared_artifacts(self):
        outcome = self.run_small()
        self.assertEqual(
            sorted(path.name for path in self.result_dir.iterdir()),
            sorted([SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME]),
        )
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.result["status"], STATUS_OK)

    def test_result_schema_contains_all_five_candidate_ids(self):
        outcome = self.run_small()
        result = outcome.result
        self.assertEqual(result["schema"], SCHEMA_RESULT)
        self.assertEqual(result["command"], RESULT_COMMAND)
        self.assertEqual(sorted(result["candidates"]), sorted(CANDIDATE_IDS))
        for candidate_id, candidate in result["candidates"].items():
            with self.subTest(candidate=candidate_id):
                self.assertEqual(candidate["candidate_id"], candidate_id)
                self.assertIn(candidate["status"], ("OK", "PARTIAL", "INVALID"))
                self.assertEqual(sorted(candidate["invariants"]), [
                    "candle_range_correlation",
                    "finite_output",
                    "no_lookahead",
                    "numerical_magnitude",
                    "ohlc_ordering",
                    "price_scale_invariance",
                    "runtime",
                    "volume_unit_invariance",
                ])
                self.assertEqual(candidate["output_shape"], [CONTEXT_LENGTH, 5])
                self.assertIn("sensory_projection", candidate)
                self.assertIn("interpretation", candidate)
        on_disk = json.loads((self.result_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk, result)

    def test_no_winner_is_promoted_and_the_report_says_so(self):
        outcome = self.run_small()
        result = outcome.result
        self.assertIs(result["no_winner_promoted"], True)
        self.assertTrue(result["scope_note"].startswith("2020-01-01..2020-03-31"))
        report = (self.result_dir / REPORT_FILENAME).read_text(encoding="utf-8")
        self.assertIn(NO_WINNER_STATEMENT, report)
        self.assertIn("not a profitability result", report)
        for candidate in result["candidates"].values():
            self.assertNotIn("score", candidate)
            self.assertNotIn("rank", candidate)

    def test_spec_freezes_the_contract_values_and_resolved_inputs(self):
        outcome = self.run_small()
        spec = outcome.spec
        on_disk = json.loads((self.result_dir / SPEC_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk, spec)
        self.assertEqual(spec["schema"], SCHEMA_SPEC)
        self.assertEqual(spec["commit_sha"], COMMIT_SHA)
        self.assertEqual(spec["resolved_inputs"]["archives"], ["BTCUSDT-1m-2020-01.zip"])
        fixed = spec["frozen_contract"]["fixed_data_scope"]
        self.assertEqual(fixed["context_length"], CONTEXT_LENGTH)
        self.assertEqual(fixed["max_evaluation_windows"], MAX_EVALUATION_WINDOWS)
        self.assertEqual(fixed["symbol"], "BTCUSDT")
        self.assertEqual(fixed["interval_months"], list(INTERVAL_MONTHS))
        numeric = spec["frozen_contract"]["numeric_contract"]
        self.assertEqual(numeric["eps_v"], EPS_V)
        self.assertEqual(numeric["eps_std"], EPS_STD)
        projection = spec["frozen_sensory_projection"]
        self.assertEqual(projection["seed"], PROJECTION_SEED)
        self.assertEqual(projection["shape"], list(PROJECTION_SHAPE))
        self.assertIs(projection["trained_or_mutated"], False)
        self.assertEqual(
            [candidate["candidate_id"] for candidate in spec["frozen_contract"]["candidates"]],
            list(CANDIDATE_IDS),
        )

    def test_spec_and_endpoint_digest_are_deterministic_across_runs(self):
        first = self.run_small()
        second_dir = self.tmp / "results-second"
        second = experiment.run_experiment(
            result_dir=second_dir,
            klines_root=self.root,
            months=("2020-01",),
            max_windows=64,
            env={"CB16_COMMIT_SHA": COMMIT_SHA},
        )
        self.assertEqual(first.spec, second.spec)
        self.assertEqual(
            first.result["window_selection"]["endpoint_sha256"],
            second.result["window_selection"]["endpoint_sha256"],
        )
        self.assertEqual(
            first.result["window_selection"]["selected_window_count"],
            second.result["window_selection"]["selected_window_count"],
        )
        self.assertEqual(
            first.result["candidates"]["N0"]["sensory_projection"]["checksum"],
            second.result["candidates"]["N0"]["sensory_projection"]["checksum"],
        )

    def test_result_is_deterministic_except_for_runtime_measurements(self):
        first = self.run_small().result
        second_dir = self.tmp / "results-third"
        second = experiment.run_experiment(
            result_dir=second_dir,
            klines_root=self.root,
            months=("2020-01",),
            max_windows=64,
            env={"CB16_COMMIT_SHA": COMMIT_SHA},
        ).result
        for result in (first, second):
            for candidate in result["candidates"].values():
                candidate["invariants"]["runtime"] = "<measured>"
        self.assertEqual(first, second)

    def test_window_selection_is_shared_and_spaced(self):
        outcome = self.run_small()
        selection = outcome.result["window_selection"]
        self.assertEqual(selection["context_length"], CONTEXT_LENGTH)
        self.assertEqual(selection["retained_predecessor_bars"], 1)
        self.assertEqual(selection["selected_window_count"], 64)
        self.assertTrue(selection["same_endpoints_for_all_candidates"])
        self.assertEqual(
            selection["candidate_valid_endpoints"],
            fixtures.month_minutes("2020-01") - CONTEXT_LENGTH,
        )

    def test_windows_come_from_the_read_only_tree_without_writing_to_it(self):
        before = sorted(path.name for path in (self.root / "BTCUSDT").iterdir())
        self.run_small()
        after = sorted(path.name for path in (self.root / "BTCUSDT").iterdir())
        self.assertEqual(before, after)


class CandidateInvalidWindowTests(unittest.TestCase):
    """A candidate-level sigma failure must stay candidate-local."""

    def test_near_zero_sigma_invalidates_one_window_for_n2_and_n4_only(self):
        series = fixtures.generate_ohlcv(160, seed=13)
        context_length = CONTEXT_LENGTH
        endpoints = np.array([64, 72, 80, 88], dtype=np.int64)
        raw_windows = window_matrix(series, endpoints, context_length)
        constant = raw_windows[1, -1, 3]
        raw_windows[1, :, 0] = constant
        raw_windows[1, :, 1] = constant
        raw_windows[1, :, 2] = constant
        raw_windows[1, :, 3] = constant
        matrix = diagnostics.projection_matrix()

        candidates = {candidate["candidate_id"]: candidate for candidate in CANDIDATES}
        for candidate_id in ("N2", "N4"):
            with self.subTest(candidate=candidate_id):
                evaluated = experiment.evaluate_candidate(
                    candidates[candidate_id],
                    raw_windows,
                    series,
                    endpoints,
                    context_length,
                    matrix,
                )
                self.assertEqual(evaluated["invalid_window_count"], 1)
                self.assertEqual(evaluated["valid_window_count"], 3)
                self.assertEqual(evaluated["status"], "PARTIAL")
                self.assertIn("eps_std", evaluated["invalid_window_reason"])
                self.assertLess(
                    evaluated["invariants"]["finite_output"]["finite_entry_rate"], 1.0
                )
        for candidate_id in ("N0", "N1", "N3"):
            with self.subTest(candidate=candidate_id):
                evaluated = experiment.evaluate_candidate(
                    candidates[candidate_id],
                    raw_windows,
                    series,
                    endpoints,
                    context_length,
                    matrix,
                )
                self.assertEqual(evaluated["invalid_window_count"], 0)
                self.assertEqual(evaluated["status"], "OK")


class EntrypointTests(unittest.TestCase):
    """Both new commands execute locally against fixtures."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="cb16-entrypoint-test-"))
        cls.root = fixtures.build_klines_root(cls.tmp / "klines")
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)

    def run_entrypoint(self, result_dir: Path, data_root: Path):
        env = support.entrypoint_env(
            result_dir, data_root=data_root, extra={"CB16_COMMIT_SHA": "b" * 40}
        )
        return subprocess.run(
            [sys.executable, "-m", "cb16_science.normalization"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

    def test_entrypoint_writes_the_three_artifacts_against_a_fixture(self):
        result_dir = self.tmp / "entrypoint-results"
        completed = self.run_entrypoint(result_dir, self.root)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("BUILD_REPORT", completed.stdout)
        self.assertEqual(
            sorted(path.name for path in result_dir.iterdir()),
            sorted([SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME]),
        )
        result = json.loads((result_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(sorted(result["candidates"]), sorted(CANDIDATE_IDS))
        self.assertEqual(result["dataset"]["archive_count"], 3)
        self.assertEqual(
            result["dataset"]["archives"],
            [f"BTCUSDT-1m-{month}.zip" for month in INTERVAL_MONTHS],
        )
        self.assertEqual(result["dataset"]["loaded_bar_count"], 131_040)
        self.assertEqual(result["dataset"]["csv_header_detected"], False)
        self.assertEqual(result["dataset"]["continuous_60s_spacing"], True)
        self.assertEqual(result["window_selection"]["selected_window_count"], 4096)
        report = (result_dir / REPORT_FILENAME).read_text(encoding="utf-8")
        self.assertIn(NO_WINNER_STATEMENT, report)
        spec = json.loads((result_dir / SPEC_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(spec["commit_sha"], "b" * 40)
        self.assertEqual(
            spec["resolved_inputs"]["archives"],
            [f"BTCUSDT-1m-{month}.zip" for month in INTERVAL_MONTHS],
        )

    def test_entrypoint_fails_closed_and_keeps_evidence_without_archives(self):
        empty_root = self.tmp / "empty-klines"
        (empty_root / "BTCUSDT").mkdir(parents=True)
        result_dir = self.tmp / "missing-input-results"
        completed = self.run_entrypoint(result_dir, empty_root)
        self.assertEqual(completed.returncode, experiment.EXIT_INPUT_UNAVAILABLE)
        result = json.loads((result_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(result["status"], STATUS_INPUT_UNAVAILABLE)
        self.assertEqual(result["dataset"]["archive_count"], 0)
        self.assertEqual(sorted(result["candidates"]), sorted(CANDIDATE_IDS))
        self.assertTrue(result["warnings"])
        report = (result_dir / REPORT_FILENAME).read_text(encoding="utf-8")
        self.assertIn(STATUS_INPUT_UNAVAILABLE, report)


class AllowlistTests(unittest.TestCase):
    def load_dispatcher(self):
        spec = importlib.util.spec_from_file_location(
            "cb16_dispatch_for_normalization_test", REPO_ROOT / "scripts" / "cb16_dispatch.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_allowlist_resolves_both_new_science_commands(self):
        dispatcher = self.load_dispatcher()
        allowlist = dispatcher.load_allowlist(ALLOWLIST_PATH)
        expectations = {
            "cb16.runtime-preflight@v1": {
                "argv": ["python3", "-m", "cb16_science.runtime_preflight"],
                "module": REPO_ROOT / "science" / "cb16_science" / "runtime_preflight.py",
                "produces": ["REPORT.md", "RESULT.json"],
            },
            "cb16.normalization-ablation@v1": {
                "argv": ["python3", "-m", "cb16_science.normalization"],
                "module": REPO_ROOT / "science" / "cb16_science" / "normalization" / "__main__.py",
                "produces": ["REPORT.md", "RESULT.json", "experiment_spec.json"],
            },
        }
        for command, expected in expectations.items():
            with self.subTest(command=command):
                entry = allowlist["entrypoints"][command]
                self.assertEqual(entry["argv"], expected["argv"])
                self.assertEqual(entry["env"]["PYTHONPATH"], "science")
                self.assertFalse(entry.get("dry_run_only"))
                self.assertEqual(sorted(entry["produces"]), sorted(expected["produces"]))
                self.assertTrue(expected["module"].is_file())
                for token in entry["argv"]:
                    self.assertFalse(token.startswith("/"))
                    self.assertNotIn("..", Path(token).parts)

                task = dispatcher.validate_metadata(
                    {
                        "mode": "science",
                        "commit_sha": "0" * 40,
                        "experiment_spec": "docs/experiments/frozen/R12_NORMALIZATION_ABLATION_R0.md",
                        "result_command": command,
                    },
                    "science",
                    allowlist=allowlist,
                )
                self.assertEqual(task.result_command, command)
                self.assertEqual(
                    dispatcher.resolve_allowlisted_argv(entry, REPO_ROOT), expected["argv"]
                )


class InterpretationTableTests(unittest.TestCase):
    """Focused interpretation labels: N0=true N1=true N2=false N3=false N4=false.

    The price channels of N2 and N4 are divided by a window-derived dispersion
    statistic, so their relative price path is no longer determined up to one
    positive common scale factor; the labels are declared from the fixed
    formulas, never fitted to the probes.
    """

    #: Frozen review table for
    #: ``invertible_to_relative_price_path_up_to_common_scale``.
    EXPECTED_INVERTIBLE = {
        "N0": True,
        "N1": True,
        "N2": False,
        "N3": False,
        "N4": False,
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="cb16-interpretation-test-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        cls.root = fixtures.build_klines_root(cls.tmp / "klines", months=("2020-01",))
        cls.result_dir = cls.tmp / "results"
        cls.outcome = experiment.run_experiment(
            result_dir=cls.result_dir,
            klines_root=cls.root,
            months=("2020-01",),
            max_windows=32,
            env={"CB16_COMMIT_SHA": COMMIT_SHA},
        )
        cls.report = (cls.result_dir / REPORT_FILENAME).read_text(encoding="utf-8")

    def test_declared_contract_invertibility_table(self):
        declared = {
            candidate["candidate_id"]: bool(
                candidate["invertible_to_relative_price_path_up_to_common_scale"]
            )
            for candidate in CANDIDATES
        }
        self.assertEqual(declared, self.EXPECTED_INVERTIBLE)

    def test_result_json_invertibility_table(self):
        emitted = {
            candidate_id: candidate["interpretation"][
                "invertible_to_relative_price_path_up_to_common_scale"
            ]
            for candidate_id, candidate in self.outcome.result["candidates"].items()
        }
        self.assertEqual(emitted, self.EXPECTED_INVERTIBLE)

    def test_report_interpretation_table_states_the_frozen_table(self):
        rows = _interpretation_rows(self.report)
        self.assertEqual(sorted(rows), sorted(self.EXPECTED_INVERTIBLE))
        for candidate_id, expected in self.EXPECTED_INVERTIBLE.items():
            with self.subTest(candidate=candidate_id):
                self.assertEqual(
                    rows[candidate_id]["invertible"], "true" if expected else "false"
                )

    def test_n2_and_n4_keep_volatility_amplitude_removed(self):
        for candidate_id in ("N2", "N4"):
            with self.subTest(candidate=candidate_id):
                removed = self.outcome.result["candidates"][candidate_id][
                    "interpretation"
                ]["known_information_removed"]
                self.assertEqual(
                    removed, ["window_location", "window_scale", "volatility_amplitude"]
                )
        removed_n3 = self.outcome.result["candidates"]["N3"]["interpretation"][
            "known_information_removed"
        ]
        self.assertIn("per_channel_relative_geometry", removed_n3)


class ContractAlignmentTests(unittest.TestCase):
    def test_projection_width_matches_the_frozen_context_length(self):
        self.assertEqual(PROJECTION_SHAPE, (32, CONTEXT_LENGTH * 5))

    def test_candidate_ids_match_the_contract_document(self):
        contract = (REPO_ROOT / "docs" / "experiments" / "frozen" / "R12_NORMALIZATION_ABLATION_R0.md").read_text(
            encoding="utf-8"
        )
        for candidate_id in CANDIDATE_IDS:
            with self.subTest(candidate=candidate_id):
                self.assertIn(f"Candidate {candidate_id} —", contract)
        self.assertIn("NO WINNER PROMOTED BY BUILDER", contract)


if __name__ == "__main__":
    unittest.main()
