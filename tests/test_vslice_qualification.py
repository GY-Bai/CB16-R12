"""R12 VS-C qualification-runner regression tests (contract requirements 13-20).

Required by ``docs/tasks/R12_VS_C_CONTROLLED_LEARNABILITY_R0.md`` section
"Required implementation tests":

13. PRE evaluation occurs before any learner update and POST only after exactly
    the preregistered generation count.
14. Positive/control learners for each paired seed begin with bitwise-identical
    Actor and Critic parameters.
15. Each training generation consumes only current-generation complete
    trajectories and increments exactly once.
16. Gate evaluator reproduces hand-computed Task-A and Task-B pass/fail
    examples.
17. Formal runner reads the committed JSON and offers no parameter override
    path.
18. Result serialization contains per-seed PRE/POST positive/control metrics and
    every gate component.
19. ``experiment_spec.json`` written to the result dir is byte-for-byte
    equivalent in parsed JSON content to the committed spec.
20. allowlist resolves to the frozen UV command and keeps the venv/cache under
    ``/tmp``, never in the read-only Science worktree.

The suite also covers the review delta ``exact_commit_implementation_test_gate``:
the formal runner must execute the repository suite for the exact commit, bind
the evidence to that commit and to the implementation-surface digest, and refuse
to produce a scientific verdict unless that gate is green.

The suite deliberately runs only reduced-budget smoke configurations and never
executes the formal preregistered eight-seed qualification; it also never runs
the repository suite from inside itself (the formal gate is exercised with
synthetic evidence).

    .venv/bin/python -m unittest tests.test_vslice_qualification -v
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:  # CPU PyTorch lives in the task venv; a bare interpreter records a skip.
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without PyTorch
    raise unittest.SkipTest(f"CPU PyTorch is required for the R12 VS-C runner: {exc}")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))

from cb16_science.vslice import controlled_tasks as controlled  # noqa: E402
from cb16_science.vslice import qualification  # noqa: E402
from cb16_science.vslice.contracts import ContractError  # noqa: E402
from cb16_science.vslice.learner import OnPolicyLearner  # noqa: E402

ALLOWLIST_PATH = REPO_ROOT / "config" / "cb16_science_allowlist.json"
SPEC_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_c_r0.json"
COMMAND = "cb16.vs-c-controlled-learnability@v1"


def committed_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def reduced_spec(seeds=(1201,), generations: int = 2) -> dict:
    """The committed spec with only the training budget and seed list reduced.

    Reduced budgets are implementation smoke fixtures only; no scientific
    threshold, control construction or task geometry is touched.
    """

    spec = copy.deepcopy(committed_spec())
    spec["task_a"]["generations"] = generations
    spec["task_b"]["generations"] = generations
    spec["paired_seeds"] = list(seeds)
    return spec


def seed_record(
    seed: int,
    *,
    positive_pass: bool,
    control_pass: bool,
    target_improvement: float,
    strictly_better: bool,
    growth_improvement: float,
    growth_difference: float,
) -> dict:
    return {
        "seed": seed,
        "task_a": {
            "positive": {"seed_gate": {"passed": positive_pass}},
            "control": {"seed_gate": {"passed": control_pass}},
            "paired": {
                "positive_target_mae_improvement": target_improvement,
                "positive_post_target_mae_strictly_better_than_control": strictly_better,
            },
        },
        "task_b": {
            "positive": {"seed_gate": {"passed": positive_pass}},
            "control": {"seed_gate": {"passed": control_pass}},
            "paired": {
                "positive_true_delayed_log_growth_improvement": growth_improvement,
                "positive_minus_control_true_delayed_log_growth": growth_difference,
            },
        },
    }


def green_implementation_tests(**overrides) -> dict:
    """Synthetic green exact-commit implementation-test evidence.

    Unit tests must never execute the repository suite from inside itself, so
    the gate is exercised with evidence carrying the same exact commit and
    implementation-surface digest the real runner records.
    """

    digest = qualification.implementation_source_sha256()
    evidence = {
        "status": "PASS",
        "passed": True,
        "runner_executed_suite": True,
        "required_by_global_gate": True,
        "commit_sha": qualification.commit_sha(),
        "git_head": qualification.repository_head(),
        "command": [sys.executable, *qualification.IMPLEMENTATION_TEST_ARGV],
        "exit_code": 0,
        "tests_run": 401,
        "skipped": None,
        "source_sha256": digest,
        "copy_sha256": digest,
        "source_unchanged": True,
        "copy_matches_source": True,
        "copied_file_count": 25,
        "git_snapshot": True,
        "stdout_tail": "Ran 401 tests in 60.000s\n\nOK\n",
        "stderr_tail": "",
        "preregistered_requirement": "synthetic evidence for implementation tests",
        "note": "synthetic evidence for implementation tests",
    }
    evidence.update(overrides)
    return evidence


class ProtocolTests(unittest.TestCase):
    def test_pre_and_post_bracket_exactly_the_preregistered_updates(self):
        """Requirements 13 and 15."""

        spec = reduced_spec(seeds=(1201,), generations=2)
        generations = spec["task_a"]["generations"]
        self.assertEqual(spec["task_b"]["generations"], generations)

        original_update = OnPolicyLearner.update
        original_evaluate_a = controlled.evaluate_task_a
        original_evaluate_b = controlled.evaluate_task_b
        events = []

        def update_spy(self, trajectories):
            batch = tuple(trajectories)
            before = self.generation_id
            report = original_update(self, batch)
            events.append(
                {
                    "kind": "update",
                    "learner": id(self),
                    "before": before,
                    "after": self.generation_id,
                    "batch_generations": sorted({step.generation_id for step in batch}),
                    "trajectory_count": len(batch),
                    "lengths": sorted({trajectory.length for trajectory in batch}),
                    "complete": all(trajectory.complete for trajectory in batch),
                    "truncated": any(trajectory.truncated for trajectory in batch),
                }
            )
            return report

        def evaluate_a_spy(learner, env):
            events.append(
                {"kind": "eval_a", "learner": id(learner), "generation": learner.generation_id}
            )
            return original_evaluate_a(learner, env)

        def evaluate_b_spy(learner, env):
            events.append(
                {"kind": "eval_b", "learner": id(learner), "generation": learner.generation_id}
            )
            return original_evaluate_b(learner, env)

        with mock.patch.object(OnPolicyLearner, "update", update_spy), mock.patch.object(
            controlled, "evaluate_task_a", evaluate_a_spy
        ), mock.patch.object(controlled, "evaluate_task_b", evaluate_b_spy):
            qualification.run_experiment(
                spec, seeds=(1201,), implementation_tests=green_implementation_tests()
            )

        updates = [event for event in events if event["kind"] == "update"]
        evaluations = [event for event in events if event["kind"].startswith("eval")]
        self.assertEqual(len(updates), 4 * generations)
        self.assertEqual(len(evaluations), 4 * 2)

        by_learner = {}
        for event in updates:
            by_learner.setdefault(event["learner"], []).append(event)
        self.assertEqual(len(by_learner), 4)
        for learner_events in by_learner.values():
            self.assertEqual([event["before"] for event in learner_events], [0, 1])
            self.assertEqual([event["after"] for event in learner_events], [1, 2])
            for event in learner_events:
                self.assertEqual(event["batch_generations"], [event["before"]])
                self.assertTrue(event["complete"])
                self.assertFalse(event["truncated"])
                if event["lengths"] == [1]:
                    self.assertEqual(event["trajectory_count"], 90)
                else:
                    self.assertEqual(event["lengths"], [2])
                    self.assertEqual(event["trajectory_count"], 128)
        self.assertEqual(sum(1 for event in updates if event["lengths"] == [1]), 4)
        self.assertEqual(sum(1 for event in updates if event["lengths"] == [2]), 4)

        evaluations_by_learner = {}
        for event in evaluations:
            evaluations_by_learner.setdefault(event["learner"], []).append(event)
        self.assertEqual(len(evaluations_by_learner), 4)
        for learner_events in evaluations_by_learner.values():
            self.assertEqual([event["generation"] for event in learner_events], [0, generations])

        first_update, last_update, first_eval, last_eval = {}, {}, {}, {}
        for index, event in enumerate(events):
            if event["kind"] == "update":
                first_update.setdefault(event["learner"], index)
                last_update[event["learner"]] = index
            else:
                first_eval.setdefault(event["learner"], index)
                last_eval[event["learner"]] = index
        self.assertEqual(set(first_update), set(first_eval))
        for learner in first_update:
            self.assertLess(first_eval[learner], first_update[learner])
            self.assertGreater(last_eval[learner], last_update[learner])

    def test_paired_learners_start_bitwise_identical(self):
        """Requirement 14."""

        spec = committed_spec()
        authority = qualification.parse_authority(spec)
        optimizer = qualification.parse_optimizer(spec)

        torch.manual_seed(1201)
        positive = qualification.build_learner(authority, optimizer)
        torch.manual_seed(1201)
        control = qualification.build_learner(authority, optimizer)
        qualification.require_identical_parameters(positive, control)
        for left, right in zip(positive.actor.parameters(), control.actor.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(positive.critic.parameters(), control.critic.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(
            torch.equal(positive.sensory.projection, control.sensory.projection)
        )

        torch.manual_seed(1202)
        other = qualification.build_learner(authority, optimizer)
        with self.assertRaises(ContractError):
            qualification.require_identical_parameters(positive, other)


class GateEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = committed_spec()

    def test_task_a_seed_gate_matches_hand_computed_examples(self):
        """Requirement 16 (Task A seed gate)."""

        gate = qualification.gate_section(self.spec, "task_a", "seed_pass_gate")
        passing = qualification.task_a_seed_gate(
            {"direction_correct_count": 3, "target_exposure_mae": 0.15}, gate
        )
        self.assertTrue(passing["passed"])
        self.assertEqual(
            passing["preregistered"],
            {"direction_correct_count_min": 3, "target_exposure_mae_max": 0.15},
        )
        self.assertTrue(all(passing["conditions"].values()))
        self.assertFalse(
            qualification.task_a_seed_gate(
                {"direction_correct_count": 3, "target_exposure_mae": 0.1500001}, gate
            )["passed"]
        )
        self.assertFalse(
            qualification.task_a_seed_gate(
                {"direction_correct_count": 2, "target_exposure_mae": 0.0}, gate
            )["passed"]
        )

    def test_task_a_aggregate_gate_matches_hand_computed_examples(self):
        """Requirement 16 (Task A aggregate gate)."""

        gate = qualification.gate_section(self.spec, "task_a", "aggregate_gate")
        improvements = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
        records = [
            seed_record(
                seed,
                positive_pass=seed < 6,
                control_pass=seed < 2,
                target_improvement=improvement,
                strictly_better=seed < 6,
                growth_improvement=0.0,
                growth_difference=0.0,
            )
            for seed, improvement in enumerate(improvements)
        ]
        result = qualification.task_a_aggregate_gate(records, gate)
        self.assertEqual(result["components"]["positive_seed_pass_count"], 6)
        self.assertEqual(result["components"]["control_seed_pass_count"], 2)
        self.assertAlmostEqual(
            result["components"]["positive_median_pre_to_post_target_mae_improvement"], 0.17
        )
        self.assertEqual(
            result["components"]["paired_positive_target_mae_strictly_better_than_control_count"], 6
        )
        self.assertTrue(result["passed"])

        fewer_positive = [dict(record) for record in records]
        for seed in (5, 6):
            fewer_positive[seed] = seed_record(
                seed,
                positive_pass=False,
                control_pass=False,
                target_improvement=0.20,
                strictly_better=True,
                growth_improvement=0.0,
                growth_difference=0.0,
            )
        self.assertFalse(qualification.task_a_aggregate_gate(fewer_positive, gate)["passed"])

        more_controls = [dict(record) for record in records]
        more_controls[2] = seed_record(
            2,
            positive_pass=True,
            control_pass=True,
            target_improvement=0.14,
            strictly_better=True,
            growth_improvement=0.0,
            growth_difference=0.0,
        )
        self.assertFalse(qualification.task_a_aggregate_gate(more_controls, gate)["passed"])

        weak_improvement = [
            seed_record(
                seed,
                positive_pass=seed < 6,
                control_pass=seed < 2,
                target_improvement=improvement,
                strictly_better=seed < 6,
                growth_improvement=0.0,
                growth_difference=0.0,
            )
            for seed, improvement in enumerate([0.02, 0.06, 0.10, 0.14, 0.14, 0.16, 0.20, 0.24])
        ]
        self.assertFalse(
            qualification.task_a_aggregate_gate(weak_improvement, gate)["passed"]
        )

        weak_pairing = [
            seed_record(
                seed,
                positive_pass=seed < 6,
                control_pass=seed < 2,
                target_improvement=improvement,
                strictly_better=seed < 5,
                growth_improvement=0.0,
                growth_difference=0.0,
            )
            for seed, improvement in enumerate(improvements)
        ]
        self.assertFalse(qualification.task_a_aggregate_gate(weak_pairing, gate)["passed"])

    def test_task_b_seed_gate_matches_hand_computed_examples(self):
        """Requirement 16 (Task B seed gate)."""

        gate = qualification.gate_section(self.spec, "task_b", "seed_pass_gate")

        def metrics(up_direction, down_direction, up_risk, down_risk, growth):
            return {
                "up": {"direction": up_direction, "requested_risk": up_risk},
                "down": {"direction": down_direction, "requested_risk": down_risk},
                "mean_true_delayed_log_growth": growth,
            }

        self.assertTrue(
            qualification.task_b_seed_gate(
                metrics("LONG", "SHORT", 0.5, 0.5, 0.048790164169432), gate
            )["passed"]
        )
        self.assertFalse(
            qualification.task_b_seed_gate(metrics("FLAT", "SHORT", 0.5, 0.5, 0.05), gate)["passed"]
        )
        self.assertFalse(
            qualification.task_b_seed_gate(metrics("LONG", "LONG", 0.5, 0.5, 0.05), gate)["passed"]
        )
        self.assertFalse(
            qualification.task_b_seed_gate(
                metrics("LONG", "SHORT", 0.4999, 0.5, 0.05), gate
            )["passed"]
        )
        self.assertFalse(
            qualification.task_b_seed_gate(
                metrics("LONG", "SHORT", 0.5, 0.4999, 0.05), gate
            )["passed"]
        )
        self.assertFalse(
            qualification.task_b_seed_gate(
                metrics("LONG", "SHORT", 0.5, 0.5, 0.0299999), gate
            )["passed"]
        )
        boundary = qualification.task_b_seed_gate(
            metrics("LONG", "SHORT", 0.5, 0.5, 0.03), gate
        )
        self.assertTrue(boundary["passed"])
        self.assertEqual(boundary["preregistered"]["up_requested_risk_min"], 0.5)
        self.assertEqual(boundary["preregistered"]["mean_true_delayed_log_growth_min"], 0.03)

    def test_task_b_aggregate_gate_matches_hand_computed_examples(self):
        """Requirement 16 (Task B aggregate gate)."""

        gate = qualification.gate_section(self.spec, "task_b", "aggregate_gate")
        improvements = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        differences = [0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06]
        records = [
            seed_record(
                seed,
                positive_pass=seed < 7,
                control_pass=seed < 1,
                target_improvement=0.0,
                strictly_better=False,
                growth_improvement=improvement,
                growth_difference=difference,
            )
            for seed, (improvement, difference) in enumerate(zip(improvements, differences))
        ]
        result = qualification.task_b_aggregate_gate(records, gate)
        self.assertEqual(result["components"]["positive_seed_pass_count"], 7)
        self.assertEqual(result["components"]["control_seed_pass_count"], 1)
        self.assertAlmostEqual(
            result["components"][
                "positive_median_pre_to_post_true_delayed_log_growth_improvement"
            ],
            0.045,
        )
        self.assertEqual(result["components"]["paired_positive_growth_minus_control_count"], 7)
        self.assertTrue(result["passed"])

        fewer_positive = [dict(record) for record in records]
        for seed in (5, 6):
            fewer_positive[seed] = seed_record(
                seed,
                positive_pass=False,
                control_pass=False,
                target_improvement=0.0,
                strictly_better=False,
                growth_improvement=0.07,
                growth_difference=0.05,
            )
        self.assertFalse(qualification.task_b_aggregate_gate(fewer_positive, gate)["passed"])

        weak_differences = [
            seed_record(
                seed,
                positive_pass=seed < 7,
                control_pass=seed < 1,
                target_improvement=0.0,
                strictly_better=False,
                growth_improvement=improvement,
                growth_difference=difference,
            )
            for seed, (improvement, difference) in enumerate(
                zip(improvements, [0.0, 0.0, 0.0, 0.02, 0.02, 0.02, 0.02, 0.03])
            )
        ]
        self.assertFalse(
            qualification.task_b_aggregate_gate(weak_differences, gate)["passed"]
        )

        weak_improvement = [
            seed_record(
                seed,
                positive_pass=seed < 7,
                control_pass=seed < 1,
                target_improvement=0.0,
                strictly_better=False,
                growth_improvement=0.0,
                growth_difference=difference,
            )
            for seed, difference in enumerate(differences)
        ]
        self.assertFalse(
            qualification.task_b_aggregate_gate(weak_improvement, gate)["passed"]
        )


class ImplementationTestGateTests(unittest.TestCase):
    """Review delta ``exact_commit_implementation_test_gate``."""

    def test_unittest_summary_parsing(self):
        green = qualification.parse_unittest_summary(
            "....\n----------------------------------------------------------------------\n"
            "Ran 401 tests in 66.072s\n\nOK\n"
        )
        self.assertEqual(green["tests_run"], 401)
        self.assertIsNone(green["skipped"])
        self.assertTrue(green["unittest_ok"])

        skipped = qualification.parse_unittest_summary("Ran 5 tests in 0.100s\n\nOK (skipped=1)\n")
        self.assertEqual(skipped["tests_run"], 5)
        self.assertEqual(skipped["skipped"], 1)
        self.assertTrue(skipped["unittest_ok"])

        failed = qualification.parse_unittest_summary(
            "Ran 5 tests in 0.100s\n\nFAILED (errors=1, skipped=2)\n"
        )
        self.assertEqual(failed["tests_run"], 5)
        self.assertEqual(failed["skipped"], 2)
        self.assertFalse(failed["unittest_ok"])

        self.assertIsNone(qualification.parse_unittest_summary("")["tests_run"])

    def test_unittest_run_status_reads_both_streams(self):
        """unittest reports on stderr; a zero exit code alone is not green."""

        stderr_only = qualification.summarise_unittest_run(
            0, "", "...\nRan 408 tests in 65.066s\n\nOK\n"
        )
        self.assertEqual(stderr_only["status"], "PASS")
        self.assertEqual(stderr_only["tests_run"], 408)
        self.assertTrue(stderr_only["unittest_ok"])

        stdout_only = qualification.summarise_unittest_run(
            0, "....\nRan 3 tests in 0.100s\n\nOK\n", ""
        )
        self.assertEqual(stdout_only["status"], "PASS")

        no_summary = qualification.summarise_unittest_run(0, "", "")
        self.assertEqual(no_summary["status"], "FAIL")
        self.assertFalse(no_summary["unittest_ok"])

        failed = qualification.summarise_unittest_run(
            1, "", "Ran 3 tests in 0.100s\n\nFAILED (failures=1)\n"
        )
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(failed["exit_code"], 1)

    def test_implementation_copy_is_byte_identical_and_excludes_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            copied = qualification.copy_implementation_tree(qualification.REPO_ROOT, root)
            self.assertGreater(copied, 0)
            for excluded in (".git", ".venv", ".cb16", ".pip-cache", ".uv-cache"):
                self.assertFalse((root / excluded).exists(), excluded)
            self.assertFalse(any(path.name == "__pycache__" for path in root.rglob("*")))
            self.assertEqual(
                qualification.implementation_source_sha256(root),
                qualification.implementation_source_sha256(qualification.REPO_ROOT),
            )

    def test_repository_head_is_recorded_as_best_effort_evidence(self):
        head = qualification.repository_head()
        self.assertTrue(head is None or (len(head) == 40 and set(head) <= set("0123456789abcdef")))

    def test_implementation_surface_covers_code_config_and_spec(self):
        files = {
            path.relative_to(qualification.REPO_ROOT).as_posix()
            for path in qualification.implementation_surface_files(qualification.REPO_ROOT)
        }
        self.assertIn("science/cb16_science/vslice/qualification.py", files)
        self.assertIn("tests/test_vslice_qualification.py", files)
        self.assertIn("scripts/cb16_dispatch.py", files)
        self.assertIn("config/experiments/r12_vs_c_r0.json", files)
        self.assertIn("config/cb16_science_allowlist.json", files)
        self.assertFalse(any("__pycache__" in name for name in files))
        self.assertFalse(any(name.endswith(".pyc") for name in files))

    def test_implementation_test_gate_requires_a_green_exact_commit_suite(self):
        commit = qualification.commit_sha()
        gate = qualification.implementation_test_gate(
            green_implementation_tests(commit_sha=commit), commit=commit
        )
        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["conditions"].values()))
        self.assertEqual(gate["components"]["tests_run"], 401)

        variants = {
            "suite_green": green_implementation_tests(commit_sha=commit, status="FAIL"),
            "exact_commit": green_implementation_tests(commit_sha="0" * 40),
            "copy_matches_source": green_implementation_tests(
                commit_sha=commit, copy_matches_source=False
            ),
            "source_unchanged": green_implementation_tests(
                commit_sha=commit, source_unchanged=False
            ),
            "source_sha256_matches_current_tree": green_implementation_tests(
                commit_sha=commit, source_sha256="0" * 64
            ),
        }
        for unmet, variant in variants.items():
            with self.subTest(unmet=unmet):
                result = qualification.implementation_test_gate(variant, commit=commit)
                self.assertFalse(result["passed"])
                self.assertFalse(result["conditions"][unmet])
                self.assertTrue(
                    all(value for key, value in result["conditions"].items() if key != unmet)
                )

    def test_implementation_failure_taxonomy(self):
        """Executed red suite -> CONTRACT_MISMATCH; unexecutable suite -> EXECUTION_BLOCKED."""

        commit = qualification.commit_sha()
        red = green_implementation_tests(commit_sha=commit, status="FAIL", exit_code=1)
        timeout = green_implementation_tests(
            commit_sha=commit, status="TIMEOUT", exit_code=124, passed=False
        )
        not_executable = green_implementation_tests(
            commit_sha=commit, status="NOT_EXECUTABLE", exit_code=126, passed=False
        )
        no_verdict = green_implementation_tests(commit_sha=commit, status=None, exit_code=None)

        self.assertEqual(
            qualification.implementation_failure_classification(red),
            ("CONTRACT_MISMATCH", qualification.EXIT_CONTRACT_MISMATCH),
        )
        for evidence in (timeout, not_executable, no_verdict):
            with self.subTest(status=evidence["status"]):
                self.assertEqual(
                    qualification.implementation_failure_classification(evidence),
                    ("EXECUTION_BLOCKED", qualification.EXIT_EXECUTION_BLOCKED),
                )

        red_detail = qualification.implementation_test_failure_detail("CONTRACT_MISMATCH", red)
        blocked_detail = qualification.implementation_test_failure_detail(
            "EXECUTION_BLOCKED", timeout
        )
        self.assertIn("ran and did not pass", red_detail)
        self.assertIn("could not be executed", blocked_detail)
        self.assertIn("TIMEOUT", blocked_detail)

    def test_run_experiment_refuses_unverified_implementation_tests(self):
        spec = reduced_spec(seeds=(1201,), generations=1)
        red_variants = (
            green_implementation_tests(status="FAIL"),
            green_implementation_tests(commit_sha="0" * 40),
            green_implementation_tests(source_unchanged=False),
            green_implementation_tests(copy_matches_source=False),
            green_implementation_tests(source_sha256="0" * 64),
        )
        for evidence in red_variants:
            with self.subTest(unmet=evidence):
                with self.assertRaises(ContractError):
                    qualification.run_experiment(
                        spec, seeds=(1201,), implementation_tests=evidence
                    )


class RunnerTests(unittest.TestCase):
    def test_formal_runner_has_no_parameter_override_path(self):
        """Requirement 17."""

        self.assertEqual(qualification.CONFIG_PATH, SPEC_PATH)
        self.assertEqual(list(inspect.signature(qualification.main).parameters), [])
        self.assertEqual(
            list(inspect.signature(qualification.run_qualification).parameters), ["result_dir"]
        )
        source = (
            REPO_ROOT / "science" / "cb16_science" / "vslice" / "qualification.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("argparse", source)
        self.assertNotIn("sys.argv", source)
        self.assertIn("load_spec()", source)
        self.assertEqual(qualification.load_spec(), committed_spec())
        self.assertTrue(qualification.matches_committed_spec(committed_spec()))

    def test_formal_entrypoint_requires_result_dir_and_commit_sha(self):
        """Requirement 17 (the formal inputs cannot be invented)."""

        base_env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("CB16_RESULT_DIR", "CB16_COMMIT_SHA")
        }
        base_env["CB16_RESULT_DIR"] = str(Path(tempfile.gettempdir()) / "cb16-vs-c-required-env")
        with mock.patch.dict(os.environ, base_env, clear=True):
            with self.assertRaises(ContractError):
                qualification.required_environment()
            os.environ["CB16_RESULT_DIR"] = base_env["CB16_RESULT_DIR"]
            os.environ["CB16_COMMIT_SHA"] = "0" * 40
            result_dir, commit = qualification.required_environment()
            self.assertEqual(commit, "0" * 40)
            self.assertEqual(result_dir, Path(base_env["CB16_RESULT_DIR"]))
        with mock.patch.dict(os.environ, {"CB16_COMMIT_SHA": "0" * 40}, clear=True):
            with self.assertRaises(ContractError):
                qualification.required_environment()

    def test_formal_runner_fails_closed_when_the_exact_commit_suite_is_red(self):
        """The exact-commit implementation test gate blocks the science run."""

        red = green_implementation_tests(
            status="FAIL", passed=False, exit_code=1, tests_run=401
        )
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            with mock.patch.dict(
                os.environ, {"CB16_COMMIT_SHA": "0" * 40}, clear=False
            ), mock.patch.object(
                qualification, "run_implementation_test_suite", return_value=red
            ), mock.patch.object(
                qualification,
                "run_experiment",
                side_effect=AssertionError("science must not run on a red suite"),
            ) as experiment:
                outcome = qualification.run_qualification(result_dir=result_dir)
            experiment.assert_not_called()

            self.assertEqual(outcome.classification, "CONTRACT_MISMATCH")
            self.assertEqual(outcome.exit_code, qualification.EXIT_CONTRACT_MISMATCH)
            for name in (
                qualification.SPEC_FILENAME,
                qualification.RESULT_FILENAME,
                qualification.REPORT_FILENAME,
            ):
                self.assertTrue((result_dir / name).is_file(), name)
            result = json.loads(
                (result_dir / qualification.RESULT_FILENAME).read_text(encoding="utf-8")
            )
            written_spec = json.loads(
                (result_dir / qualification.SPEC_FILENAME).read_text(encoding="utf-8")
            )
            report = (result_dir / qualification.REPORT_FILENAME).read_text(encoding="utf-8")

        self.assertEqual(result["classification"], "CONTRACT_MISMATCH")
        self.assertEqual(result["implementation_tests"]["status"], "FAIL")
        self.assertFalse(result["implementation_tests"]["gate"]["passed"])
        self.assertEqual(
            result["verdicts"],
            {
                "task_a": "NOT_EVALUATED",
                "task_b": "NOT_EVALUATED",
                "global": "CONTRACT_MISMATCH",
            },
        )
        self.assertEqual(result["seeds"], [])
        self.assertEqual(result["aggregate_gates"], {})
        self.assertEqual(written_spec, committed_spec())
        self.assertIn("CONTRACT_MISMATCH", report)
        for section in (
            "## Implementation correctness",
            "## Task A controlled learnability",
            "## Task B delayed-credit learnability",
            "## Negative controls",
            "## Scientific verdict",
            "## Limitations",
        ):
            self.assertIn(section, report)

    def test_formal_runner_classifies_unexecutable_suite_as_execution_blocked(self):
        """Timeout / missing executable are execution/environment blockers, not contract failures."""

        blockers = (
            green_implementation_tests(
                status="TIMEOUT",
                passed=False,
                exit_code=124,
                tests_run=None,
                stdout_tail="",
                stderr_tail="",
            ),
            green_implementation_tests(
                status="NOT_EXECUTABLE",
                passed=False,
                exit_code=126,
                tests_run=None,
                stderr_tail="FileNotFoundError: [Errno 2] No such file or directory",
            ),
        )
        for evidence in blockers:
            with self.subTest(status=evidence["status"]):
                with tempfile.TemporaryDirectory() as tmp:
                    result_dir = Path(tmp) / "result"
                    with mock.patch.dict(
                        os.environ, {"CB16_COMMIT_SHA": "0" * 40}, clear=False
                    ), mock.patch.object(
                        qualification, "run_implementation_test_suite", return_value=evidence
                    ), mock.patch.object(
                        qualification,
                        "run_experiment",
                        side_effect=AssertionError("science must not run without a suite verdict"),
                    ) as experiment:
                        outcome = qualification.run_qualification(result_dir=result_dir)
                    experiment.assert_not_called()

                    self.assertEqual(outcome.classification, "EXECUTION_BLOCKED")
                    self.assertEqual(outcome.exit_code, qualification.EXIT_EXECUTION_BLOCKED)
                    for name in (
                        qualification.SPEC_FILENAME,
                        qualification.RESULT_FILENAME,
                        qualification.REPORT_FILENAME,
                    ):
                        self.assertTrue((result_dir / name).is_file(), name)
                    result = json.loads(
                        (result_dir / qualification.RESULT_FILENAME).read_text(encoding="utf-8")
                    )
                    written_spec = json.loads(
                        (result_dir / qualification.SPEC_FILENAME).read_text(encoding="utf-8")
                    )
                    report = (result_dir / qualification.REPORT_FILENAME).read_text(
                        encoding="utf-8"
                    )

                self.assertEqual(result["classification"], "EXECUTION_BLOCKED")
                self.assertEqual(result["error"]["type"], "EXECUTION_BLOCKED")
                self.assertEqual(result["implementation_tests"]["status"], evidence["status"])
                self.assertEqual(
                    result["implementation_tests"]["failure_classification"],
                    "EXECUTION_BLOCKED",
                )
                self.assertFalse(result["implementation_tests"]["gate"]["passed"])
                self.assertEqual(
                    result["verdicts"],
                    {
                        "task_a": "NOT_EVALUATED",
                        "task_b": "NOT_EVALUATED",
                        "global": "EXECUTION_BLOCKED",
                    },
                )
                self.assertEqual(result["seeds"], [])
                self.assertEqual(result["aggregate_gates"], {})
                self.assertEqual(written_spec, committed_spec())
                self.assertIn("EXECUTION_BLOCKED", report)
                self.assertIn("could not be executed", report)
                for section in (
                    "## Implementation correctness",
                    "## Task A controlled learnability",
                    "## Task B delayed-credit learnability",
                    "## Negative controls",
                    "## Scientific verdict",
                    "## Limitations",
                ):
                    self.assertIn(section, report)

    def test_formal_runner_passes_its_own_test_evidence_into_the_experiment(self):
        """The runner produces the evidence itself; the experiment receives it."""

        commit = qualification.commit_sha()
        evidence = green_implementation_tests(commit_sha=commit)
        captured = {}

        def fake_experiment(spec, *, seeds, implementation_tests):
            captured["spec"] = spec
            captured["seeds"] = tuple(seeds)
            captured["implementation_tests"] = implementation_tests
            return qualification.QualificationOutcome(
                spec=dict(spec),
                result={"schema": "cb16.result.v1", "classification": "PASS"},
                report="# stub\n",
                classification="PASS",
                exit_code=0,
                summary_line="stub",
            )

        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            with mock.patch.object(
                qualification, "run_implementation_test_suite", return_value=evidence
            ), mock.patch.object(
                qualification, "run_experiment", side_effect=fake_experiment
            ):
                outcome = qualification.run_qualification(result_dir=result_dir)
            self.assertTrue((result_dir / qualification.RESULT_FILENAME).is_file())

        self.assertEqual(outcome.classification, "PASS")
        self.assertIs(captured["implementation_tests"], evidence)
        self.assertEqual(captured["spec"], committed_spec())
        self.assertEqual(
            captured["seeds"], tuple(qualification.paired_seeds(committed_spec()))
        )

    def test_experiment_spec_artifact_matches_the_committed_spec(self):
        """Requirement 19."""

        outcome = qualification.QualificationOutcome(
            spec=qualification.load_spec(),
            result={"schema": "cb16.result.v1"},
            report="# smoke\n",
            classification="NOT_RUN",
            exit_code=0,
            summary_line="smoke",
        )
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            qualification.write_artifacts(result_dir, outcome)
            written = json.loads(
                (result_dir / qualification.SPEC_FILENAME).read_text(encoding="utf-8")
            )
        self.assertEqual(written, committed_spec())

    def test_repeated_runs_are_bitwise_identical(self):
        """No wall-clock or process entropy may change a controlled run."""

        spec = reduced_spec(seeds=(1201,), generations=2)
        first = qualification.run_experiment(
            spec, seeds=(1201,), implementation_tests=green_implementation_tests()
        )
        second = qualification.run_experiment(
            spec, seeds=(1201,), implementation_tests=green_implementation_tests()
        )
        self.assertEqual(
            json.dumps(first.result["seeds"], sort_keys=True),
            json.dumps(second.result["seeds"], sort_keys=True),
        )
        self.assertEqual(
            json.dumps(first.result["aggregate_gates"], sort_keys=True),
            json.dumps(second.result["aggregate_gates"], sort_keys=True),
        )
        self.assertEqual(first.classification, second.classification)

    def test_reduced_run_serializes_every_gate_component(self):
        """Requirement 18 (with requirement 13's PRE/POST protocol)."""

        spec = reduced_spec(seeds=(1201, 1202), generations=2)
        outcome = qualification.run_experiment(
            spec, seeds=(1201, 1202), implementation_tests=green_implementation_tests()
        )
        self.assertIn(outcome.classification, {"PASS", "SCIENTIFIC_FAIL"})
        self.assertEqual(outcome.exit_code, 0 if outcome.classification == "PASS" else 1)

        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            qualification.write_artifacts(result_dir, outcome)
            for name in (
                qualification.SPEC_FILENAME,
                qualification.RESULT_FILENAME,
                qualification.REPORT_FILENAME,
            ):
                self.assertTrue((result_dir / name).is_file(), name)
            written_spec = json.loads(
                (result_dir / qualification.SPEC_FILENAME).read_text(encoding="utf-8")
            )
            result = json.loads(
                (result_dir / qualification.RESULT_FILENAME).read_text(encoding="utf-8")
            )
            report = (result_dir / qualification.REPORT_FILENAME).read_text(encoding="utf-8")

        self.assertEqual(written_spec, spec)
        self.assertEqual(result["experiment_id"], qualification.EXPERIMENT_ID)
        self.assertEqual(result["schema"], "cb16.result.v1")
        self.assertEqual(result["runtime"]["device"], "cpu")
        self.assertEqual(result["runtime"]["torch_num_threads"], 1)
        self.assertTrue(result["runtime"]["deterministic_algorithms"])
        self.assertEqual(result["implementation_tests"]["status"], "PASS")
        self.assertTrue(result["implementation_tests"]["runner_executed_suite"])
        self.assertTrue(result["implementation_tests"]["gate"]["passed"])
        self.assertEqual(
            set(result["implementation_tests"]["gate"]["conditions"]),
            {
                "suite_green",
                "exact_commit",
                "copy_matches_source",
                "source_unchanged",
                "source_sha256_matches_current_tree",
            },
        )
        self.assertEqual(result["preregistered_spec"]["paired_seeds"], [1201, 1202])
        self.assertFalse(result["preregistered_spec"]["matches_committed_spec"])
        self.assertEqual(len(result["seeds"]), 2)

        for record in result["seeds"]:
            for task_key in ("task_a", "task_b"):
                for arm in ("positive", "control"):
                    block = record[task_key][arm]
                    for phase in ("pre", "post"):
                        self.assertIn(phase, block)
                    self.assertIn("components", block["seed_gate"])
                    self.assertIn("conditions", block["seed_gate"])
                    self.assertIn("passed", block["seed_gate"])
                    self.assertTrue(block["seed_gate"]["preregistered"])
                    self.assertEqual(
                        set(block["seed_gate"]["conditions"]),
                        set(block["seed_gate"]["components"]),
                    )
                    for condition in block["seed_gate"]["conditions"].values():
                        self.assertIsInstance(condition, bool)
                self.assertIn("paired", record[task_key])
            self.assertEqual(len(record["task_a"]["positive"]["post"]["cases"]), 3)
            self.assertEqual(len(record["task_a"]["control"]["pre"]["cases"]), 3)
            for arm in ("positive", "control"):
                for cue in ("up", "down"):
                    metrics = record["task_b"][arm]["post"][cue]
                    self.assertIn("direction", metrics)
                    self.assertIn("requested_risk", metrics)
                    self.assertIn("delayed_log_growth", metrics)
                    self.assertIn("categorical_probabilities", metrics)
                self.assertIn("direction_probability_margin", record["task_b"][arm]["post"])

        for task_key, expected_components in (
            (
                "task_a",
                {
                    "positive_seed_pass_count",
                    "control_seed_pass_count",
                    "positive_median_pre_to_post_target_mae_improvement",
                    "paired_positive_target_mae_strictly_better_than_control_count",
                },
            ),
            (
                "task_b",
                {
                    "positive_seed_pass_count",
                    "control_seed_pass_count",
                    "positive_median_pre_to_post_true_delayed_log_growth_improvement",
                    "paired_positive_growth_minus_control_count",
                },
            ),
        ):
            gate = result["aggregate_gates"][task_key]
            self.assertTrue(expected_components.issubset(set(gate["components"])))
            self.assertEqual(set(gate["conditions"]), expected_components)
            for condition in gate["conditions"]:
                self.assertTrue(
                    any(
                        key.startswith(condition)
                        for key in gate["preregistered"]
                    ),
                    f"no preregistered threshold recorded for condition {condition}",
                )
            self.assertIn("passed", gate)
            self.assertEqual(gate["seed_count"], 2)

        self.assertEqual(set(result["verdicts"]), {"task_a", "task_b", "global"})
        self.assertEqual(result["classification"], outcome.classification)
        for section in (
            "## Implementation correctness",
            "## Task A controlled learnability",
            "## Task B delayed-credit learnability",
            "## Negative controls",
            "## Scientific verdict",
            "## Limitations",
        ):
            self.assertIn(section, report)


class AllowlistTests(unittest.TestCase):
    def load_dispatcher(self):
        spec = importlib.util.spec_from_file_location(
            "cb16_dispatch_for_vs_c_test", REPO_ROOT / "scripts" / "cb16_dispatch.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_allowlist_resolves_the_locked_uv_command(self):
        """Requirement 20."""

        dispatcher = self.load_dispatcher()
        allowlist = dispatcher.load_allowlist(ALLOWLIST_PATH)
        entry = allowlist["entrypoints"][COMMAND]
        expected_argv = [
            "uv",
            "run",
            "--frozen",
            "--project",
            ".",
            "--python",
            "3.12",
            "python",
            "-m",
            "cb16_science.vslice.qualification",
        ]
        self.assertEqual(entry["argv"], expected_argv)
        self.assertEqual(entry["argv"][0], "uv")
        self.assertEqual(dispatcher.resolve_allowlisted_argv(entry, REPO_ROOT), expected_argv)
        self.assertEqual(entry["env"]["PYTHONPATH"], "science")
        self.assertEqual(entry["env"]["UV_LINK_MODE"], "copy")
        for key in ("UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR"):
            value = entry["env"][key]
            self.assertTrue(value.startswith("/tmp/"), value)
            self.assertFalse(value.startswith(str(REPO_ROOT)), value)
        self.assertFalse(entry.get("dry_run_only"))
        self.assertEqual(
            sorted(entry["produces"]), ["REPORT.md", "RESULT.json", "experiment_spec.json"]
        )
        self.assertTrue(
            (REPO_ROOT / "science" / "cb16_science" / "vslice" / "qualification.py").is_file()
        )

        task = dispatcher.validate_metadata(
            {
                "mode": "science",
                "commit_sha": "0" * 40,
                "experiment_spec": "docs/tasks/R12_VS_C_CONTROLLED_LEARNABILITY_R0.md",
                "result_command": COMMAND,
            },
            "science",
            allowlist=allowlist,
        )
        self.assertEqual(task.result_command, COMMAND)


if __name__ == "__main__":
    unittest.main()
