"""R12 VS-C formal qualification runner (allowlisted Science entrypoint).

Authority:

* ``config/experiments/r12_vs_c_r0.json`` -- the preregistered machine-readable
  spec.  The formal entrypoint reads exactly that file; there is no CLI
  argument, environment variable or alternate path that can override a
  scientific parameter.
* ``docs/tasks/R12_VS_C_CONTROLLED_LEARNABILITY_R0.md`` -- the task contract.

The runner implements the preregistered protocol literally:

```text
run the full implementation test suite against a byte-identical copy of this commit
fail closed unless that exact-commit test gate is green:
  test failures / broken implementation  -> CONTRACT_MISMATCH
  suite cannot execute (timeout, missing
  executable, analogous runtime blocker) -> EXECUTION_BLOCKED
torch.manual_seed(seed) before each positive/control learner construction
paired learners share bitwise-identical initial Actor/Critic parameters
deterministic collection streams derived only from seed/task/arm/generation
PRE deterministic-adapter evaluation before any learner update
exactly ``generations`` on-policy updates, one generation per update
POST deterministic-adapter evaluation after the final generation
mechanical gate evaluation from the committed JSON thresholds
```

The implementation test gate is what makes the contract's Global PASS condition
("the full implementation test suite is green") a measured fact rather than a
recorded promise: the runner executes the repository-owned suite
(``python -m unittest discover -s tests -t .``) against an exact-content copy of
the worktree it will qualify, binds the evidence to ``CB16_COMMIT_SHA`` and to a
SHA-256 digest of the implementation surface, and refuses to produce any
scientific verdict unless the suite is green, the copy is byte-identical to the
source, and the source is unchanged by the test run.

A qualification blocked by the implementation test gate is never a scientific
result: Task A and Task B stay ``NOT_EVALUATED`` and the artifacts carry no
scientific verdict.  A suite that ran and failed is a contract violation
(``CONTRACT_MISMATCH``); a suite that could not be executed is an
execution/environment blocker (``EXECUTION_BLOCKED``).

It writes only the three preregistered artifacts (``experiment_spec.json``,
``RESULT.json``, ``REPORT.md``) under ``CB16_RESULT_DIR``.  It never writes to
the read-only Science worktree, never reads historical market data, and never
touches the final holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from . import controlled_tasks as tasks
from .contracts import ContractError
from .learner import OnPolicyLearner
from .policy import Actor, ValueCritic
from .sensory import FrozenSensory

#: Allowlist identifier (``config/cb16_science_allowlist.json``).
RESULT_COMMAND = "cb16.vs-c-controlled-learnability@v1"
EXPERIMENT_ID = "r12.vs_c.controlled_learnability.r0"

#: Repository root resolved from this module (``science/cb16_science/vslice``).
REPO_ROOT = Path(__file__).resolve().parents[3]
#: The committed preregistered spec.  This is the only spec the formal runner reads.
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_c_r0.json"

SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

#: Exit codes.  A completed run is 0 (PASS) or 1 (SCIENTIFIC_FAIL); a contract
#: or environment problem that prevented a complete run is 4 or 3.
EXIT_PASS = 0
EXIT_SCIENTIFIC_FAIL = 1
EXIT_EXECUTION_BLOCKED = 3
EXIT_CONTRACT_MISMATCH = 4

_ARM_ID = {tasks.ARM_POSITIVE: 0, tasks.ARM_CONTROL: 1}
_TASK_ID = {"task_a": tasks.TASK_A_ID, "task_b": tasks.TASK_B_ID}

# ---------------------------------------------------------------------------
# Exact-commit implementation test gate
# ---------------------------------------------------------------------------

#: The repository-owned implementation suite, run exactly as the dispatcher runs it.
IMPLEMENTATION_TEST_ARGV: Tuple[str, ...] = (
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-t",
    ".",
)
#: Hard upper bound for the implementation suite inside the formal run.
IMPLEMENTATION_TEST_TIMEOUT_SECONDS = 1800.0

#: Directories whose code/config content defines the implementation under test.
IMPLEMENTATION_SURFACE_DIRECTORIES: Tuple[str, ...] = ("science", "tests", "config", "scripts")
#: Root files that also define the implementation under test.
IMPLEMENTATION_SURFACE_FILES: Tuple[str, ...] = ("pyproject.toml", "uv.lock")
#: Only source-like files enter the implementation digest.
IMPLEMENTATION_SURFACE_SUFFIXES = frozenset({".py", ".json"})
#: Tree members never copied and never hashed.
IMPLEMENTATION_TREE_EXCLUDES = frozenset(
    {
        ".git",
        ".venv",
        ".pip-cache",
        ".uv-cache",
        ".cb16",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
#: Bounded tails of suite output retained as evidence.
IMPLEMENTATION_TEST_TAIL_CHARS = 4000


# ---------------------------------------------------------------------------
# Preregistered spec
# ---------------------------------------------------------------------------


def load_spec() -> Dict[str, Any]:
    """Read the committed preregistered JSON.  The formal path has no alternative."""

    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read preregistered spec {CONFIG_PATH}: {exc}") from exc
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"preregistered spec is not valid JSON: {exc}") from exc
    return tasks.spec_mapping(spec, "preregistered spec")


def spec_sha256(spec: Mapping[str, Any]) -> str:
    """Canonical SHA-256 of the parsed spec actually used by this run."""

    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def matches_committed_spec(spec: Mapping[str, Any]) -> bool:
    """Whether the parsed spec used here equals the committed preregistered spec."""

    try:
        committed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return committed == spec


def paired_seeds(spec: Mapping[str, Any]) -> Tuple[int, ...]:
    """The preregistered paired seeds, in order."""

    seeds = tuple(
        tasks.spec_int(value, "paired_seeds entry", minimum=1)
        for value in tasks.spec_sequence(spec["paired_seeds"], "paired_seeds")
    )
    if not seeds:
        raise ContractError("paired_seeds must not be empty")
    return seeds


def task_section(spec: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return tasks.spec_mapping(spec[key], key)


def gate_section(spec: Mapping[str, Any], task_key: str, gate_key: str) -> Mapping[str, Any]:
    return tasks.spec_mapping(
        task_section(spec, task_key)[gate_key], f"{task_key}.{gate_key}"
    )


@dataclass(frozen=True)
class Authority:
    """Preregistered runtime/architecture authority."""

    context_length: int
    sensory_seed: int
    sensory_z_dim: int
    actor_hidden_width: int
    actor_hidden_layers: int
    device: str
    torch_num_threads: int
    deterministic_algorithms: bool


@dataclass(frozen=True)
class OptimizerSettings:
    """Preregistered optimizer settings (Adam, no entropy, no discount, no replay)."""

    actor_lr: float
    critic_lr: float


def parse_authority(spec: Mapping[str, Any]) -> Authority:
    authority = tasks.spec_mapping(spec["authority"], "authority")
    parsed = Authority(
        context_length=tasks.spec_int(
            authority["context_length"], "authority.context_length", minimum=2
        ),
        sensory_seed=tasks.spec_int(authority["sensory_seed"], "authority.sensory_seed"),
        sensory_z_dim=tasks.spec_int(authority["sensory_z_dim"], "authority.sensory_z_dim", minimum=1),
        actor_hidden_width=tasks.spec_int(
            authority["actor_hidden_width"], "authority.actor_hidden_width", minimum=1
        ),
        actor_hidden_layers=tasks.spec_int(
            authority["actor_hidden_layers"], "authority.actor_hidden_layers", minimum=1
        ),
        device=tasks.spec_str(authority["device"], "authority.device"),
        torch_num_threads=tasks.spec_int(
            authority["torch_num_threads"], "authority.torch_num_threads", minimum=1
        ),
        deterministic_algorithms=tasks.spec_bool(
            authority["deterministic_algorithms"], "authority.deterministic_algorithms"
        ),
    )
    if parsed.device != "cpu":
        raise ContractError(f"VS-C R0 is CPU-only, got device {parsed.device!r}")
    if parsed.torch_num_threads != 1:
        raise ContractError(
            f"VS-C R0 fixes torch_num_threads=1, got {parsed.torch_num_threads!r}"
        )
    if not parsed.deterministic_algorithms:
        raise ContractError("VS-C R0 requires torch deterministic algorithms")
    return parsed


def parse_optimizer(spec: Mapping[str, Any]) -> OptimizerSettings:
    optimizer = tasks.spec_mapping(spec["optimizer"], "optimizer")
    for name in ("actor", "critic"):
        algorithm = tasks.spec_str(optimizer[name], f"optimizer.{name}")
        if algorithm != "Adam":
            raise ContractError(f"optimizer.{name} must be 'Adam', got {algorithm!r}")
    entropy = tasks.spec_float(optimizer["entropy_bonus"], "optimizer.entropy_bonus")
    if entropy != 0.0:
        raise ContractError(f"R0 forbids an entropy bonus, got {entropy!r}")
    discount = tasks.spec_float(optimizer["discount_factor"], "optimizer.discount_factor")
    if discount != 1.0:
        raise ContractError(f"R0 uses undiscounted return-to-go, got discount {discount!r}")
    if tasks.spec_bool(optimizer["replay"], "optimizer.replay"):
        raise ContractError("R0 forbids replay")
    return OptimizerSettings(
        actor_lr=tasks.spec_float(optimizer["actor_lr"], "optimizer.actor_lr"),
        critic_lr=tasks.spec_float(optimizer["critic_lr"], "optimizer.critic_lr"),
    )


def validate_spec(spec: Mapping[str, Any]) -> None:
    """Fail closed on a spec whose preregistered semantics this implementation cannot honour."""

    tasks.spec_mapping(spec, "preregistered spec")
    experiment_id = tasks.spec_str(spec["experiment_id"], "experiment_id")
    if experiment_id != EXPERIMENT_ID:
        raise ContractError(
            f"experiment_id must be {EXPERIMENT_ID!r}, got {experiment_id!r}"
        )
    parse_authority(spec)
    parse_optimizer(spec)
    paired_seeds(spec)

    global_gate = tasks.spec_mapping(spec["global_gate"], "global_gate")
    if not tasks.spec_bool(
        global_gate["pass_requires_task_a_and_task_b"], "global_gate.pass_requires_task_a_and_task_b"
    ):
        raise ContractError("global gate must require both Task A and Task B")
    if not tasks.spec_bool(
        global_gate["implementation_tests_must_pass"], "global_gate.implementation_tests_must_pass"
    ):
        raise ContractError("global gate must require the implementation test suite")
    if not tasks.spec_bool(
        global_gate["no_threshold_changes_after_results"],
        "global_gate.no_threshold_changes_after_results",
    ):
        raise ContractError("R0 forbids threshold changes after results")
    expected_classifications = {
        "scientific_gate_miss_classification": "SCIENTIFIC_FAIL",
        "contract_violation_classification": "CONTRACT_MISMATCH",
        "runtime_or_environment_blocker_classification": "EXECUTION_BLOCKED",
    }
    for key, expected in expected_classifications.items():
        value = tasks.spec_str(global_gate[key], f"global_gate.{key}")
        if value != expected:
            raise ContractError(f"global_gate.{key} must be {expected!r}, got {value!r}")

    for task_key in ("task_a", "task_b"):
        task = task_section(spec, task_key)
        tasks.spec_int(task["generations"], f"{task_key}.generations", minimum=1)
        tasks.spec_int(task["batch_trajectories"], f"{task_key}.batch_trajectories", minimum=1)
        tasks.spec_int(task["trajectory_length"], f"{task_key}.trajectory_length", minimum=1)
        tasks.spec_str(
            tasks.spec_mapping(task["evaluation"], f"{task_key}.evaluation")["adapter"],
            f"{task_key}.evaluation.adapter",
        )
        gate_section(spec, task_key, "seed_pass_gate")
        gate_section(spec, task_key, "aggregate_gate")

    required = {
        tasks.spec_str(value, "required_artifacts entry")
        for value in tasks.spec_sequence(spec["required_artifacts"], "required_artifacts")
    }
    expected_artifacts = {SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME}
    if required != expected_artifacts:
        raise ContractError(
            f"required_artifacts must be exactly {sorted(expected_artifacts)}, got {sorted(required)}"
        )


def configure_deterministic_runtime() -> None:
    """CPU-only, one thread, deterministic algorithms (preregistered authority)."""

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------


def build_learner(
    authority: Authority,
    optimizer: OptimizerSettings,
    *,
    direction_entropy_coefficient: float = 0.0,
) -> OnPolicyLearner:
    """One frozen-sensory / Actor / Critic learner with preregistered shapes.

    The caller seeds the ambient torch RNG immediately before this call, so two
    successive calls with the same seed produce bitwise-identical parameters.
    """

    sensory = FrozenSensory(
        z_dim=authority.sensory_z_dim,
        context_length=authority.context_length,
        seed=authority.sensory_seed,
    )
    actor = Actor(
        z_dim=authority.sensory_z_dim,
        hidden_width=authority.actor_hidden_width,
        hidden_layers=authority.actor_hidden_layers,
    )
    critic = ValueCritic(
        z_dim=authority.sensory_z_dim,
        hidden_width=authority.actor_hidden_width,
        hidden_layers=authority.actor_hidden_layers,
    )
    return OnPolicyLearner(
        sensory,
        actor,
        critic,
        actor_lr=optimizer.actor_lr,
        critic_lr=optimizer.critic_lr,
        direction_entropy_coefficient=direction_entropy_coefficient,
    )


def require_identical_parameters(positive: OnPolicyLearner, control: OnPolicyLearner) -> None:
    """Fail closed unless the paired learners start bitwise identically."""

    for name, left, right in (
        ("actor", positive.actor, control.actor),
        ("critic", positive.critic, control.critic),
    ):
        left_parameters = tuple(left.parameters())
        right_parameters = tuple(right.parameters())
        if len(left_parameters) != len(right_parameters):
            raise ContractError(f"paired {name} parameter counts differ")
        for position, (expected, actual) in enumerate(zip(left_parameters, right_parameters)):
            if expected.shape != actual.shape:
                raise ContractError(f"paired {name} parameter {position} shapes differ")
            if not torch.equal(expected.detach(), actual.detach()):
                raise ContractError(
                    f"paired {name} parameter {position} is not bitwise identical at construction"
                )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _diagnostics_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"generations": 0}
    return {
        "generations": len(records),
        "first_actor_loss": records[0]["actor_loss"],
        "last_actor_loss": records[-1]["actor_loss"],
        "first_critic_loss": records[0]["critic_loss"],
        "last_critic_loss": records[-1]["critic_loss"],
        "note": "loss diagnostics only; no gate is derived from them",
    }


def _collection_seed(seed: int, task_key: str, mode: str, generation: int) -> int:
    return tasks.derived_stream_seed(
        tasks.COLLECTION_DOMAIN, seed, _TASK_ID[task_key], _ARM_ID[mode], generation
    )


def train_task_a(
    learner: OnPolicyLearner,
    env: tasks.TaskAEnvironment,
    *,
    seed: int,
    mode: str,
    generations: int,
) -> Dict[str, Any]:
    """Exactly ``generations`` Task-A updates, each consuming one current generation."""

    records: List[Dict[str, Any]] = []
    for generation in range(generations):
        if learner.generation_id != generation:
            raise ContractError(
                f"learner generation {learner.generation_id} does not match the scheduled "
                f"generation {generation}"
            )
        torch.manual_seed(_collection_seed(seed, "task_a", mode, generation))
        batch = tasks.build_task_a_generation(learner, env, generation, mode=mode)
        report = learner.update(batch)
        if report.next_generation_id != generation + 1:
            raise ContractError(
                f"a single update must advance generation {generation} to {generation + 1}, got "
                f"{report.next_generation_id}"
            )
        records.append(
            {
                "generation_id": generation,
                "actor_loss": report.actor_loss,
                "critic_loss": report.critic_loss,
            }
        )
    return _diagnostics_summary(records)


def train_task_b(
    learner: OnPolicyLearner,
    env: tasks.TaskBEnvironment,
    *,
    seed: int,
    mode: str,
    generations: int,
) -> Dict[str, Any]:
    """Exactly ``generations`` Task-B updates, each consuming one current generation."""

    records: List[Dict[str, Any]] = []
    for generation in range(generations):
        if learner.generation_id != generation:
            raise ContractError(
                f"learner generation {learner.generation_id} does not match the scheduled "
                f"generation {generation}"
            )
        torch.manual_seed(_collection_seed(seed, "task_b", mode, generation))
        batch = tasks.build_task_b_generation(
            learner, env, generation, mode=mode, seed=seed
        )
        report = learner.update(batch)
        if report.next_generation_id != generation + 1:
            raise ContractError(
                f"a single update must advance generation {generation} to {generation + 1}, got "
                f"{report.next_generation_id}"
            )
        records.append(
            {
                "generation_id": generation,
                "actor_loss": report.actor_loss,
                "critic_loss": report.critic_loss,
            }
        )
    return _diagnostics_summary(records)


# ---------------------------------------------------------------------------
# Gates (computed mechanically from the committed JSON)
# ---------------------------------------------------------------------------


def task_a_seed_gate(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> Dict[str, Any]:
    """Task-A seed PASS: 3/3 correct directions and target MAE <= threshold."""

    direction_minimum = tasks.spec_int(
        gate["direction_correct_count_min"], "task_a.seed_pass_gate.direction_correct_count_min"
    )
    mae_maximum = tasks.spec_float(
        gate["target_exposure_mae_max"], "task_a.seed_pass_gate.target_exposure_mae_max"
    )
    components = {
        "direction_correct_count": metrics["direction_correct_count"],
        "target_exposure_mae": metrics["target_exposure_mae"],
    }
    conditions = {
        "direction_correct_count": components["direction_correct_count"] >= direction_minimum,
        "target_exposure_mae": components["target_exposure_mae"] <= mae_maximum,
    }
    return {
        "preregistered": {
            "direction_correct_count_min": direction_minimum,
            "target_exposure_mae_max": mae_maximum,
        },
        "components": components,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def task_b_seed_gate(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> Dict[str, Any]:
    """Task-B seed PASS: correct UP/DOWN directions, risks >= floor, growth >= floor."""

    up_direction = tasks.spec_str(gate["up_direction"], "task_b.seed_pass_gate.up_direction")
    down_direction = tasks.spec_str(
        gate["down_direction"], "task_b.seed_pass_gate.down_direction"
    )
    up_risk_minimum = tasks.spec_float(
        gate["up_requested_risk_min"], "task_b.seed_pass_gate.up_requested_risk_min"
    )
    down_risk_minimum = tasks.spec_float(
        gate["down_requested_risk_min"], "task_b.seed_pass_gate.down_requested_risk_min"
    )
    growth_minimum = tasks.spec_float(
        gate["mean_true_delayed_log_growth_min"],
        "task_b.seed_pass_gate.mean_true_delayed_log_growth_min",
    )
    components = {
        "up_direction": metrics["up"]["direction"],
        "down_direction": metrics["down"]["direction"],
        "up_requested_risk": metrics["up"]["requested_risk"],
        "down_requested_risk": metrics["down"]["requested_risk"],
        "mean_true_delayed_log_growth": metrics["mean_true_delayed_log_growth"],
    }
    conditions = {
        "up_direction": components["up_direction"] == up_direction,
        "down_direction": components["down_direction"] == down_direction,
        "up_requested_risk": components["up_requested_risk"] >= up_risk_minimum,
        "down_requested_risk": components["down_requested_risk"] >= down_risk_minimum,
        "mean_true_delayed_log_growth": components["mean_true_delayed_log_growth"]
        >= growth_minimum,
    }
    return {
        "preregistered": {
            "up_direction": up_direction,
            "down_direction": down_direction,
            "up_requested_risk_min": up_risk_minimum,
            "down_requested_risk_min": down_risk_minimum,
            "mean_true_delayed_log_growth_min": growth_minimum,
        },
        "components": components,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def task_a_aggregate_gate(
    seed_records: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> Dict[str, Any]:
    """All four preregistered Task-A aggregate conditions."""

    positive_passes = [
        bool(record["task_a"]["positive"]["seed_gate"]["passed"]) for record in seed_records
    ]
    control_passes = [
        bool(record["task_a"]["control"]["seed_gate"]["passed"]) for record in seed_records
    ]
    improvements = [
        float(record["task_a"]["paired"]["positive_target_mae_improvement"])
        for record in seed_records
    ]
    strictly_better = [
        bool(record["task_a"]["paired"]["positive_post_target_mae_strictly_better_than_control"])
        for record in seed_records
    ]

    positive_minimum = tasks.spec_int(
        gate["positive_seed_pass_count_min"], "task_a.aggregate_gate.positive_seed_pass_count_min"
    )
    control_maximum = tasks.spec_int(
        gate["control_seed_pass_count_max"], "task_a.aggregate_gate.control_seed_pass_count_max"
    )
    improvement_minimum = tasks.spec_float(
        gate["positive_median_pre_to_post_target_mae_improvement_min"],
        "task_a.aggregate_gate.positive_median_pre_to_post_target_mae_improvement_min",
    )
    better_count_minimum = tasks.spec_int(
        gate["paired_positive_target_mae_strictly_better_than_control_count_min"],
        "task_a.aggregate_gate.paired_positive_target_mae_strictly_better_than_control_count_min",
    )

    components = {
        "positive_seed_pass_count": sum(positive_passes),
        "control_seed_pass_count": sum(control_passes),
        "positive_median_pre_to_post_target_mae_improvement": _median(improvements),
        "paired_positive_target_mae_strictly_better_than_control_count": sum(strictly_better),
    }
    conditions = {
        "positive_seed_pass_count": components["positive_seed_pass_count"] >= positive_minimum,
        "control_seed_pass_count": components["control_seed_pass_count"] <= control_maximum,
        "positive_median_pre_to_post_target_mae_improvement": components[
            "positive_median_pre_to_post_target_mae_improvement"
        ]
        >= improvement_minimum,
        "paired_positive_target_mae_strictly_better_than_control_count": components[
            "paired_positive_target_mae_strictly_better_than_control_count"
        ]
        >= better_count_minimum,
    }
    return {
        "preregistered": {
            "positive_seed_pass_count_min": positive_minimum,
            "control_seed_pass_count_max": control_maximum,
            "positive_median_pre_to_post_target_mae_improvement_min": improvement_minimum,
            "paired_positive_target_mae_strictly_better_than_control_count_min": better_count_minimum,
        },
        "components": components,
        "conditions": conditions,
        "passed": all(conditions.values()),
        "seed_count": len(seed_records),
    }


def task_b_aggregate_gate(
    seed_records: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> Dict[str, Any]:
    """All four preregistered Task-B aggregate conditions.

    ``paired_positive_growth_minus_control_min`` is the per-seed margin of the
    paired count condition (a seed counts only when its POST positive-minus-
    control true delayed log growth reaches that margin); it is not a separate
    aggregate statistic.  The observed minimum is reported as a component.
    """

    positive_passes = [
        bool(record["task_b"]["positive"]["seed_gate"]["passed"]) for record in seed_records
    ]
    control_passes = [
        bool(record["task_b"]["control"]["seed_gate"]["passed"]) for record in seed_records
    ]
    improvements = [
        float(record["task_b"]["paired"]["positive_true_delayed_log_growth_improvement"])
        for record in seed_records
    ]
    differences = [
        float(record["task_b"]["paired"]["positive_minus_control_true_delayed_log_growth"])
        for record in seed_records
    ]

    positive_minimum = tasks.spec_int(
        gate["positive_seed_pass_count_min"], "task_b.aggregate_gate.positive_seed_pass_count_min"
    )
    control_maximum = tasks.spec_int(
        gate["control_seed_pass_count_max"], "task_b.aggregate_gate.control_seed_pass_count_max"
    )
    improvement_minimum = tasks.spec_float(
        gate["positive_median_pre_to_post_true_delayed_log_growth_improvement_min"],
        "task_b.aggregate_gate.positive_median_pre_to_post_true_delayed_log_growth_improvement_min",
    )
    difference_minimum = tasks.spec_float(
        gate["paired_positive_growth_minus_control_min"],
        "task_b.aggregate_gate.paired_positive_growth_minus_control_min",
    )
    difference_count_minimum = tasks.spec_int(
        gate["paired_positive_growth_minus_control_count_min"],
        "task_b.aggregate_gate.paired_positive_growth_minus_control_count_min",
    )

    components = {
        "positive_seed_pass_count": sum(positive_passes),
        "control_seed_pass_count": sum(control_passes),
        "positive_median_pre_to_post_true_delayed_log_growth_improvement": _median(improvements),
        "paired_positive_growth_minus_control_count": sum(
            1 for difference in differences if difference >= difference_minimum
        ),
        "paired_positive_growth_minus_control_min_observed": (
            float(min(differences)) if differences else 0.0
        ),
    }
    conditions = {
        "positive_seed_pass_count": components["positive_seed_pass_count"] >= positive_minimum,
        "control_seed_pass_count": components["control_seed_pass_count"] <= control_maximum,
        "positive_median_pre_to_post_true_delayed_log_growth_improvement": components[
            "positive_median_pre_to_post_true_delayed_log_growth_improvement"
        ]
        >= improvement_minimum,
        "paired_positive_growth_minus_control_count": components[
            "paired_positive_growth_minus_control_count"
        ]
        >= difference_count_minimum,
    }
    return {
        "preregistered": {
            "positive_seed_pass_count_min": positive_minimum,
            "control_seed_pass_count_max": control_maximum,
            "positive_median_pre_to_post_true_delayed_log_growth_improvement_min": improvement_minimum,
            "paired_positive_growth_minus_control_min": difference_minimum,
            "paired_positive_growth_minus_control_count_min": difference_count_minimum,
        },
        "components": components,
        "conditions": conditions,
        "passed": all(conditions.values()),
        "seed_count": len(seed_records),
    }


# ---------------------------------------------------------------------------
# Paired-seed run
# ---------------------------------------------------------------------------


def run_paired_seed(
    spec: Mapping[str, Any],
    seed: int,
    authority: Authority,
    optimizer: OptimizerSettings,
    env_a: tasks.TaskAEnvironment,
    env_b: tasks.TaskBEnvironment,
) -> Dict[str, Any]:
    """Run both tasks for one preregistered seed, positive and matched control."""

    task_a = task_section(spec, "task_a")
    task_b = task_section(spec, "task_b")
    generations_a = tasks.spec_int(task_a["generations"], "task_a.generations", minimum=1)
    generations_b = tasks.spec_int(task_b["generations"], "task_b.generations", minimum=1)
    gate_a = gate_section(spec, "task_a", "seed_pass_gate")
    gate_b = gate_section(spec, "task_b", "seed_pass_gate")

    # --- Task A -----------------------------------------------------------
    torch.manual_seed(seed)
    positive_a = build_learner(authority, optimizer)
    torch.manual_seed(seed)
    control_a = build_learner(authority, optimizer)
    require_identical_parameters(positive_a, control_a)

    pre_positive_a = tasks.evaluate_task_a(positive_a, env_a)
    pre_control_a = tasks.evaluate_task_a(control_a, env_a)
    diagnostics_positive_a = train_task_a(
        positive_a, env_a, seed=seed, mode=tasks.ARM_POSITIVE, generations=generations_a
    )
    diagnostics_control_a = train_task_a(
        control_a, env_a, seed=seed, mode=tasks.ARM_CONTROL, generations=generations_a
    )
    post_positive_a = tasks.evaluate_task_a(positive_a, env_a)
    post_control_a = tasks.evaluate_task_a(control_a, env_a)

    paired_a = {
        "pre_positive_target_mae": pre_positive_a["target_exposure_mae"],
        "post_positive_target_mae": post_positive_a["target_exposure_mae"],
        "pre_control_target_mae": pre_control_a["target_exposure_mae"],
        "post_control_target_mae": post_control_a["target_exposure_mae"],
        "positive_target_mae_improvement": (
            pre_positive_a["target_exposure_mae"] - post_positive_a["target_exposure_mae"]
        ),
        "control_target_mae_improvement": (
            pre_control_a["target_exposure_mae"] - post_control_a["target_exposure_mae"]
        ),
        "positive_post_target_mae_strictly_better_than_control": bool(
            post_positive_a["target_exposure_mae"] < post_control_a["target_exposure_mae"]
        ),
    }

    # --- Task B (fresh matched learners for the same paired seed) ---------
    torch.manual_seed(seed)
    positive_b = build_learner(authority, optimizer)
    torch.manual_seed(seed)
    control_b = build_learner(authority, optimizer)
    require_identical_parameters(positive_b, control_b)

    pre_positive_b = tasks.evaluate_task_b(positive_b, env_b)
    pre_control_b = tasks.evaluate_task_b(control_b, env_b)
    diagnostics_positive_b = train_task_b(
        positive_b, env_b, seed=seed, mode=tasks.ARM_POSITIVE, generations=generations_b
    )
    diagnostics_control_b = train_task_b(
        control_b, env_b, seed=seed, mode=tasks.ARM_CONTROL, generations=generations_b
    )
    post_positive_b = tasks.evaluate_task_b(positive_b, env_b)
    post_control_b = tasks.evaluate_task_b(control_b, env_b)

    paired_b = {
        "pre_positive_true_delayed_log_growth": pre_positive_b[
            "mean_true_delayed_log_growth"
        ],
        "post_positive_true_delayed_log_growth": post_positive_b[
            "mean_true_delayed_log_growth"
        ],
        "pre_control_true_delayed_log_growth": pre_control_b["mean_true_delayed_log_growth"],
        "post_control_true_delayed_log_growth": post_control_b["mean_true_delayed_log_growth"],
        "positive_true_delayed_log_growth_improvement": (
            post_positive_b["mean_true_delayed_log_growth"]
            - pre_positive_b["mean_true_delayed_log_growth"]
        ),
        "control_true_delayed_log_growth_improvement": (
            post_control_b["mean_true_delayed_log_growth"]
            - pre_control_b["mean_true_delayed_log_growth"]
        ),
        "positive_minus_control_true_delayed_log_growth": (
            post_positive_b["mean_true_delayed_log_growth"]
            - post_control_b["mean_true_delayed_log_growth"]
        ),
    }

    return {
        "seed": seed,
        "task_a": {
            "positive": {
                "pre": pre_positive_a,
                "post": post_positive_a,
                "seed_gate": task_a_seed_gate(post_positive_a, gate_a),
                "diagnostics": diagnostics_positive_a,
            },
            "control": {
                "pre": pre_control_a,
                "post": post_control_a,
                "seed_gate": task_a_seed_gate(post_control_a, gate_a),
                "diagnostics": diagnostics_control_a,
            },
            "paired": paired_a,
        },
        "task_b": {
            "positive": {
                "pre": pre_positive_b,
                "post": post_positive_b,
                "seed_gate": task_b_seed_gate(post_positive_b, gate_b),
                "diagnostics": diagnostics_positive_b,
            },
            "control": {
                "pre": pre_control_b,
                "post": post_control_b,
                "seed_gate": task_b_seed_gate(post_control_b, gate_b),
                "diagnostics": diagnostics_control_b,
            },
            "paired": paired_b,
        },
    }


# ---------------------------------------------------------------------------
# Experiment orchestration and artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualificationOutcome:
    """One complete qualification result plus its artifacts."""

    spec: Mapping[str, Any]
    result: Mapping[str, Any]
    report: str
    classification: str
    exit_code: int
    summary_line: str


def commit_sha() -> str:
    value = (os.environ.get("CB16_COMMIT_SHA") or "").strip()
    return value or "not-supplied"


def runtime_record() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "uv_project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT", "not-supplied"),
    }


def implementation_surface_files(root: Path = REPO_ROOT) -> Tuple[Path, ...]:
    """The exact files whose content defines the implementation under test.

    The surface is the repository code and configuration the formal run and its
    tests execute: every ``.py`` under ``science``/``tests``/``scripts``, every
    ``.json`` under ``config`` (the preregistered spec and the Science
    allowlist), plus the dependency lock metadata.  Caches and git state are
    excluded, so the digest is a pure content identity.
    """

    root = Path(root)
    files: List[Path] = []
    for directory in IMPLEMENTATION_SURFACE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in IMPLEMENTATION_SURFACE_SUFFIXES:
                continue
            if any(part in IMPLEMENTATION_TREE_EXCLUDES for part in path.parts):
                continue
            files.append(path)
    for name in IMPLEMENTATION_SURFACE_FILES:
        path = root / name
        if path.is_file():
            files.append(path)
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def implementation_source_sha256(root: Path = REPO_ROOT) -> str:
    """SHA-256 over the implementation surface content of ``root``."""

    root = Path(root)
    digest = hashlib.sha256()
    for path in implementation_surface_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_implementation_tree(source: Path, destination: Path) -> int:
    """Copy the worktree (minus git/caches) into ``destination``; return file count.

    The implementation suite is executed against this copy instead of the
    read-only Science worktree, so no test can write into the immutable source
    tree; :func:`implementation_source_sha256` proves the copy is
    byte-identical for every file that defines the implementation.
    """

    source = Path(source)
    destination = Path(destination)
    copied = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in IMPLEMENTATION_TREE_EXCLUDES for part in relative.parts):
            continue
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    return copied


def snapshot_git_repository(root: Path, *, timeout: float = 180.0) -> bool:
    """Best-effort git snapshot of a copied tree.

    A few repository tests assert git tracking of plugin sources and skip when
    the tree is not a checkout; a fresh single-commit snapshot lets them run in
    the copy.  Failure is recorded, never fatal: the content digest, not git,
    is what binds the evidence to the exact commit.
    """

    commands = (
        ("git", "init", "-q", "-b", "main"),
        ("git", "add", "-A"),
        (
            "git",
            "-c",
            "user.email=cb16@example.invalid",
            "-c",
            "user.name=CB16",
            "commit",
            "-q",
            "-m",
            "exact-content implementation snapshot",
        ),
    )
    for argv in commands:
        try:
            proc = subprocess.run(
                list(argv), cwd=str(root), capture_output=True, text=True, timeout=timeout
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if proc.returncode != 0:
            return False
    return True


def repository_head(root: Path = REPO_ROOT, *, timeout: float = 60.0) -> "str | None":
    """Best-effort HEAD SHA of the worktree.

    Recorded as evidence only: the gate binds the run to the exact commit by the
    recorded ``CB16_COMMIT_SHA`` and by the implementation-surface digest, so a
    sandbox that cannot read git state must not block a qualification run.
    """

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    return value or None


def parse_unittest_summary(stdout: str) -> Dict[str, Any]:
    """Extract the unittest summary facts from captured suite output."""

    text = stdout or ""
    run_match = re.search(r"^Ran (\d+) tests? in ", text, re.MULTILINE)
    skip_match = re.search(r"skipped=(\d+)", text)
    return {
        "tests_run": int(run_match.group(1)) if run_match else None,
        "skipped": int(skip_match.group(1)) if skip_match else None,
        "unittest_ok": bool(re.search(r"^OK(\s|\(|$)", text, re.MULTILINE)),
    }


def summarise_unittest_run(exit_code: int, stdout: str, stderr: str) -> Dict[str, Any]:
    """Status facts for one unittest invocation.

    ``unittest`` writes its progress and summary to stderr, so both streams are
    read; an exit code of zero alone is never treated as green.
    """

    summary = parse_unittest_summary(f"{stdout or ''}\n{stderr or ''}")
    summary["exit_code"] = int(exit_code)
    summary["status"] = (
        "PASS" if summary["exit_code"] == 0 and summary["unittest_ok"] else "FAIL"
    )
    return summary


def _bounded_tail(text: str, limit: int = IMPLEMENTATION_TEST_TAIL_CHARS) -> str:
    value = text or ""
    return value if len(value) <= limit else value[-limit:]


def _science_first_pythonpath(existing: str, root: Path) -> str:
    science = str(Path(root) / "science")
    parts = [part for part in (existing or "").split(os.pathsep) if part]
    if science in parts:
        parts.remove(science)
    return os.pathsep.join([science, *parts])


def run_implementation_test_suite(
    *, timeout: float = IMPLEMENTATION_TEST_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Run the repository implementation suite for this exact commit.

    The suite runs in a disposable copy of the worktree under the per-run
    writable tmp area, with the same interpreter that runs the formal
    entrypoint.  The returned evidence is exact-commit-bound by the recorded
    ``commit_sha`` and by the implementation-surface digest of the source tree
    before the run, of the copy that was tested, and of the source tree after
    the run.
    """

    source_before = implementation_source_sha256(REPO_ROOT)
    argv = [sys.executable, *IMPLEMENTATION_TEST_ARGV]
    evidence: Dict[str, Any] = {
        "status": "FAIL",
        "passed": False,
        "runner_executed_suite": True,
        "required_by_global_gate": True,
        "commit_sha": commit_sha(),
        "git_head": repository_head(),
        "command": argv,
        "exit_code": None,
        "tests_run": None,
        "skipped": None,
        "source_sha256": source_before,
        "copy_sha256": None,
        "source_unchanged": False,
        "copy_matches_source": False,
        "copied_file_count": 0,
        "git_snapshot": False,
        "stdout_tail": "",
        "stderr_tail": "",
        "preregistered_requirement": (
            "global_gate.implementation_tests_must_pass: the full implementation test suite "
            "must be green at the exact commit"
        ),
        "note": (
            "The suite is executed by this runner against a byte-identical copy of the exact "
            "commit's implementation surface; a non-green suite is CONTRACT_MISMATCH and no "
            "scientific verdict is produced."
        ),
    }

    with tempfile.TemporaryDirectory(prefix="cb16-vs-c-tests-") as tmp:
        copy_root = Path(tmp) / "repo"
        copy_root.mkdir(parents=True, exist_ok=True)
        evidence["copied_file_count"] = copy_implementation_tree(REPO_ROOT, copy_root)
        copy_digest = implementation_source_sha256(copy_root)
        evidence["copy_sha256"] = copy_digest
        evidence["copy_matches_source"] = copy_digest == source_before

        stdout = ""
        stderr = ""
        if evidence["copy_matches_source"]:
            evidence["git_snapshot"] = snapshot_git_repository(copy_root)
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = _science_first_pythonpath(env.get("PYTHONPATH", ""), copy_root)
            try:
                proc = subprocess.run(
                    argv,
                    cwd=str(copy_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                evidence["status"] = "TIMEOUT"
                evidence["exit_code"] = 124
                stdout = _as_text(exc.stdout)
                stderr = _as_text(exc.stderr)
            except OSError as exc:
                evidence["status"] = "NOT_EXECUTABLE"
                evidence["exit_code"] = 126
                stderr = f"{type(exc).__name__}: {exc}"
            else:
                evidence["exit_code"] = proc.returncode
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                # unittest reports on stderr; both streams are summarised.
                evidence.update(
                    summarise_unittest_run(proc.returncode, stdout, stderr)
                )
        evidence["stdout_tail"] = _bounded_tail(stdout)
        evidence["stderr_tail"] = _bounded_tail(stderr)

    source_after = implementation_source_sha256(REPO_ROOT)
    evidence["source_unchanged"] = source_after == source_before
    evidence["passed"] = bool(
        evidence["status"] == "PASS"
        and evidence["copy_matches_source"]
        and evidence["source_unchanged"]
    )
    return evidence


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def implementation_test_gate(evidence: Mapping[str, Any], *, commit: str) -> Dict[str, Any]:
    """The exact-commit implementation test gate.

    Global PASS requires every condition: the suite is green, the evidence was
    produced for the exact commit of this run, the tested copy was
    byte-identical to the source, the source tree was unchanged by the test run,
    and the recorded digest still matches the current tree.
    """

    conditions = {
        "suite_green": evidence.get("status") == "PASS",
        "exact_commit": evidence.get("commit_sha") == commit,
        "copy_matches_source": bool(evidence.get("copy_matches_source")),
        "source_unchanged": bool(evidence.get("source_unchanged")),
        "source_sha256_matches_current_tree": (
            evidence.get("source_sha256") == implementation_source_sha256(REPO_ROOT)
        ),
    }
    return {
        "components": {
            "status": evidence.get("status"),
            "commit_sha": evidence.get("commit_sha"),
            "git_head": evidence.get("git_head"),
            "git_head_matches_declared_commit": (
                None
                if evidence.get("git_head") is None
                else evidence.get("git_head") == commit
            ),
            "command": list(evidence.get("command") or []),
            "exit_code": evidence.get("exit_code"),
            "tests_run": evidence.get("tests_run"),
            "skipped": evidence.get("skipped"),
            "source_sha256": evidence.get("source_sha256"),
            "copy_sha256": evidence.get("copy_sha256"),
            "copied_file_count": evidence.get("copied_file_count"),
            "git_snapshot": evidence.get("git_snapshot"),
        },
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


#: Suite status that means the runner executed the suite and unittest reported a
#: non-green result.  Every other non-green status (``TIMEOUT``,
#: ``NOT_EXECUTABLE``, a missing verdict, or a green run whose binding evidence
#: did not hold) means the qualification could not be validly executed.
IMPLEMENTATION_TEST_RED_STATUS = "FAIL"

#: Classification and exit code for each implementation-test failure class.
IMPLEMENTATION_TEST_FAILURE_TAXONOMY = {
    "CONTRACT_MISMATCH": EXIT_CONTRACT_MISMATCH,
    "EXECUTION_BLOCKED": EXIT_EXECUTION_BLOCKED,
}


def implementation_failure_classification(evidence: Mapping[str, Any]) -> Tuple[str, int]:
    """Map a non-green exact-commit implementation test gate to its taxonomy.

    ``FAIL`` is the one status that proves the implementation itself is broken:
    the suite ran at the exact commit and unittest reported failures, so the
    contract violation is ``CONTRACT_MISMATCH``.  ``TIMEOUT``, ``NOT_EXECUTABLE``
    and every other non-green outcome mean the suite produced no verdict (or
    could not be bound to the exact commit), which is a runtime/environment
    blocker: ``EXECUTION_BLOCKED``.

    Neither case is a scientific result; Task A and Task B stay
    ``NOT_EVALUATED`` in both.
    """

    status = evidence.get("status")
    classification = (
        "CONTRACT_MISMATCH"
        if status == IMPLEMENTATION_TEST_RED_STATUS
        else "EXECUTION_BLOCKED"
    )
    return classification, IMPLEMENTATION_TEST_FAILURE_TAXONOMY[classification]


def implementation_test_failure_detail(classification: str, evidence: Mapping[str, Any]) -> str:
    """Human-readable detail for a blocked implementation test gate."""

    status = evidence.get("status")
    exit_code = evidence.get("exit_code")
    commit = evidence.get("commit_sha")
    if classification == "CONTRACT_MISMATCH":
        return (
            "the exact-commit implementation test suite ran and did not pass "
            f"(status={status!r}, exit_code={exit_code!r}, "
            f"tests_run={evidence.get('tests_run')!r}, commit={commit!r}); "
            "no scientific result was produced"
        )
    return (
        "the exact-commit implementation test suite could not be executed "
        f"(status={status!r}, exit_code={exit_code!r}, commit={commit!r}); "
        "no scientific result was produced"
    )


def run_experiment(
    spec: Mapping[str, Any], *, seeds: Sequence[int], implementation_tests: Mapping[str, Any]
) -> QualificationOutcome:
    """Run the full preregistered protocol over ``seeds``.

    ``seeds`` and ``implementation_tests`` are evidence inputs, never scientific
    parameters: the formal entrypoint always passes the preregistered
    ``paired_seeds`` list and the exact-commit implementation-test evidence it
    produced itself.  A non-green, unbound or non-exact-commit test gate is
    rejected before any learner is constructed.
    """

    configure_deterministic_runtime()
    validate_spec(spec)
    seed_list = tuple(seeds)
    if not seed_list:
        raise ContractError("run_experiment requires at least one paired seed")
    gate = implementation_test_gate(implementation_tests, commit=commit_sha())
    if not gate["passed"]:
        raise ContractError(
            "the exact-commit implementation test gate is not green; no scientific result "
            "may be produced"
        )
    global_gate = tasks.spec_mapping(spec["global_gate"], "global_gate")
    implementation = {
        **dict(implementation_tests),
        "gate": gate,
        "required_by_global_gate": bool(global_gate["implementation_tests_must_pass"]),
        "runner_executed_suite": True,
    }
    authority = parse_authority(spec)
    optimizer = parse_optimizer(spec)
    env_a = tasks.task_a_environment(spec)
    env_b = tasks.task_b_environment(spec)

    seed_records = [
        run_paired_seed(spec, seed, authority, optimizer, env_a, env_b) for seed in seed_list
    ]
    aggregate_a = task_a_aggregate_gate(seed_records, gate_section(spec, "task_a", "aggregate_gate"))
    aggregate_b = task_b_aggregate_gate(seed_records, gate_section(spec, "task_b", "aggregate_gate"))

    verdict_a = "PASS" if aggregate_a["passed"] else "FAIL"
    verdict_b = "PASS" if aggregate_b["passed"] else "FAIL"
    global_pass = bool(
        aggregate_a["passed"]
        and aggregate_b["passed"]
        and implementation["gate"]["passed"]
        and implementation["required_by_global_gate"]
    )
    classification = "PASS" if global_pass else "SCIENTIFIC_FAIL"

    result = {
        "schema": "cb16.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": commit_sha(),
        "classification": classification,
        "runtime": runtime_record(),
        "implementation_tests": implementation,
        "preregistered_spec": {
            "path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "canonical_sha256": spec_sha256(spec),
            "matches_committed_spec": matches_committed_spec(spec),
            "paired_seeds": list(seed_list),
            "declared_status": spec.get("status"),
        },
        "seeds": seed_records,
        "aggregate_gates": {"task_a": aggregate_a, "task_b": aggregate_b},
        "verdicts": {
            "task_a": verdict_a,
            "task_b": verdict_b,
            "global": "PASS" if global_pass else "FAIL",
        },
    }
    report = render_report(result)
    return QualificationOutcome(
        spec=dict(spec),
        result=result,
        report=report,
        classification=classification,
        exit_code=EXIT_PASS if global_pass else EXIT_SCIENTIFIC_FAIL,
        summary_line=(
            f"{RESULT_COMMAND}: classification={classification} "
            f"task_a={verdict_a} task_b={verdict_b} seeds={len(seed_records)}"
        ),
    )


def write_artifacts(result_dir: Path, outcome: QualificationOutcome) -> None:
    """Write exactly the three preregistered artifacts under ``CB16_RESULT_DIR``."""

    directory = Path(result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SPEC_FILENAME).write_text(
        json.dumps(outcome.spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / RESULT_FILENAME).write_text(
        json.dumps(outcome.result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / REPORT_FILENAME).write_text(outcome.report, encoding="utf-8")


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def _task_a_seed_rows(result: Mapping[str, Any]) -> List[str]:
    rows = [
        "| seed | arm | PRE directions | POST directions | PRE MAE | POST MAE | improvement | seed gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in result["seeds"]:
        for arm in ("positive", "control"):
            block = record["task_a"][arm]
            pre_directions = ",".join(case["direction"] for case in block["pre"]["cases"])
            post_directions = ",".join(case["direction"] for case in block["post"]["cases"])
            improvement = record["task_a"]["paired"][
                "positive_target_mae_improvement"
                if arm == "positive"
                else "control_target_mae_improvement"
            ]
            rows.append(
                f"| {record['seed']} | {arm} | {pre_directions} | {post_directions} | "
                f"{block['pre']['target_exposure_mae']:.6f} | "
                f"{block['post']['target_exposure_mae']:.6f} | {improvement:+.6f} | "
                f"{'PASS' if block['seed_gate']['passed'] else 'FAIL'} |"
            )
    return rows


def _task_b_seed_rows(result: Mapping[str, Any]) -> List[str]:
    rows = [
        "| seed | arm | PRE UP/DOWN | POST UP/DOWN | POST risks | POST growth | PRE->POST growth | margin | seed gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in result["seeds"]:
        for arm in ("positive", "control"):
            block = record["task_b"][arm]
            pre = block["pre"]
            post = block["post"]
            improvement = record["task_b"]["paired"][
                "positive_true_delayed_log_growth_improvement"
                if arm == "positive"
                else "control_true_delayed_log_growth_improvement"
            ]
            rows.append(
                f"| {record['seed']} | {arm} | "
                f"{pre['up']['direction']}/{pre['down']['direction']} | "
                f"{post['up']['direction']}/{post['down']['direction']} | "
                f"{post['up']['requested_risk']:.6f}/{post['down']['requested_risk']:.6f} | "
                f"{post['mean_true_delayed_log_growth']:+.6f} | {improvement:+.6f} | "
                f"{post['direction_probability_margin']:+.6f} | "
                f"{'PASS' if block['seed_gate']['passed'] else 'FAIL'} |"
            )
    return rows


def _gate_component_rows(gate: Mapping[str, Any]) -> List[str]:
    rows = ["| component | value | condition |", "| --- | --- | --- |"]
    for name, value in gate["components"].items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        if name in gate["conditions"]:
            state = "ok" if gate["conditions"][name] else "not met"
        else:
            state = "observed (no condition)"
        rows.append(f"| {name} | {rendered} | {state} |")
    return rows


def render_report(result: Mapping[str, Any]) -> str:
    """Render REPORT.md with the contract's required section separation."""

    runtime = result["runtime"]
    seeds = result["preregistered_spec"]["paired_seeds"]
    lines: List[str] = [
        "# R12 VS-C Controlled Learnability R0 — REPORT",
        "",
        f"- experiment id: `{result['experiment_id']}`",
        f"- result command: `{result['result_command']}`",
        f"- commit: `{result['commit_sha']}`",
        f"- classification: **{result['classification']}**",
        f"- paired seeds: {seeds}",
        f"- runtime: python {runtime['python']}, numpy {runtime['numpy']}, "
        f"torch {runtime['torch']}, device `{runtime['device']}`, "
        f"torch threads {runtime['torch_num_threads']}, "
        f"deterministic algorithms {runtime['deterministic_algorithms']}",
        f"- preregistered spec: `{result['preregistered_spec']['path']}` "
        f"canonical sha256 `{result['preregistered_spec']['canonical_sha256']}` "
        f"(matches committed: {result['preregistered_spec']['matches_committed_spec']})",
        "",
        "## Implementation correctness",
        "",
        "- The formal entrypoint reads only the committed preregistered JSON; no CLI "
        "argument or environment variable can override a scientific parameter.",
        "- Both tasks construct their synthetic windows through the canonical "
        "`normalize_market_window` N0 boundary and the frozen sensory projection.",
        "- Reward comes only from the merged VS-A `execute_transition`; the learner "
        "spine (`OnPolicyLearner`, `Actor`, `ValueCritic`, `FrozenSensory`) is unchanged.",
        "- Each paired seed constructs its positive and matched-control learners "
        "immediately after `torch.manual_seed(seed)`, so they start bitwise identical; "
        "construction is guarded at runtime.",
        "- PRE evaluation happens before any learner update and POST only after exactly "
        "the preregistered generation count; every update consumes exactly one "
        "current-generation batch and advances the generation counter once.",
        f"- exact-commit implementation tests: "
        f"**{result['implementation_tests']['status']}** "
        f"({result['implementation_tests']['tests_run']} tests, exit "
        f"{result['implementation_tests']['exit_code']}, skipped "
        f"{result['implementation_tests']['skipped']}, source sha256 "
        f"`{(result['implementation_tests'].get('source_sha256') or '')[:16]}`) "
        f"for commit `{result['commit_sha']}`",
        "- The repository suite ran against a byte-identical copy of this exact "
        "commit's implementation surface; the gate requires a green suite, an "
        "identity copy and an unchanged source tree, and Global PASS includes it.",
        f"- artifacts: `{SPEC_FILENAME}`, `{RESULT_FILENAME}`, `{REPORT_FILENAME}`; "
        "loss values are diagnostics only and no gate is derived from them.",
        "",
        "## Task A controlled learnability",
        "",
        "Task A asks whether the frozen learner maps an identical flat market plus the "
        "true `AccountState` to the economically correct maintenance target "
        "(`-0.5 / 0.0 / +0.5`).  Direction correctness and target-exposure MAE are "
        "measured with the deterministic adapter on the true account states.",
        "",
        *_task_a_seed_rows(result),
        "",
        "Task A aggregate gate:",
        "",
        *_gate_component_rows(result["aggregate_gates"]["task_a"]),
        "",
        "## Task B delayed-credit learnability",
        "",
        "Task B asks whether the frozen learner turns the delayed step-1 economic "
        "consequence of action 0 into the correct UP=LONG / DOWN=SHORT behavior with "
        "requested risk at or above the preregistered floor.  The step-0 reward is "
        "exactly zero by construction, so only the delayed reward can carry credit.",
        "",
        *_task_b_seed_rows(result),
        "",
        "Task B aggregate gate:",
        "",
        *_gate_component_rows(result["aggregate_gates"]["task_b"]),
        "",
        "The Task B paired condition counts seeds whose POST positive-minus-control "
        "true delayed log growth reaches the preregistered per-seed margin "
        f"`{result['aggregate_gates']['task_b']['preregistered']['paired_positive_growth_minus_control_min']}`; "
        "the smallest observed paired difference is reported above as an observed "
        "component without its own condition.",
        "",
        "## Negative controls",
        "",
        "- Task A control: every actual-account x observed-account pair appears exactly "
        "`batch_trajectories / 9` times, so the two marginals are balanced and exactly "
        "independent; reward/Physics still use the actual `AccountTruth`.  A policy "
        "cannot recover the true account from the observed one.",
        "- Task B control: states and actions are unchanged and the delayed-reward "
        "multiset is preserved exactly, but each trajectory receives another "
        "trajectory's delayed reward through a deterministic seed-and-generation "
        "permutation with no fixed points.  The assignment cannot encode one stable "
        "remapping.",
        "- A positive result that does not separate from its matched control is not "
        "controlled learnability; the aggregate gates therefore require both a "
        "positive pass count and a bounded control pass count.",
        "",
        "## Scientific verdict",
        "",
        f"- Task A aggregate verdict: **{result['verdicts']['task_a']}**",
        f"- Task B aggregate verdict: **{result['verdicts']['task_b']}**",
        f"- exact-commit implementation test gate: "
        f"**{'PASS' if result['implementation_tests']['gate']['passed'] else 'FAIL'}**",
        f"- global verdict: **{result['verdicts']['global']}**",
        f"- classification: **{result['classification']}**",
        "",
        "A gate miss under a correct implementation is `SCIENTIFIC_FAIL`; it is never "
        "permission to change thresholds, budgets, seeds, friction, gap magnitude or "
        "controls under the same R0 identity.",
        "",
        "## Limitations",
        "",
        "- These are controlled synthetic known-answer tasks.  They make no historical "
        "profitability claim and open no final holdout.",
        "- The frozen sensory projection is a controlled fixed random projection, not a "
        "pretrained market model; no learnability claim is attached to it.",
        "- Only eight preregistered seeds and two tasks are qualified; the aggregate "
        "gates are about controlled separation from the matched controls, not about "
        "population-level generalization.",
        "- The exact-commit implementation test gate proves the repository suite was green "
        "for this commit's content; it cannot prove the controlled task specification "
        "itself is free of error.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def result_dir_from_environment() -> Path:
    raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw:
        raise ContractError("CB16_RESULT_DIR is not set")
    return Path(raw)


def required_environment() -> Tuple[Path, str]:
    """The two formal inputs the runner must never invent.

    A result directory is where the artifacts go; the exact commit SHA is what
    makes them evidence for one immutable commit.  Either one missing is an
    environment blocker, not a scientific result.
    """

    result_dir = result_dir_from_environment()
    commit = (os.environ.get("CB16_COMMIT_SHA") or "").strip()
    if not commit:
        raise ContractError(
            "CB16_COMMIT_SHA is not set; the formal artifact must carry the exact commit SHA"
        )
    return result_dir, commit


def run_qualification(*, result_dir: Path) -> QualificationOutcome:
    """Formal path: exact-commit test gate, committed spec, preregistered seeds, artifacts.

    There is deliberately no spec path, seed list, budget, threshold or test
    status parameter here: the formal run is defined entirely by
    ``config/experiments/r12_vs_c_r0.json`` plus implementation-test evidence
    this runner produces itself for the exact commit ``CB16_COMMIT_SHA``.
    """

    spec = load_spec()
    implementation_tests = run_implementation_test_suite()
    gate = implementation_test_gate(implementation_tests, commit=commit_sha())
    if not gate["passed"]:
        outcome = implementation_failure_outcome(spec, implementation_tests, gate)
        write_artifacts(result_dir, outcome)
        return outcome
    outcome = run_experiment(
        spec, seeds=paired_seeds(spec), implementation_tests=implementation_tests
    )
    write_artifacts(result_dir, outcome)
    return outcome


def render_failure_report(
    *,
    classification: str,
    commit: str,
    detail: str,
    implementation_tests: Mapping[str, Any] | None = None,
    gate: Mapping[str, Any] | None = None,
) -> str:
    """REPORT.md for a run that produced no scientific verdict."""

    if implementation_tests is None:
        implementation_lines = ["No implementation-test evidence was recorded for this run."]
    else:
        implementation_lines = [
            f"- exact-commit implementation test status: "
            f"**{implementation_tests.get('status')}** "
            f"(exit {implementation_tests.get('exit_code')}, "
            f"{implementation_tests.get('tests_run')} tests, "
            f"skipped {implementation_tests.get('skipped')}, "
            f"commit `{implementation_tests.get('commit_sha')}`)",
            f"- implementation source sha256 `{implementation_tests.get('source_sha256')}`; "
            f"tested copy `{implementation_tests.get('copy_sha256')}`; "
            f"copied files {implementation_tests.get('copied_file_count')}; "
            f"source unchanged by the test run: "
            f"{implementation_tests.get('source_unchanged')}",
        ]
        if gate is not None:
            unmet = sorted(name for name, ok in gate["conditions"].items() if not ok)
            implementation_lines.append(
                f"- implementation test gate: {'PASS' if gate['passed'] else 'FAIL'}"
                + (f" (unmet conditions: {', '.join(unmet)})" if unmet else "")
            )
    lines = [
        "# R12 VS-C Controlled Learnability R0 — REPORT",
        "",
        f"- classification: **{classification}**",
        f"- commit: `{commit}`",
        "",
        "## Implementation correctness",
        "",
        *implementation_lines,
        "",
        "No scientific result was produced.",
        "",
        "## Task A controlled learnability",
        "",
        "NOT EVALUATED",
        "",
        "## Task B delayed-credit learnability",
        "",
        "NOT EVALUATED",
        "",
        "## Negative controls",
        "",
        "NOT EVALUATED",
        "",
        "## Scientific verdict",
        "",
        f"`{classification}`: {detail}",
        "",
        "## Limitations",
        "",
        "The run did not complete, so no scientific gate component is reported.",
        "",
    ]
    return "\n".join(lines)


def implementation_failure_outcome(
    spec: Mapping[str, Any], evidence: Mapping[str, Any], gate: Mapping[str, Any]
) -> QualificationOutcome:
    """Artifacts for a run blocked by the exact-commit implementation test gate.

    The classification follows :func:`implementation_failure_classification`:
    an executed red suite is ``CONTRACT_MISMATCH`` (the implementation is
    broken); an unexecutable suite is ``EXECUTION_BLOCKED`` (runtime /
    environment blocker).  Either way the run is not a scientific result: Task A
    and Task B stay ``NOT_EVALUATED``, no seeds and no aggregate gates are
    recorded, and no scientific verdict is produced.
    """

    classification, exit_code = implementation_failure_classification(evidence)
    detail = implementation_test_failure_detail(classification, evidence)
    global_gate = spec.get("global_gate")
    required_by_global_gate = (
        bool(global_gate.get("implementation_tests_must_pass"))
        if isinstance(global_gate, Mapping)
        else True
    )
    implementation = {
        **dict(evidence),
        "gate": dict(gate),
        "required_by_global_gate": required_by_global_gate,
        "runner_executed_suite": True,
        "failure_classification": classification,
    }
    result = {
        "schema": "cb16.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": commit_sha(),
        "classification": classification,
        "runtime": runtime_record(),
        "implementation_tests": implementation,
        "preregistered_spec": {
            "path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "canonical_sha256": spec_sha256(spec),
            "matches_committed_spec": matches_committed_spec(spec),
            "paired_seeds": list(paired_seeds(spec)),
            "declared_status": spec.get("status"),
        },
        "seeds": [],
        "aggregate_gates": {},
        "verdicts": {
            "task_a": "NOT_EVALUATED",
            "task_b": "NOT_EVALUATED",
            "global": classification,
        },
        "error": {"type": classification, "detail": detail},
    }
    report = render_failure_report(
        classification=classification,
        commit=result["commit_sha"],
        detail=detail,
        implementation_tests=implementation,
        gate=gate,
    )
    return QualificationOutcome(
        spec=dict(spec),
        result=result,
        report=report,
        classification=classification,
        exit_code=exit_code,
        summary_line=(
            f"{RESULT_COMMAND}: classification={classification} "
            f"implementation_tests={evidence.get('status')}; no scientific verdict produced"
        ),
    )


def _write_failure_artifacts(result_dir: Path, classification: str, detail: str, exit_code: int) -> int:
    """Best-effort evidence when no complete run could be produced."""

    result = {
        "schema": "cb16.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": commit_sha(),
        "classification": classification,
        "runtime": runtime_record(),
        "error": {"type": classification, "detail": detail[:1000]},
        "seeds": [],
        "aggregate_gates": {},
        "verdicts": {"task_a": "NOT_EVALUATED", "task_b": "NOT_EVALUATED", "global": classification},
    }
    report = render_failure_report(
        classification=classification, commit=result["commit_sha"], detail=detail
    )
    try:
        directory = Path(result_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RESULT_FILENAME).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / REPORT_FILENAME).write_text(report, encoding="utf-8")
        try:
            committed = load_spec()
        except ContractError:
            committed = None
        if committed is not None:
            (directory / SPEC_FILENAME).write_text(
                json.dumps(committed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except OSError as exc:  # pragma: no cover - only when the result dir is unwritable
        print(f"{RESULT_COMMAND}: cannot write failure artifacts: {exc}", file=sys.stderr)
    return exit_code


def main() -> int:
    """Science-lane entrypoint.  Takes no arguments and reads no CLI input."""

    try:
        result_dir, _commit = required_environment()
    except ContractError as exc:
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED

    try:
        outcome = run_qualification(result_dir=result_dir)
    except ContractError as exc:
        print(f"{RESULT_COMMAND}: CONTRACT_MISMATCH: {exc}", file=sys.stderr)
        return _write_failure_artifacts(
            result_dir,
            classification="CONTRACT_MISMATCH",
            detail=f"{type(exc).__name__}: {exc}",
            exit_code=EXIT_CONTRACT_MISMATCH,
        )
    except Exception as exc:  # noqa: BLE001 - environment/runtime blocker, not a science result
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _write_failure_artifacts(
            result_dir,
            classification="EXECUTION_BLOCKED",
            detail=f"{type(exc).__name__}: {exc}",
            exit_code=EXIT_EXECUTION_BLOCKED,
        )

    print(outcome.summary_line)
    print(
        f"BUILD_REPORT: {RESULT_COMMAND} classification={outcome.classification}; artifacts "
        f"{SPEC_FILENAME}, {RESULT_FILENAME}, {REPORT_FILENAME} written; no threshold in this "
        "run was chosen after viewing results"
    )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "EXIT_CONTRACT_MISMATCH",
    "EXIT_EXECUTION_BLOCKED",
    "EXIT_PASS",
    "EXIT_SCIENTIFIC_FAIL",
    "EXPERIMENT_ID",
    "IMPLEMENTATION_TEST_ARGV",
    "IMPLEMENTATION_TEST_TIMEOUT_SECONDS",
    "REPORT_FILENAME",
    "RESULT_COMMAND",
    "RESULT_FILENAME",
    "SPEC_FILENAME",
    "Authority",
    "OptimizerSettings",
    "QualificationOutcome",
    "build_learner",
    "commit_sha",
    "configure_deterministic_runtime",
    "copy_implementation_tree",
    "gate_section",
    "implementation_failure_classification",
    "implementation_failure_outcome",
    "implementation_source_sha256",
    "implementation_surface_files",
    "implementation_test_failure_detail",
    "implementation_test_gate",
    "load_spec",
    "main",
    "matches_committed_spec",
    "paired_seeds",
    "parse_authority",
    "parse_optimizer",
    "parse_unittest_summary",
    "render_failure_report",
    "render_report",
    "repository_head",
    "require_identical_parameters",
    "required_environment",
    "result_dir_from_environment",
    "run_experiment",
    "run_implementation_test_suite",
    "run_paired_seed",
    "run_qualification",
    "runtime_record",
    "snapshot_git_repository",
    "spec_sha256",
    "summarise_unittest_run",
    "task_a_aggregate_gate",
    "task_a_seed_gate",
    "task_b_aggregate_gate",
    "task_b_seed_gate",
    "task_section",
    "train_task_a",
    "train_task_b",
    "validate_spec",
    "write_artifacts",
]
