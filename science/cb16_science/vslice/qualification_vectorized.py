"""Vectorized/4-worker execution engine for controlled VS learnability tasks.

This is an execution engine for later experiment identities. It is deliberately
not wired to the already-completed VS-C R0 command: R0's SCIENTIFIC_FAIL is
immutable evidence and must not be rescued by an implementation change.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Dict, Mapping, Sequence

import torch

from . import controlled_tasks as tasks
from . import qualification as q
from . import vectorized_tasks as vt
from .contracts import ContractError

MAX_SEED_WORKERS = 4
ENGINE_ID = "vectorized-batched-v1"


def _train_task_a(learner, env, *, seed: int, mode: str, generations: int) -> Dict[str, Any]:
    records = []
    for generation in range(generations):
        if learner.generation_id != generation:
            raise ContractError(
                f"learner generation {learner.generation_id} != scheduled {generation}"
            )
        torch.manual_seed(q._collection_seed(seed, "task_a", mode, generation))
        batch = vt.build_task_a_one_step_batch(learner, env, generation, mode=mode)
        report = learner.update_one_step_batch(batch)
        records.append({"actor_loss": report.actor_loss, "critic_loss": report.critic_loss})
    return q._diagnostics_summary(records)


def _train_task_b(learner, env, *, seed: int, mode: str, generations: int) -> Dict[str, Any]:
    records = []
    for generation in range(generations):
        if learner.generation_id != generation:
            raise ContractError(
                f"learner generation {learner.generation_id} != scheduled {generation}"
            )
        torch.manual_seed(q._collection_seed(seed, "task_b", mode, generation))
        batch = vt.build_task_b_generation(
            learner, env, generation, mode=mode, seed=seed
        )
        report = learner.update(batch)
        records.append({"actor_loss": report.actor_loss, "critic_loss": report.critic_loss})
    return q._diagnostics_summary(records)


def run_paired_seed_vectorized(
    spec: Mapping[str, Any],
    seed: int,
    authority: q.Authority,
    optimizer: q.OptimizerSettings,
    env_a: tasks.TaskAEnvironment,
    env_b: tasks.TaskBEnvironment,
) -> Dict[str, Any]:
    task_a = q.task_section(spec, "task_a")
    task_b = q.task_section(spec, "task_b")
    generations_a = tasks.spec_int(task_a["generations"], "task_a.generations", minimum=1)
    generations_b = tasks.spec_int(task_b["generations"], "task_b.generations", minimum=1)
    gate_a = q.gate_section(spec, "task_a", "seed_pass_gate")
    gate_b = q.gate_section(spec, "task_b", "seed_pass_gate")

    torch.manual_seed(seed)
    positive_a = q.build_learner(authority, optimizer)
    torch.manual_seed(seed)
    control_a = q.build_learner(authority, optimizer)
    q.require_identical_parameters(positive_a, control_a)
    pre_positive_a = tasks.evaluate_task_a(positive_a, env_a)
    pre_control_a = tasks.evaluate_task_a(control_a, env_a)
    diagnostics_positive_a = _train_task_a(
        positive_a, env_a, seed=seed, mode=tasks.ARM_POSITIVE, generations=generations_a
    )
    diagnostics_control_a = _train_task_a(
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

    torch.manual_seed(seed)
    positive_b = q.build_learner(authority, optimizer)
    torch.manual_seed(seed)
    control_b = q.build_learner(authority, optimizer)
    q.require_identical_parameters(positive_b, control_b)
    pre_positive_b = tasks.evaluate_task_b(positive_b, env_b)
    pre_control_b = tasks.evaluate_task_b(control_b, env_b)
    diagnostics_positive_b = _train_task_b(
        positive_b, env_b, seed=seed, mode=tasks.ARM_POSITIVE, generations=generations_b
    )
    diagnostics_control_b = _train_task_b(
        control_b, env_b, seed=seed, mode=tasks.ARM_CONTROL, generations=generations_b
    )
    post_positive_b = tasks.evaluate_task_b(positive_b, env_b)
    post_control_b = tasks.evaluate_task_b(control_b, env_b)
    paired_b = {
        "pre_positive_true_delayed_log_growth": pre_positive_b["mean_true_delayed_log_growth"],
        "post_positive_true_delayed_log_growth": post_positive_b["mean_true_delayed_log_growth"],
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
                "seed_gate": q.task_a_seed_gate(post_positive_a, gate_a),
                "diagnostics": diagnostics_positive_a,
            },
            "control": {
                "pre": pre_control_a,
                "post": post_control_a,
                "seed_gate": q.task_a_seed_gate(post_control_a, gate_a),
                "diagnostics": diagnostics_control_a,
            },
            "paired": paired_a,
        },
        "task_b": {
            "positive": {
                "pre": pre_positive_b,
                "post": post_positive_b,
                "seed_gate": q.task_b_seed_gate(post_positive_b, gate_b),
                "diagnostics": diagnostics_positive_b,
            },
            "control": {
                "pre": pre_control_b,
                "post": post_control_b,
                "seed_gate": q.task_b_seed_gate(post_control_b, gate_b),
                "diagnostics": diagnostics_control_b,
            },
            "paired": paired_b,
        },
    }


def _seed_worker(payload: tuple[dict[str, Any], int]) -> Dict[str, Any]:
    spec, seed = payload
    q.configure_deterministic_runtime()
    authority = q.parse_authority(spec)
    optimizer = q.parse_optimizer(spec)
    env_a = tasks.task_a_environment(spec)
    env_b = tasks.task_b_environment(spec)
    return run_paired_seed_vectorized(spec, seed, authority, optimizer, env_a, env_b)


def run_seed_set_vectorized(
    spec: Mapping[str, Any], seeds: Sequence[int], *, workers: int = MAX_SEED_WORKERS
) -> list[Dict[str, Any]]:
    """Run independent paired seeds with at most four one-thread ARM workers.

    ``executor.map`` preserves the preregistered seed order in returned evidence
    even though workers execute concurrently.
    """
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ContractError("seed set must be non-empty")
    worker_count = max(1, min(int(workers), MAX_SEED_WORKERS, len(seed_list)))
    if worker_count == 1:
        return [_seed_worker((dict(spec), seed)) for seed in seed_list]
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        return list(executor.map(_seed_worker, [(dict(spec), seed) for seed in seed_list]))


__all__ = [
    "ENGINE_ID",
    "MAX_SEED_WORKERS",
    "run_paired_seed_vectorized",
    "run_seed_set_vectorized",
]
