"""Deterministic dispatcher tests for the CB16-R12 GitHub -> OCI bridge.

Runs with the standard library only:

    python3 -m unittest discover -s tests -t .

The suite covers the twelve-plus verification points required by the OCI
bootstrap task: metadata validation, lane separation, shell-text refusal,
labelling rules, worktree isolation, concurrency, credential scrubbing,
control-plane detection, evidence collection, deterministic classification,
and dry-run coverage.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DISPATCH_PATH = REPO_ROOT / "scripts" / "cb16_dispatch.py"
ALLOWLIST_PATH = REPO_ROOT / "config" / "cb16_science_allowlist.json"
TRUSTED_REPO = "GY-Bai/CB16-R12"


def load_dispatcher():
    spec = importlib.util.spec_from_file_location("cb16_dispatch_under_test", DISPATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves annotations through sys.modules on Python 3.9, so the
    # module must be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dispatcher = load_dispatcher()

_counter = itertools.count()


def metadata_body(meta: dict) -> str:
    lines = "\n".join(f"{key}: {value}" for key, value in meta.items())
    return (
        "## Trusted dispatch metadata\n\n"
        f"```cb16\n{lines}\n```\n\n"
        "# Objective\n\nA bounded task.\n\n# Done when\n\nIt is done.\n"
    )


def make_event(
    body: str,
    *,
    label: str = "ds:run",
    action: str = "labeled",
    number: int = 7,
    sender: str = "GY-Bai",
    repo: str = TRUSTED_REPO,
) -> dict:
    return {
        "action": action,
        "label": {"name": label},
        "issue": {"number": number, "body": body, "title": "[R12] test task"},
        "repository": {"full_name": repo, "default_branch": "main"},
        "sender": {"login": sender},
    }


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


class DispatchTestCase(unittest.TestCase):
    """Shared fixture: a throwaway git repository plus a throwaway environment."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cb16-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo, self.sha = self._make_repo()
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.token_primary = "ghp_primarytokenvalue000000"
        self.token_secondary = "ghp_secondarytokenvalue111"
        self.base_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "GITHUB_TOKEN": self.token_primary,
            "CB16_GITHUB_TOKEN": self.token_secondary,
            "AWS_SECRET_ACCESS_KEY": "aws-not-a-real-secret",
            "SSH_AUTH_SOCK": "/tmp/cb16-test-agent.sock",
            "MY_SERVICE_API_KEY": "service-not-a-real-key",
            "DEEPSEEK_API_KEY": "deepseek-not-a-real-key",
        }

    # -- fixtures ---------------------------------------------------------

    def _make_repo(self):
        repo = self.tmp / "repo"
        (repo / "docs" / "tasks").mkdir(parents=True)
        (repo / "config").mkdir(parents=True)
        (repo / "tests").mkdir(parents=True)
        shutil.copy(REPO_ROOT / ".gitignore", repo / ".gitignore")
        shutil.copy(ALLOWLIST_PATH, repo / "config" / "cb16_science_allowlist.json")
        shutil.copytree(REPO_ROOT / "science", repo / "science")
        (repo / "docs" / "tasks" / "TASK.md").write_text(
            "# Objective\n\nBounded task.\n", encoding="utf-8"
        )
        (repo / "docs" / "TASK_SPEC.json").write_text("{}\n", encoding="utf-8")
        (repo / "README.md").write_text("# fixture\n", encoding="utf-8")
        (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (repo / "tests" / "test_fixture_smoke.py").write_text(
            "import unittest\n\n\n"
            "class FixtureSmoke(unittest.TestCase):\n"
            "    def test_fixture_repo_is_healthy(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        git(repo, "init", "-q", "-b", "main")
        git(repo, "config", "user.email", "fixture@example.com")
        git(repo, "config", "user.name", "Fixture")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "fixture")
        return repo, git(repo, "rev-parse", "HEAD")

    def builder_meta(self, **overrides) -> dict:
        meta = {
            "mode": "build",
            "base_sha": self.sha,
            "branch": "ds/test-task",
            "task_file": "docs/tasks/TASK.md",
        }
        meta.update(overrides)
        return meta

    def science_meta(self, **overrides) -> dict:
        meta = {
            "mode": "science",
            "commit_sha": self.sha,
            "experiment_spec": "docs/TASK_SPEC.json",
            "result_command": "cb16.noop@v1",
        }
        meta.update(overrides)
        return meta

    # -- dispatch driver --------------------------------------------------

    def dispatch(
        self,
        *,
        lane: str = "builder",
        meta: dict | None = None,
        event: dict | None = None,
        dry_run: bool = False,
        publish: bool = False,
        dsh_invoker=None,
        science_invoker=None,
        token: str | None = None,
    ):
        if event is None:
            payload = meta if meta is not None else (
                self.builder_meta() if lane == "builder" else self.science_meta()
            )
            label = "ds:run" if lane == "builder" else "science:run"
            event = make_event(metadata_body(payload), label=label)
        report_dir = self.tmp / f"report-{next(_counter)}"
        return dispatcher.dispatch(
            lane=lane,
            event=event,
            repo_dir=self.repo,
            work_root=self.tmp / "worktrees",
            state_dir=self.tmp / "state",
            report_dir=report_dir,
            allowlist_path=ALLOWLIST_PATH,
            repo=TRUSTED_REPO,
            trusted_actors=("GY-Bai",),
            dry_run=dry_run,
            publish=publish,
            github_token=token,
            dsh_invoker=dsh_invoker,
            science_invoker=science_invoker,
            base_env=self.base_env,
        )

    @staticmethod
    def builder_spy(changes: dict | None = None, *, exit_code: int = 0, report: str = "BUILD_REPORT: ok\n"):
        """Fake Builder lane.  Records the child environment it was handed."""

        captured: dict = {}

        def spy(worktree, **kwargs):
            captured.update(kwargs)
            captured["worktree"] = Path(worktree)
            for relative, content in (changes or {}).items():
                target = Path(worktree) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            return dispatcher.RunResult(exit_code=exit_code, stdout=report)

        spy.captured = captured  # type: ignore[attr-defined]
        return spy

    @staticmethod
    def science_spy(exit_code: int = 0, *, write_artifacts: bool = True, report: str = "BUILD_REPORT: ok\n"):
        captured: dict = {}

        def spy(worktree, spec, **kwargs):
            captured.update(kwargs)
            captured["spec"] = spec
            result_dir = Path(kwargs["result_dir"])
            result_dir.mkdir(parents=True, exist_ok=True)
            if write_artifacts:
                (result_dir / "RESULT.json").write_text('{"status": "DRY_RUN"}\n', encoding="utf-8")
                (result_dir / "REPORT.md").write_text("# report\n", encoding="utf-8")
            return dispatcher.RunResult(exit_code=exit_code, stdout=report)

        spy.captured = captured  # type: ignore[attr-defined]
        return spy


# ---------------------------------------------------------------------------
# 1-3: metadata validation
# ---------------------------------------------------------------------------


class MetadataValidationTests(DispatchTestCase):
    def test_malformed_metadata_lines_are_rejected(self):
        for block in (
            "mode build",  # missing colon
            "mode: build\nmode: fix",  # duplicate key
            "unknown_key: value",  # unknown key
            "mode:",  # empty value
            "",  # empty block
        ):
            body = f"```cb16\n{block}\n```\n"
            with self.subTest(block=block):
                with self.assertRaises(dispatcher.ContractMismatch):
                    dispatcher.parse_metadata_block(dispatcher.extract_metadata_block(body))

    def test_missing_metadata_block_is_rejected(self):
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.extract_metadata_block("# Objective\n\nno block here\n")

    def test_two_metadata_blocks_are_rejected(self):
        body = "```cb16\nmode: build\n```\n\n```cb16\nmode: fix\n```\n"
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.extract_metadata_block(body)

    def test_missing_base_sha_is_rejected(self):
        meta = self.builder_meta()
        meta.pop("base_sha")
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.validate_metadata(meta, "builder")

    def test_missing_commit_sha_is_rejected(self):
        meta = self.science_meta()
        meta.pop("commit_sha")
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.validate_metadata(meta, "science", allowlist=dispatcher.load_allowlist(ALLOWLIST_PATH))

    def test_non_sha_revision_is_rejected(self):
        for value in ("main", "HEAD", "de2483a", "z" * 40):
            with self.subTest(value=value):
                with self.assertRaises(dispatcher.ContractMismatch):
                    dispatcher.validate_metadata(self.builder_meta(base_sha=value), "builder")

    def test_invalid_branch_names_are_rejected(self):
        for branch in (
            "main",
            "master",
            "feature/thing",  # no ds/ or science/ prefix
            "ds/../escape",
            "ds/x;touch /tmp/pwned",
            "ds/x`id`",
            "ds/x$(id)",
            "ds/",
            "-ds/leading-dash",
            "ds/x.lock",
            "ds/x//y",
        ):
            with self.subTest(branch=branch):
                with self.assertRaises(dispatcher.ContractMismatch):
                    dispatcher.validate_metadata(self.builder_meta(branch=branch), "builder")

    def test_valid_branch_names_are_accepted(self):
        for branch in ("ds/task-1", "ds/r12.dispatch_bootstrap", "science/run-abc123"):
            with self.subTest(branch=branch):
                spec = dispatcher.validate_metadata(self.builder_meta(branch=branch), "builder")
                self.assertEqual(spec.branch, branch)

    def test_escaping_task_file_paths_are_rejected(self):
        for path in ("/etc/passwd", "../outside.md", "docs/../../etc/passwd", "docs/tasks/x.md\nrm"):
            with self.subTest(path=path):
                with self.assertRaises(dispatcher.ContractMismatch):
                    dispatcher.validate_metadata(self.builder_meta(task_file=path), "builder")


# ---------------------------------------------------------------------------
# 4: shell text is never executed
# ---------------------------------------------------------------------------


class ShellTextTests(DispatchTestCase):
    def test_shell_text_in_issue_metadata_is_refused_and_not_executed(self):
        marker = self.tmp / "PWNED"
        payloads = (
            self.builder_meta(branch=f"ds/x; touch {marker}"),
            self.builder_meta(task_file=f"docs/tasks/TASK.md; touch {marker}"),
            self.builder_meta(mode=f"build; touch {marker}"),
        )
        for meta in payloads:
            with self.subTest(meta=meta):
                with self.assertRaises(dispatcher.ContractMismatch):
                    self.dispatch(meta=meta)
        self.assertFalse(marker.exists(), "Issue text must never reach a shell")

    def test_science_result_command_rejects_shell_text(self):
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.validate_metadata(
                self.science_meta(result_command="cb16.noop@v1; rm -rf /"),
                "science",
                allowlist=dispatcher.load_allowlist(ALLOWLIST_PATH),
            )


# ---------------------------------------------------------------------------
# 5-6: label gating and lane separation
# ---------------------------------------------------------------------------


class LabelAndLaneTests(DispatchTestCase):
    def test_unsupported_labels_do_not_dispatch(self):
        for label in ("enhancement", "sol:review", "blocked", "ds:run-extra", ""):
            event = make_event(metadata_body(self.builder_meta()), label=label)
            with self.subTest(label=label):
                with self.assertRaises(dispatcher.ContractMismatch):
                    self.dispatch(event=event)

    def test_unlabeled_action_does_not_dispatch(self):
        event = make_event(metadata_body(self.builder_meta()), action="unlabeled")
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(event=event)

    def test_untrusted_sender_does_not_dispatch(self):
        event = make_event(metadata_body(self.builder_meta()), sender="random-user")
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(event=event)

    def test_foreign_repository_event_does_not_dispatch(self):
        event = make_event(metadata_body(self.builder_meta()), repo="attacker/fork")
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(event=event)

    def test_label_lane_cannot_be_confused(self):
        # A ds:run label must never drive the science lane and vice versa.
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(lane="science", meta=self.science_meta(),
                          event=make_event(metadata_body(self.science_meta()), label="ds:run"))
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(lane="builder", meta=self.builder_meta(),
                          event=make_event(metadata_body(self.builder_meta()), label="science:run"))

    def test_lane_specific_modes_are_enforced(self):
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.validate_metadata(self.builder_meta(mode="science"), "builder")
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.validate_metadata(self.science_meta(mode="build"), "science")

    def test_builder_and_science_fields_cannot_be_mixed(self):
        mixed_builder = self.builder_meta()
        mixed_builder["commit_sha"] = self.sha
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.validate_metadata(mixed_builder, "builder")
        mixed_science = self.science_meta()
        mixed_science["task_file"] = "docs/tasks/TASK.md"
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.validate_metadata(mixed_science, "science")

    def test_missing_commit_is_rejected_before_any_work(self):
        meta = self.builder_meta(base_sha="0" * 40)
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(meta=meta)


# ---------------------------------------------------------------------------
# 7-8: Science lane cannot reach DSH and only runs allowlisted entrypoints
# ---------------------------------------------------------------------------


class ScienceLaneTests(DispatchTestCase):
    def test_science_lane_never_invokes_dsh(self):
        def forbidden_dsh(*args, **kwargs):
            raise AssertionError("the Science lane must never invoke DSH")

        spy = self.science_spy()
        outcome = self.dispatch(
            lane="science", dry_run=False, dsh_invoker=forbidden_dsh, science_invoker=spy
        )
        self.assertEqual(outcome.classification, dispatcher.CLASS_OK)
        self.assertIn("spec", spy.captured)

    def test_science_entrypoint_source_has_no_dsh_call(self):
        # Structural check on the compiled function: no name it can call refers
        # to DSH, so the Science lane has no route to the Builder agent.
        names = set(dispatcher.run_science_entrypoint.__code__.co_names)
        self.assertFalse([name for name in names if "dsh" in name.lower()])
        self.assertNotIn("invoke_dsh", names)
        self.assertNotIn("dsh_bin", names)

    def test_science_only_runs_allowlisted_entrypoints(self):
        meta = self.science_meta(result_command="cb16.arbitrary@v1")
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(lane="science", meta=meta)

    def test_dry_run_only_entrypoint_is_blocked_in_formal_runs(self):
        allowlist = dispatcher.load_allowlist(ALLOWLIST_PATH)
        spec = dispatcher.validate_metadata(self.science_meta(), "science", allowlist=allowlist)
        with self.assertRaises(dispatcher.ContractMismatch):
            dispatcher.run_science_entrypoint(
                self.repo,
                spec,
                allowlist=allowlist,
                env={"PATH": os.environ.get("PATH", "")},
                result_dir=self.tmp / "formal-results",
            )

    def test_science_lane_rejects_worktree_mutation(self):
        spy = self.science_spy()

        def mutating(*args, **kwargs):
            result = spy(*args, **kwargs)
            worktree = Path(args[0])
            (worktree / "src").mkdir(parents=True, exist_ok=True)
            (worktree / "src" / "leak.py").write_text("# mutated\n", encoding="utf-8")
            return result

        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(lane="science", dry_run=True, science_invoker=mutating)


# ---------------------------------------------------------------------------
# 9: concurrency
# ---------------------------------------------------------------------------


class ConcurrencyTests(DispatchTestCase):
    def test_same_issue_cannot_run_twice_concurrently(self):
        state = self.tmp / "state"
        with dispatcher.TaskLock(state, 7, "builder"):
            with self.assertRaises(dispatcher.ExecutionBlocked):
                with dispatcher.TaskLock(state, 7, "builder"):
                    pass

    def test_different_issues_use_independent_locks(self):
        state = self.tmp / "state"
        with dispatcher.TaskLock(state, 7, "builder"):
            with dispatcher.TaskLock(state, 8, "builder"):
                pass


# ---------------------------------------------------------------------------
# 10: credential boundary
# ---------------------------------------------------------------------------


class CredentialBoundaryTests(DispatchTestCase):
    def test_scrubbed_env_drops_every_token_shaped_variable(self):
        env = dispatcher.scrubbed_env(self.base_env)
        for key in (
            "GITHUB_TOKEN",
            "CB16_GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "SSH_AUTH_SOCK",
            "MY_SERVICE_API_KEY",
        ):
            self.assertNotIn(key, env)
        self.assertEqual(env["DEEPSEEK_API_KEY"], "deepseek-not-a-real-key")
        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(env["GIT_ASKPASS"], "/bin/true")
        self.assertEqual(env["GIT_CONFIG_GLOBAL"], "/dev/null")

    def test_dsh_child_never_sees_github_credentials(self):
        spy = self.builder_spy({"docs/dispatch_smoke/DRY_RUN_FIXTURE.md": "# fixture\n"})
        self.dispatch(dsh_invoker=spy)
        handed = spy.captured["env"]
        self.assertNotIn("GITHUB_TOKEN", handed)
        self.assertNotIn("CB16_GITHUB_TOKEN", handed)
        self.assertEqual(handed["DEEPSEEK_API_KEY"], "deepseek-not-a-real-key")

    def test_no_token_is_written_into_the_worktree(self):
        spy = self.builder_spy({"docs/dispatch_smoke/DRY_RUN_FIXTURE.md": "# fixture\n"})
        self.dispatch(dsh_invoker=spy)
        worktree = spy.captured["worktree"]
        for path in worktree.rglob("*"):
            if path.is_file() and ".git/" not in str(path):
                text = path.read_text(encoding="utf-8", errors="ignore")
                self.assertNotIn(self.token_primary, text)
                self.assertNotIn(self.token_secondary, text)

    def test_host_credential_warnings_are_reported(self):
        (self.home / ".ssh").mkdir(parents=True, exist_ok=True)
        (self.home / ".ssh" / "id_ed25519").write_text("not-a-real-key\n", encoding="utf-8")
        warnings = dispatcher.credential_warnings(self.home)
        self.assertTrue(any("SSH private keys" in w for w in warnings))


# ---------------------------------------------------------------------------
# 11: control-plane protection
# ---------------------------------------------------------------------------


class ControlPlaneTests(DispatchTestCase):
    def test_control_plane_violations_are_detected(self):
        hits = dispatcher.control_plane_violations(
            [".github/workflows/evil.yml", "README.md", "scripts/cb16_dispatch.py"]
        )
        self.assertEqual(hits, [".github/workflows/evil.yml", "scripts/cb16_dispatch.py"])

    def test_unauthorised_control_plane_edit_is_rejected(self):
        spy = self.builder_spy({".github/workflows/evil.yml": "name: evil\n"})
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(dsh_invoker=spy)

    def test_authorised_control_plane_edit_is_allowed(self):
        spy = self.builder_spy({".github/workflows/authorised.yml": "name: ok\n"})
        outcome = self.dispatch(
            meta=self.builder_meta(allow_control_plane="true"), dsh_invoker=spy
        )
        self.assertEqual(outcome.classification, dispatcher.CLASS_OK)

    def test_packet_directory_is_not_reported_as_a_violation(self):
        self.assertEqual(dispatcher.control_plane_violations([".cb16/TASK_PACKET.md"]), [])


# ---------------------------------------------------------------------------
# 12-13: evidence and deterministic classification
# ---------------------------------------------------------------------------


class EvidenceAndClassificationTests(DispatchTestCase):
    def test_build_report_is_preserved_on_success(self):
        spy = self.builder_spy(
            {"docs/dispatch_smoke/DRY_RUN_FIXTURE.md": "# fixture\n"},
            report="did work\n\nBUILD_REPORT\nTask: demo\nChanged: one file\n",
        )
        outcome = self.dispatch(dsh_invoker=spy)
        self.assertIn("BUILD_REPORT", outcome.build_report)
        self.assertIn("Task: demo", outcome.build_report)
        self.assertTrue((outcome.evidence["summary"]).exists())
        self.assertTrue((outcome.evidence["build_report"]).exists())
        self.assertTrue((outcome.evidence["lane_stdout"]).exists())

    def test_build_report_is_synthesised_when_the_agent_omits_it(self):
        spy = self.builder_spy({"docs/dispatch_smoke/DRY_RUN_FIXTURE.md": "# fixture\n"}, report="no report here")
        outcome = self.dispatch(dsh_invoker=spy)
        self.assertIn("BUILD_REPORT", outcome.build_report)
        self.assertIn(dispatcher.CLASS_OK, outcome.build_report)

    def test_earlier_prose_mention_does_not_displace_the_report(self):
        # A mention inside the opening prose must not hijack the captured
        # section; capture starts at the line that actually opens the report.
        text = (
            "I will finish by writing a `BUILD_REPORT` below.\n\n"
            "## BUILD_REPORT\nTask: demo\nChanged: one file\n"
        )
        report = dispatcher.extract_build_report(text)
        self.assertIsNotNone(report)
        self.assertTrue(report.startswith("## BUILD_REPORT"))
        self.assertIn("Task: demo", report)
        self.assertNotIn("I will finish", report)

    def test_report_marker_inside_prose_only_is_not_a_report(self):
        text = "I did not produce a BUILD_REPORT section this time.\n"
        self.assertIsNone(dispatcher.extract_build_report(text))

    def test_evidence_is_written_on_failure(self):
        spy = self.builder_spy(exit_code=9, report="")
        outcome = self.dispatch(dsh_invoker=spy)
        self.assertEqual(outcome.classification, dispatcher.CLASS_BUILDER_FAIL)
        self.assertEqual(outcome.exit_code, dispatcher.EXIT_BUILDER_FAIL)
        self.assertTrue((outcome.evidence["build_report"]).exists())
        self.assertIn(dispatcher.CLASS_BUILDER_FAIL, outcome.build_report)

    def test_builder_failure_is_not_reported_as_a_scientific_failure(self):
        spy = self.builder_spy(exit_code=1)
        outcome = self.dispatch(dsh_invoker=spy)
        self.assertNotEqual(outcome.classification, dispatcher.CLASS_SCIENTIFIC_FAIL)

    def test_timeout_classifies_as_hardware_limit(self):
        def timing_out(worktree, **kwargs):
            return dispatcher.RunResult(exit_code=124, timed_out=True)

        outcome = self.dispatch(dsh_invoker=timing_out)
        self.assertEqual(outcome.classification, dispatcher.CLASS_HARDWARE_LIMIT)

    def test_missing_science_artifacts_classify_as_evidence_insufficient(self):
        spy = self.science_spy(write_artifacts=False)
        outcome = self.dispatch(lane="science", dry_run=True, science_invoker=spy)
        self.assertEqual(outcome.classification, dispatcher.CLASS_EVIDENCE_INSUFFICIENT)

    def test_failing_science_entrypoint_classifies_as_scientific_fail(self):
        spy = self.science_spy(exit_code=3)
        outcome = self.dispatch(lane="science", dry_run=True, science_invoker=spy)
        self.assertEqual(outcome.classification, dispatcher.CLASS_SCIENTIFIC_FAIL)

    def test_missing_event_file_classifies_as_execution_blocked(self):
        with self.assertRaises(dispatcher.ExecutionBlocked):
            dispatcher.load_event(self.tmp / "missing.json")

    def test_unsupported_command_label_is_contract_mismatch(self):
        event = make_event(metadata_body(self.builder_meta()), label="blocked")
        with self.assertRaises(dispatcher.ContractMismatch) as ctx:
            self.dispatch(event=event)
        self.assertEqual(ctx.exception.classification, dispatcher.CLASS_CONTRACT_MISMATCH)
        self.assertEqual(ctx.exception.exit_code, dispatcher.EXIT_CONTRACT_MISMATCH)


# ---------------------------------------------------------------------------
# 14: dry runs
# ---------------------------------------------------------------------------


class DryRunTests(DispatchTestCase):
    def test_builder_dry_run_exercises_the_full_path(self):
        outcome = self.dispatch(dry_run=True)
        self.assertEqual(outcome.classification, dispatcher.CLASS_OK)
        self.assertIn("docs/dispatch_smoke/DRY_RUN_FIXTURE.md", outcome.summary["changed_files"])
        self.assertTrue(outcome.summary["dry_run"])
        self.assertTrue(outcome.summary["worktree_reused"] is False)

    def test_science_dry_run_runs_the_allowlisted_entrypoint(self):
        outcome = self.dispatch(lane="science", dry_run=True)
        self.assertEqual(outcome.classification, dispatcher.CLASS_OK)
        result_dir = outcome.evidence["summary"].parent / "results"
        self.assertTrue((result_dir / "RESULT.json").exists())
        self.assertTrue((result_dir / "REPORT.md").exists())

    def test_science_dry_run_artifacts_are_deterministic(self):
        first = self.dispatch(lane="science", dry_run=True)
        second = self.dispatch(lane="science", dry_run=True)
        first_result = (first.evidence["summary"].parent / "results" / "RESULT.json").read_text()
        second_result = (second.evidence["summary"].parent / "results" / "RESULT.json").read_text()
        self.assertEqual(first_result, second_result)

    def test_dry_run_does_not_publish(self):
        outcome = self.dispatch(dry_run=True, publish=True)
        self.assertFalse(outcome.summary["published"])


# ---------------------------------------------------------------------------
# Publishing helpers
# ---------------------------------------------------------------------------


class PublishingTests(DispatchTestCase):
    def test_existing_open_pr_is_updated_instead_of_failing(self):
        calls = []

        def fake_api(method, path, *, token, payload=None, timeout=30):
            calls.append((method, path))
            if method == "POST":
                raise dispatcher.ExecutionBlocked("HTTP 422 validation failed")
            if method == "GET":
                return 200, [{"number": 42, "html_url": "https://example.invalid/pr/42"}]
            return 200, {"number": 42, "html_url": "https://example.invalid/pr/42"}

        original = dispatcher.github_api
        dispatcher.github_api = fake_api  # type: ignore[assignment]
        try:
            result = dispatcher.create_or_update_draft_pr(
                token="t", slug="owner/repo", branch="ds/x", base="main",
                title="title", body="body",
            )
        finally:
            dispatcher.github_api = original  # type: ignore[assignment]

        self.assertEqual(result["number"], 42)
        self.assertIn(("GET", "/repos/owner/repo/pulls?state=open&head=owner:ds/x"), calls)
        self.assertIn(("PATCH", "/repos/owner/repo/pulls/42"), calls)

    def test_explicit_pr_number_updates_without_creating(self):
        calls = []

        def fake_api(method, path, *, token, payload=None, timeout=30):
            calls.append((method, path))
            return 200, {"number": 7, "html_url": "https://example.invalid/pr/7"}

        original = dispatcher.github_api
        dispatcher.github_api = fake_api  # type: ignore[assignment]
        try:
            dispatcher.create_or_update_draft_pr(
                token="t", slug="owner/repo", branch="ds/x", base="main",
                title="title", body="body", pr_number=7,
            )
        finally:
            dispatcher.github_api = original  # type: ignore[assignment]

        self.assertEqual(calls, [("PATCH", "/repos/owner/repo/pulls/7")])


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------


class WorktreeTests(DispatchTestCase):
    def test_one_worktree_per_task_branch_and_reuse(self):
        spy = self.builder_spy({"docs/dispatch_smoke/DRY_RUN_FIXTURE.md": "# fixture\n"})
        first = self.dispatch(dsh_invoker=spy)
        second = self.dispatch(dsh_invoker=spy)
        self.assertFalse(first.summary["worktree_reused"])
        self.assertTrue(second.summary["worktree_reused"])
        self.assertEqual(first.summary["worktree"], second.summary["worktree"])

    def test_worktree_is_rooted_at_the_task_branch(self):
        spy = self.builder_spy({"docs/dispatch_smoke/DRY_RUN_FIXTURE.md": "# fixture\n"})
        outcome = self.dispatch(dsh_invoker=spy)
        worktree = Path(outcome.summary["worktree"])
        self.assertEqual(git(worktree, "rev-parse", "--abbrev-ref", "HEAD"), "ds/test-task")

    def test_task_packet_is_materialised_and_gitignored(self):
        spy = self.builder_spy({"docs/dispatch_smoke/DRY_RUN_FIXTURE.md": "# fixture\n"})
        outcome = self.dispatch(dsh_invoker=spy)
        packet = Path(outcome.summary["worktree"]) / ".cb16" / "TASK_PACKET.md"
        self.assertTrue(packet.exists())
        text = packet.read_text(encoding="utf-8")
        self.assertIn("Task branch: ds/test-task", text)
        self.assertIn("Base SHA", text)

    def test_main_branch_is_never_the_task_branch(self):
        with self.assertRaises(dispatcher.ContractMismatch):
            self.dispatch(meta=self.builder_meta(branch="main"))


if __name__ == "__main__":
    unittest.main()
