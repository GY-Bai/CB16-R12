import { randomUUID } from "node:crypto";
import z from "@deepseek-ai/schemastery";
import { installModelSelection } from "@deepseek-ai/dsh-agent";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { SessionId } from "@deepseek-ai/dsh-session";

/**
 * CB16 Builder session runner.
 *
 * This is the one-shot headless driver with exactly one behaviour changed:
 * session identity. `--new` mints an id, `--session <id>` continues that exact
 * persisted session, and a failed continue falls back to a fresh session in the
 * same worktree rather than failing the task.
 *
 * Git stays authoritative. A resumed conversation is context only: the
 * dispatcher's follow-up prompt re-asserts the current worktree, task packet and
 * review delta, so a stale or missing session can only cost continuity, never
 * correctness.
 *
 * The driver is modelled on `@deepseek-ai/dsh-headless` (MIT); see NOTICE.
 *
 * @module dsh-cb16-builder-session
 */

/** Stable Cordis plugin name. */
const name = "cb16-session-runner";

/** Services required before the turn can start. */
const inject = ["agentDefaultModel", "agents", "sessions", "sessionPersistence"];

/** Validated runner config, supplied by the bundle patch from the startup service. */
const Config = z.object({
  mode: z.union(["new", "resume"]).required(),
  sessionId: z.string().default(""),
  task: z.string().required(),
  outputFormat: z.string().default("text")
});

/** Why a continue attempt was abandoned; recorded so a fallback is diagnosable. */
const RESUME_FAILURE = {
  NOT_FOUND: "session-not-found",
  CWD_MISMATCH: "cwd-mismatch",
  PERSISTENCE_UNAVAILABLE: "persistence-unavailable",
  INSPECT_FAILED: "inspect-failed",
  RESUME_REJECTED: "resume-rejected"
};

/** The process streams the runner writes to; tests substitute captures. */
const internals = { stdout: process.stdout, stderr: process.stderr };

/** Aggregate the last assistant text and turn outcome in one owned interval. */
function summarize(events, firstSeq) {
  let started = false;
  let text = "";
  let reason;
  for (const event of events) {
    if (event.seq < firstSeq) continue;
    if (event.type === "turn/start") {
      started = true;
      continue;
    }
    if (!started) continue;
    if (event.type === "assistant/message") {
      const joined = event.data.message.content
        .filter((block) => block.type === "text")
        .map((block) => block.text)
        .join("");
      if (joined !== "") text = joined;
    }
    if (event.type === "turn/end") reason = event.data.reason;
  }
  return { text, reason };
}

/** One line describing an error without leaking anything else. */
function describe(error) {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Decide whether a persisted session may be continued in this worktree.
 *
 * `inspect` is the non-committing read: it validates the stored header and log
 * without adopting or repairing the session, which is what a pre-resume check
 * should do. A session that exists but was created in another directory is a
 * mismatch, never a continuation - the same safety property the official ACP
 * implementation enforces.
 *
 * @returns `{ ok: true }` or `{ ok: false, failureClass, detail }`.
 */
async function validateResume(persistence, requestedId, cwd) {
  if (persistence === undefined) {
    return { ok: false, failureClass: RESUME_FAILURE.PERSISTENCE_UNAVAILABLE, detail: "no sessionPersistence service" };
  }
  let inspection;
  try {
    inspection = await persistence.inspect(SessionId(requestedId));
  } catch (error) {
    const detail = describe(error);
    const missing = /not found|no such|unknown session|missing/i.test(detail);
    return {
      ok: false,
      failureClass: missing ? RESUME_FAILURE.NOT_FOUND : RESUME_FAILURE.INSPECT_FAILED,
      detail
    };
  }
  const meta = inspection?.meta;
  if (meta === undefined || meta === null || meta.id !== requestedId) {
    return { ok: false, failureClass: RESUME_FAILURE.NOT_FOUND, detail: "no validated header for that id" };
  }
  if (meta.cwd !== cwd) {
    return {
      ok: false,
      failureClass: RESUME_FAILURE.CWD_MISMATCH,
      detail: `session cwd ${JSON.stringify(meta.cwd)} is not this worktree ${JSON.stringify(cwd)}`
    };
  }
  return { ok: true };
}

/** Build the agent options and the model-selection setup shared by create/resume. */
function agentOptionsFor(selection) {
  return {
    agentOptions: { provider: selection.provider, model: selection.model },
    setup: (agentCtx) => {
      installModelSelection(agentCtx, { current: selection, assembled: void 0 });
    }
  };
}

/**
 * Run one turn, then write the result and request process exit.
 *
 * @param ctx - plugin context carrying the Agent, model, Session and persistence services.
 * @param config - validated runner config.
 * @param io - process-facing effects.
 */
async function run(ctx, config, io) {
  await ctx.get("loader")?.await();
  const agents = ctx.get("agents");
  const defaultModel = ctx.get("agentDefaultModel");
  const sessions = ctx.get("sessions");
  const persistence = ctx.get("sessionPersistence");
  const startedAt = Date.now();
  const cwd = process.cwd();

  if (agents === void 0 || defaultModel === void 0 || sessions === void 0) {
    emit(io, config, {
      success: false,
      session_action: config.mode === "new" ? "new" : "fallback-new",
      resume_attempted: config.mode === "resume",
      resume_succeeded: false,
      resume_failure_class: "core-service-missing",
      text: "",
      turn_outcome: "not-run",
      duration_ms: Date.now() - startedAt
    });
    return;
  }

  const selection = defaultModel.currentSelection();

  let sessionAction = config.mode === "new" ? "new" : "fallback-new";
  let resumeAttempted = false;
  let resumeSucceeded = false;
  let resumeFailureClass = null;
  let resumeDetail = null;
  let agent;

  if (config.mode === "resume") {
    resumeAttempted = true;
    const verdict = await validateResume(persistence, config.sessionId, cwd);
    if (verdict.ok) {
      try {
        const handle = await agents.resume({
          resumeSessionId: SessionId(config.sessionId),
          ...agentOptionsFor(selection)
        });
        agent = handle.agent;
        resumeSucceeded = true;
        sessionAction = "resume";
      } catch (error) {
        resumeFailureClass = RESUME_FAILURE.RESUME_REJECTED;
        resumeDetail = describe(error);
      }
    } else {
      resumeFailureClass = verdict.failureClass;
      resumeDetail = verdict.detail;
    }
  }

  if (agent === void 0) {
    const handle = await agents.create({
      sessionId: SessionId(`session-${randomUUID()}`),
      meta: { cwd },
      ...agentOptionsFor(selection)
    });
    agent = handle.agent;
  }

  await agent.whenIdle();
  const firstSeq = agent.session.seq;
  agent.followup(
    createUserMessage({
      content: [{ type: "text", text: config.task }],
      source: { kind: "user" }
    })
  );
  await agent.whenIdle();
  await sessions.flush(agent.session);

  const outcome = summarize(agent.session.events, firstSeq);
  const completed = outcome.reason?.kind === "completed";
  if (outcome.reason?.kind === "error") {
    io.stderr.write(`dsh: ${outcome.reason.error.code}: ${outcome.reason.error.message}\n`);
  }

  emit(io, config, {
    success: completed,
    session_id: String(agent.session.id),
    session_action: sessionAction,
    resume_attempted: resumeAttempted,
    resume_succeeded: resumeSucceeded,
    resume_failure_class: resumeFailureClass,
    resume_detail: resumeDetail,
    provider: selection.provider,
    model: selection.model,
    reasoning_effort: selection.reasoningEffort ?? null,
    text: outcome.text,
    turn_outcome: outcome.reason?.kind ?? "unknown",
    duration_ms: Date.now() - startedAt
  });
}

/**
 * Write the turn result in the requested format and request process exit.
 * @param io - process-facing effects.
 * @param config - validated runner config.
 * @param payload - the machine-readable turn record.
 */
function emit(io, config, payload) {
  if (config.outputFormat === "json") {
    io.stdout.write(`${JSON.stringify(payload)}\n`);
  } else {
    io.stdout.write(`${payload.text ?? ""}\n`);
  }
  io.exit(payload.success === true ? 0 : 1);
}

/** Report an unexpected direct-driver failure and request a failing exit. */
function fail(io, error) {
  io.stderr.write(`dsh: ${describe(error)}\n`);
  io.exit(1);
}

/**
 * Mount the session driver.
 * @param ctx - plugin context carrying core services and the launcher-provided exit request.
 * @param config - validated runner config.
 */
function apply(ctx, config) {
  const exit = ctx.get("appExit");
  if (exit === undefined) {
    throw new Error("cb16-session-runner: the launcher must provide ctx.appExit before the tree mounts");
  }
  const io = { stdout: internals.stdout, stderr: internals.stderr, exit };
  run(ctx, config, io).catch((error) => {
    fail(io, error);
  });
}

export { Config, RESUME_FAILURE, apply, inject, internals, name, summarize, validateResume };
