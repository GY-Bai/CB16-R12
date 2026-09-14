import { Command } from "commander";
import { parseCmdline } from "@deepseek-ai/dsh-cmdline";

/**
 * Command-line provider for the CB16 Builder session runner.
 *
 * The shape deliberately mirrors `@deepseek-ai/dsh-headless/startup`: it parses
 * the command line and publishes an ordinary Cordis service that the lazy
 * runner config waits for. The difference is identity. CB16 routing must be
 * deterministic, so this profile accepts an *explicit* session id and never
 * offers "resume the latest session in this directory" - a branch must never be
 * able to continue whichever conversation happened to be written last.
 *
 * @module dsh-cb16-builder-session/startup
 */

/** Stable Cordis plugin name. */
const name = "cb16-session-startup";

/** Services required before the command line can be resolved. */
const inject = ["cmdlineArgs"];

/** Service provided by this plugin and injected by the runner. */
const CB16_SESSION_STARTUP_SERVICE = "cb16SessionStartup";

/** Output formats the runner understands. */
const OUTPUT_FORMATS = ["text", "json"];

/**
 * This app's command.
 * @returns a fresh program, so one process can parse more than once (tests).
 */
function cb16SessionCommand() {
  return new Command()
    .name("dsh --profile cb16-builder-session")
    .description("Run one CB16 Builder turn in an explicit DSH session and exit.")
    .helpOption("-h, --help", "show this help")
    .option("--new", "create a new session with a generated id")
    .option("--session <id>", "resume this exact persisted session id")
    .option(
      "--output-format <format>",
      `final message format: ${OUTPUT_FORMATS.join(" | ")}`,
      "text"
    )
    .argument("[task...]", "the task text; multiple words are joined by spaces")
    .addHelpText(
      "after",
      `
Examples:
  dsh --profile cb16-builder-session --new --output-format json "run the tests"
  dsh --profile cb16-builder-session --session session-<uuid> --output-format json "continue"
`
    );
}

/**
 * Parse and provide the session request as an ordinary Cordis service.
 *
 * A missing task, both identity flags, or neither identity flag is a usage
 * error, so on rejection nothing is provided and the tree never mounts.
 *
 * @param ctx - plugin context carrying the command line.
 */
function apply(ctx) {
  const program = cb16SessionCommand();
  program.action(() => {
    const options = program.opts();
    const task = program.args.join(" ");
    if (task.trim() === "") {
      program.error(
        'error: a task is required, for example: dsh --profile cb16-builder-session --new "run the tests"'
      );
    }
    if (options.new === true && typeof options.session === "string") {
      program.error("error: --new and --session are mutually exclusive");
    }
    if (options.new !== true && typeof options.session !== "string") {
      program.error(
        "error: pass --new or --session <id>; this profile never resumes the latest session"
      );
    }
    const outputFormat = String(options.outputFormat ?? "text");
    if (!OUTPUT_FORMATS.includes(outputFormat)) {
      program.error(`error: --output-format must be one of ${OUTPUT_FORMATS.join(", ")}`);
    }
    ctx.provide(CB16_SESSION_STARTUP_SERVICE, {
      mode: options.new === true ? "new" : "resume",
      sessionId: options.new === true ? null : String(options.session),
      task,
      outputFormat
    });
  });
  parseCmdline(ctx, program);
}

export { CB16_SESSION_STARTUP_SERVICE, OUTPUT_FORMATS, apply, cb16SessionCommand, inject, name };
