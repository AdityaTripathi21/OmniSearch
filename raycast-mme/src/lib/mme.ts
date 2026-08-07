import { getPreferenceValues } from "@raycast/api";
import { useExec } from "@raycast/utils";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

type MmePreferences = {
  mmeExecutable: string;
  mmeWorkingDirectory: string;
};

type CommandFailure = Error & {
  stderr?: string;
};

type ExecOutput = {
  stdout: string | Buffer;
  stderr: string | Buffer;
  error?: Error;
  exitCode: number | null;
};

const execFileAsync = promisify(execFile);
const LONG_RUNNING_TIMEOUT_MS = 24 * 60 * 60 * 1000;

function preferences(): MmePreferences {
  return getPreferenceValues<MmePreferences>();
}

function parseJson<T>(output: string): T {
  const trimmed = output.trim();

  try {
    return JSON.parse(trimmed) as T;
  } catch {
    // Some libraries may write informational lines before the final JSON.
  }

  const lines = trimmed.split("\n").reverse();

  for (const line of lines) {
    try {
      return JSON.parse(line) as T;
    } catch {
      continue;
    }
  }

  throw new Error(
    trimmed ? "MME returned unexpected output: " + trimmed.slice(0, 300) : "MME did not return a JSON response",
  );
}

function errorText(output: string): string | undefined {
  const trimmed = output.trim();

  if (!trimmed) {
    return undefined;
  }

  try {
    const response = JSON.parse(trimmed) as { error?: string };
    return response.error || trimmed;
  } catch {
    return trimmed;
  }
}

function parseCommandOutput<T>({ stdout, stderr, error, exitCode }: ExecOutput): T {
  const stderrMessage = errorText(String(stderr));

  if (error || (exitCode !== null && exitCode !== 0)) {
    throw new Error(stderrMessage || error?.message || "MME exited with code " + exitCode);
  }

  return parseJson<T>(String(stdout));
}

function failureMessage(error: unknown): string {
  if (error instanceof Error) {
    const stderr = (error as CommandFailure).stderr?.trim();

    if (stderr) {
      try {
        const response = JSON.parse(stderr) as { error?: string };
        return response.error || stderr;
      } catch {
        return stderr;
      }
    }

    return error.message;
  }

  return String(error);
}

export function useMmeCommand<T>(args: string[], options: { timeout?: number } = {}) {
  const config = preferences();
  const timeout = options.timeout === 0 ? LONG_RUNNING_TIMEOUT_MS : (options.timeout ?? 30_000);

  return useExec<T>(config.mmeExecutable, args, {
    cwd: config.mmeWorkingDirectory,
    timeout,
    parseOutput: (output) => parseCommandOutput<T>(output),
  });
}

export async function runMmeCommand<T>(args: string[], options: { timeout?: number } = {}): Promise<T> {
  const config = preferences();

  try {
    const { stdout } = await execFileAsync(config.mmeExecutable, args, {
      cwd: config.mmeWorkingDirectory,
      timeout: options.timeout ?? 0,
      encoding: "utf8",
    });

    return parseJson<T>(stdout);
  } catch (error) {
    throw new Error(failureMessage(error));
  }
}
