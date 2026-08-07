import { Action, ActionPanel, Detail, Icon } from "@raycast/api";

import { useMmeCommand } from "./lib/mme";
import { StageError, SyncResponse } from "./lib/types";

function errorSection(errors: StageError[]): string[] {
  if (errors.length === 0) {
    return [];
  }

  const shown = errors.slice(0, 10).map((error) => "- **" + error.path + "** — " + error.error);
  const remainder = errors.length - 10;

  return ["", "## Errors", "", ...shown, ...(remainder > 0 ? ["- …and " + remainder + " more"] : [])];
}

function summary(response: SyncResponse): string {
  const { scan, hash, index } = response.result;
  const errors = [...scan.errors, ...hash.errors, ...index.errors];
  const lines = [
    "# MME Sync Complete",
    "",
    "## Scan",
    "",
    "- Discovered: **" + scan.discovered + "**",
    "- New: **" + scan.new + "**",
    "- Changed: **" + scan.changed + "**",
    "- Unchanged: **" + scan.unchanged + "**",
    "",
    "## Hash",
    "",
    "- Selected: **" + hash.selected + "**",
    "- Hashed: **" + hash.hashed + "**",
    "- Skipped: **" + hash.skipped + "**",
    "",
    "## Index",
    "",
    "- Attempted: **" + index.selected + "**",
    "- Indexed: **" + index.indexed + "**",
    "- Chunks stored: **" + index.chunks + "**",
    "- Remaining: **" + index.remaining + "**",
  ];

  if (index.stopped_reason) {
    lines.push(
      "",
      "> Indexing stopped: **" + index.stopped_reason + "**" + (index.stopped_path ? " at " + index.stopped_path : ""),
    );
  }

  return [...lines, ...errorSection(errors)].join("\n");
}

export default function Command() {
  const { data, error, isLoading, revalidate } = useMmeCommand<SyncResponse>(
    ["sync", "--quiet", "--batch-size", "10"],
    { timeout: 0 },
  );

  const markdown = error
    ? "# Sync Failed\n\n" + error.message
    : data
      ? summary(data)
      : "# Synchronizing MME\n\nScanning, hashing, and indexing your configured files…";

  return (
    <Detail
      isLoading={isLoading}
      markdown={markdown}
      actions={
        <ActionPanel>
          <Action title="Run Sync Again" icon={Icon.ArrowClockwise} onAction={revalidate} />
        </ActionPanel>
      }
    />
  );
}
