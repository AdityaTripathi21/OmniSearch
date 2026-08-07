import { Action, ActionPanel, Alert, Detail, Icon, Toast, confirmAlert, showToast } from "@raycast/api";

import { runMmeCommand, useMmeCommand } from "./lib/mme";
import { PruneResponse, PruneSummary } from "./lib/types";

function summary(result: PruneSummary): string {
  const lines = [
    result.applied ? "# Excluded Files Removed" : "# Exclusion Cleanup Preview",
    "",
    "- Catalog files checked: **" + result.scanned + "**",
    "- Excluded files matched: **" + result.matched + "**",
  ];

  if (result.applied) {
    lines.push(
      "- Catalog rows deleted: **" + result.catalog_rows_deleted + "**",
      "- Chroma records deleted: **" + result.chroma_records_deleted + "**",
    );
  } else {
    lines.push("", "> This is a dry run. Nothing has been deleted.");
  }

  if (result.errors.length > 0) {
    lines.push(
      "",
      "## Errors",
      "",
      ...result.errors.slice(0, 10).map((error) => "- **" + error.path + "** — " + error.error),
    );
  }

  return lines.join("\n");
}

export default function Command() {
  const { data, error, isLoading, revalidate, mutate } = useMmeCommand<PruneResponse>(["prune-excluded", "--quiet"], {
    timeout: 0,
  });

  async function applyPrune() {
    const confirmed = await confirmAlert({
      title: "Remove Excluded Files?",
      message:
        "This removes matching records from MME's SQLite catalog and Chroma index. It does not delete the original files.",
      primaryAction: {
        title: "Remove from MME",
        style: Alert.ActionStyle.Destructive,
      },
    });

    if (!confirmed) {
      return;
    }

    const toast = await showToast({
      style: Toast.Style.Animated,
      title: "Removing excluded files",
    });

    try {
      const applied = await runMmeCommand<PruneResponse>(["prune-excluded", "--apply", "--quiet"], { timeout: 0 });
      await mutate(Promise.resolve(applied), {
        shouldRevalidateAfter: false,
      });
      toast.style = Toast.Style.Success;
      toast.title = "Excluded files removed";
      toast.message = applied.result.catalog_rows_deleted + " catalog files removed";
    } catch (applyError) {
      toast.style = Toast.Style.Failure;
      toast.title = "Cleanup failed";
      toast.message = applyError instanceof Error ? applyError.message : String(applyError);
    }
  }

  const markdown = error
    ? "# Cleanup Preview Failed\n\n" + error.message
    : data
      ? summary(data.result)
      : "# Checking Current Exclusions\n\nNo records will be deleted during this preview.";

  return (
    <Detail
      isLoading={isLoading}
      markdown={markdown}
      actions={
        <ActionPanel>
          {data && !data.result.applied && data.result.matched > 0 ? (
            <Action
              title="Remove Excluded Files from MME"
              icon={Icon.Trash}
              style={Action.Style.Destructive}
              onAction={applyPrune}
            />
          ) : null}
          <Action title="Refresh Preview" icon={Icon.ArrowClockwise} onAction={revalidate} />
        </ActionPanel>
      }
    />
  );
}
