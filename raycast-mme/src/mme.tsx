import { Action, ActionPanel, Grid, Icon, LaunchProps } from "@raycast/api";
import { useMemo } from "react";

import { useMmeCommand } from "./lib/mme";
import { SearchResponse, SearchResult } from "./lib/types";

function uniqueFiles(results: SearchResult[]): SearchResult[] {
  const paths = new Set<string>();

  return results.filter((result) => {
    if (!result.file_path || paths.has(result.file_path)) {
      return false;
    }

    paths.add(result.file_path);
    return true;
  });
}

function percentage(similarity: number): string {
  return Math.round(similarity * 100) + "%";
}

export default function Command(props: LaunchProps<{ arguments: Arguments.Mme }>) {
  const query = props.arguments.query.trim();
  const { data, error, isLoading, revalidate } = useMmeCommand<SearchResponse>(["search", query, "--limit", "30"], {
    timeout: 30_000,
  });
  const results = useMemo(() => uniqueFiles(data?.results ?? []), [data]);

  return (
    <Grid
      columns={5}
      aspectRatio="1"
      fit={Grid.Fit.Contain}
      inset={Grid.Inset.Small}
      isLoading={isLoading}
      navigationTitle={"Search: " + query}
      searchBarPlaceholder="Filter these results"
    >
      {error ? (
        <Grid.EmptyView
          icon={Icon.Warning}
          title="Search Failed"
          description={error.message}
          actions={
            <ActionPanel>
              <Action title="Try Again" icon={Icon.ArrowClockwise} onAction={revalidate} />
            </ActionPanel>
          }
        />
      ) : results.length === 0 && !isLoading ? (
        <Grid.EmptyView
          icon={Icon.MagnifyingGlass}
          title="No Matching Files"
          description={"MME found no indexed files for “" + query + "”."}
        />
      ) : (
        results.map((result) => (
          <Grid.Item
            key={result.file_path}
            id={result.file_path}
            content={{
              value: { fileIcon: result.file_path },
              tooltip: result.preview || result.file_path,
            }}
            title={result.file_name}
            subtitle={[result.media_category, percentage(result.similarity)].filter(Boolean).join(" · ")}
            keywords={[result.file_path, result.media_category]}
            quickLook={{
              path: result.file_path,
              name: result.file_name,
            }}
            actions={
              <ActionPanel>
                <Action.Open title="Open File" target={result.file_path} />
                <Action.ShowInFinder path={result.file_path} />
                <Action.ToggleQuickLook />
                <Action.CopyToClipboard title="Copy File Path" content={result.file_path} />
                <Action title="Run Search Again" icon={Icon.ArrowClockwise} onAction={revalidate} />
              </ActionPanel>
            }
          />
        ))
      )}
    </Grid>
  );
}
