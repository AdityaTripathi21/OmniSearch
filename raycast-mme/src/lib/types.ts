export type SearchResult = {
  id: string;
  similarity: number;
  file_path: string;
  file_name: string;
  media_category: string;
  timestamp: string;
  description: string;
  source: string;
  preview: string;
};

export type SearchResponse = {
  ok: true;
  command: "search";
  results: SearchResult[];
};

export type StageError = {
  status?: string;
  path: string;
  error: string;
};

export type ScanSummary = {
  discovered: number;
  new: number;
  changed: number;
  unchanged: number;
  errors: StageError[];
};

export type HashSummary = {
  selected: number;
  hashed: number;
  skipped: number;
  errors: StageError[];
};

export type IndexSummary = {
  selected: number;
  indexed: number;
  chunks: number;
  remaining: number;
  stopped_reason: string | null;
  stopped_path: string | null;
  errors: StageError[];
};

export type SyncResponse = {
  ok: true;
  command: "sync";
  result: {
    scan: ScanSummary;
    hash: HashSummary;
    index: IndexSummary;
  };
};

export type PruneSummary = {
  scanned: number;
  matched: number;
  applied: boolean;
  catalog_rows_deleted: number;
  chroma_records_deleted: number;
  errors: StageError[];
};

export type PruneResponse = {
  ok: true;
  command: "prune-excluded";
  result: PruneSummary;
};
