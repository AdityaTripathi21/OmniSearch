# MME for Raycast

Search and manage the local MME semantic file index from Raycast.

## Commands

- **Search MME** runs semantic search and displays unique files in a visual
  grid. Open a result, reveal it in Finder, or preview it with Quick Look.
- **Sync MME** scans, hashes, and indexes configured roots and displays the
  final pipeline summary.
- **Prune Excluded Files** previews files affected by current exclusion rules
  and requires confirmation before removing their SQLite and Chroma records.

Configure the MME executable and project directory in Raycast's extension
preferences. The original files are never deleted by the prune command.
