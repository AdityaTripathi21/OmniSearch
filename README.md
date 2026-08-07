# MME

MME is a local semantic file-search application. It discovers files from
selected folders, tracks their state in SQLite, creates Gemini embeddings,
stores vectors in ChromaDB, and lets you search from either a JSON CLI or a
native Raycast extension.

The project is designed for personal files such as notes, source code,
documents, images, audio, and video. Exclusion rules prevent dependency
folders, generated output, credentials, and other unwanted files from being
indexed.

## Features

- Discover individual files or recursively scan directories.
- Exclude exact filenames, filename patterns, and directory names.
- Track discovered files, modification state, hashes, and indexed versions in
  SQLite.
- Avoid rehashing and reindexing unchanged files.
- Split text into overlapping chunks before embedding.
- Batch text embeddings and store searchable records in ChromaDB.
- Search semantically with optional media-type filtering.
- Stop cleanly when Gemini rate limits indexing.
- Report scan, hash, index, and maintenance progress.
- Preview or remove newly excluded records from SQLite and Chroma.
- Search, synchronize, and maintain the index from Raycast.

## How It Works

~~~text
Files and folders
      |
      v
Discovery and exclusions
      |
      v
SQLite catalog
      |
      v
Content hashing
      |
      v
Chunking and Gemini embeddings
      |
      v
ChromaDB
      |
      +----------> CLI search
      |
      +----------> Raycast grid
~~~

SQLite stores filesystem state and determines which files need hashing or
indexing. ChromaDB stores embeddings and searchable chunk metadata. The file
contents used for embeddings are sent to the configured Gemini API.

## Requirements

- Python 3.11 or newer
- A Gemini API key
- macOS and Raycast for the Raycast extension
- Node.js 22.14 or newer and npm for Raycast development

## Python Setup

Create and activate a virtual environment:

~~~bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
~~~

Install the current runtime dependencies and MME in editable mode:

~~~bash
python -m pip install chromadb google-genai python-dotenv
python -m pip install -e .
~~~

The runtime dependencies are not yet declared in pyproject.toml, so they must
currently be installed explicitly.

Create a .env file in the repository root:

~~~dotenv
GEMINI_API_KEY=your-api-key
~~~

The .env file, SQLite catalog, Chroma data, virtual environments, and build
artifacts are ignored by Git.

## Configuration

Primary settings live in [src/mme/config.py](src/mme/config.py).

The default indexing roots are:

~~~python
INDEX_ROOTS = [
    "~/Documents",
    "~/Desktop",
    "~/Downloads",
]
~~~

The same file defines:

- Supported file extensions and media categories
- Exact excluded directory names
- Exact excluded filenames
- Excluded filename glob patterns
- Gemini model and embedding dimensions
- SQLite and Chroma storage locations

Review these settings before the first broad sync. Files excluded after they
were already indexed can be removed with the maintenance command described
below.

## CLI

Installing the package creates the mme command.

### Search

~~~bash
mme search "computer science notes"
mme search "database design" --limit 10
mme search "vacation photo" --media-type image
~~~

Search writes one JSON response to stdout.

### Synchronize

Use configured roots:

~~~bash
mme sync --batch-size 10
~~~

Synchronize selected paths:

~~~bash
mme sync ~/Documents ~/Downloads --batch-size 10
~~~

Disable recursive discovery:

~~~bash
mme sync ~/Documents --no-recursive
~~~

Suppress human-readable progress and keep only JSON output:

~~~bash
mme sync ~/Documents --quiet
~~~

Progress is written to stderr, while the final machine-readable response is
written to stdout.

Currently, explicit paths restrict the scan stage only. The following hash and
index stages process every pending record in the SQLite catalog, including
backlog created by earlier scans.

### Remove Newly Excluded Records

Preview the cleanup without changing anything:

~~~bash
mme prune-excluded
mme prune-excluded ~/Documents
~~~

Apply the cleanup after reviewing the summary:

~~~bash
mme prune-excluded --apply
~~~

This removes matching SQLite and Chroma records. It never deletes the original
files.

## Raycast

The Raycast extension lives in [raycast-mme/](raycast-mme).

Install its dependencies and start development mode:

~~~bash
cd raycast-mme
npm install
npm run dev
~~~

Raycast exposes three commands:

- **Search MME** — enter a semantic query and browse unique files in a visual
  grid with Finder visuals, Quick Look, Open File, and Show in Finder actions.
- **Sync MME** — run the pipeline and view its final summary.
- **Prune Excluded Files** — preview exclusion cleanup and require confirmation
  before applying it.

On first launch, configure:

- **MME Executable** — the absolute path to venv/bin/mme
- **MME Working Directory** — the absolute path to this repository

For example:

~~~text
/path/to/mme/venv/bin/mme
/path/to/mme
~~~

See the [Raycast README](raycast-mme/README.md) for extension-specific details.

## Tests

Run the Python test suite:

~~~bash
venv/bin/python3 -m unittest discover -s tests -p "test_*.py" -v
~~~

Validate the Raycast extension:

~~~bash
cd raycast-mme
npx tsc --noEmit
npm run lint
npm run build
~~~

Raycast's build command writes its development bundle outside the repository
into Raycast's local extension directory.

## Project Structure

~~~text
mme/
├── src/mme/
│   ├── discovery.py       # Filesystem discovery
│   ├── exclusions.py      # Shared exclusion rules
│   ├── scanner.py         # Catalog synchronization
│   ├── catalog.py         # SQLite state
│   ├── hasher.py          # Content hashing
│   ├── chunking.py        # Text chunking
│   ├── embeddings.py      # Gemini embeddings
│   ├── indexer.py         # Chroma indexing
│   ├── store.py           # Chroma operations
│   ├── search.py          # Semantic search
│   ├── maintenance.py     # Exclusion cleanup
│   ├── pipeline.py        # Scan/hash/index orchestration
│   └── cli.py             # JSON CLI
├── raycast-mme/           # Raycast extension
├── tests/                 # Python tests and fixtures
├── data/                  # Local SQLite and Chroma data (ignored)
└── pyproject.toml
~~~

## Current Limitations

- Encrypted or password-protected PDFs cannot be embedded directly.
- PDF, audio, video, and image files are currently indexed as single records;
  text files support multiple chunks.
- Search operates on Chroma chunks. The Raycast interface deduplicates results
  by file path, but the Python search function currently returns raw chunk
  matches.
- Sync stops on a Gemini rate limit and leaves the remaining files pending for
  a later run.
- The Raycast grid uses Finder-provided file visuals and Quick Look rather than
  a custom thumbnail-generation cache.
- The project is currently optimized for local personal use rather than
  one-command installation by other users.

## Roadmap

- Read-only MCP tools for semantic search, safe indexed-file reading, and index
  status
- Backend file-level search deduplication
- Improved PDF extraction and failure tracking
- Optional generated thumbnail caching
- Complete dependency metadata and distributable installation
