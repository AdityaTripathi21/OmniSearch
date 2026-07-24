from ingest import ingest_paths
from search import search
from ingest import discover_files
import config

# print("Starting ingest...", flush=True)
# results = ingest_paths(["test_folder"], source="files", recursive=True)
# print("Finished ingest.", flush=True)


# errors = [item for item in results if item["status"] == "error"]
# embedded = [item for item in results if item["status"] == "embedded"]
# skipped = [item for item in results if item["status"] == "skipped"]

# print(f"Embedded: {len(embedded)}")
# print(f"Skipped: {len(skipped)}")
# print(f"Errors: {len(errors)}")

# for error in errors:
#     print()
#     print(error["path"])
#     print(error["error"])

# matches = search("Computer Science", n_results=5)

# for result in matches:
#     print(result["similarity"], result["file_name"], result["file_path"])

files = discover_files(config.INDEX_ROOTS) # type: ignore

print(f"Eligible files: {len(files)}")

total_bytes = sum(path.stat().st_size for path in files)
print(f"Total size: {total_bytes:,} bytes")

for path in files[:100]:
    print(path)

largest_files = sorted(
    files,
    key=lambda path: path.stat().st_size,
    reverse=True,
)

for path in largest_files[:20]:
    print(f"{path.stat().st_size:,} bytes — {path}")