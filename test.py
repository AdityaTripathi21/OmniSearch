from ingest import ingest_directory
from search import search

print("Starting ingest...", flush=True)
results = ingest_directory("test_folder", source="files", recursive=True)
print("Finished ingest.", flush=True)


errors = [item for item in results if item["status"] == "error"]
embedded = [item for item in results if item["status"] == "embedded"]
skipped = [item for item in results if item["status"] == "skipped"]

print(f"Embedded: {len(embedded)}")
print(f"Skipped: {len(skipped)}")
print(f"Errors: {len(errors)}")

for error in errors:
    print()
    print(error["path"])
    print(error["error"])

matches = search("Computer Science", n_results=5)

for result in matches:
    print(result["similarity"], result["file_name"], result["file_path"])