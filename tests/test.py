from src.mme.discovery import discover_files
import src.mme.config as config

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