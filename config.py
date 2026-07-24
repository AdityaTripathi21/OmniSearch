from pathlib import Path

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "laptop_search"

INDEX_ROOTS = [
    "~/Documents",
    "~/Desktop",
    "~/Downloads",
]

EXCLUDED_DIR_NAMES = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".cache",
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ".DS_Store",
}

EXCLUDED_FILE_PATTERNS = {
    "*.log",
    "*.tmp",
    "*.pyc",
     "*.yaml",
    "*.yml",
    "*.sh",
}

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

SUPPORTED_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg"},
    "audio": {".mp3", ".wav"},
    "video": {".mp4", ".mov"},
    "document": {".pdf"},
    "text": {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".py", ".js", ".ts", ".go", ".rs", ".sh"},
}

ALL_EXTENSIONS = set()

for extensions in SUPPORTED_EXTENSIONS.values():
    ALL_EXTENSIONS.update(extensions)

