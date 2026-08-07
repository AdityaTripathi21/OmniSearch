from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"

CHROMA_DIR = DATA_DIR / "chroma"
CATALOG_PATH = DATA_DIR / "catalog.sqlite3"

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
    "venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".vscode",
    ".idea",
    "coverage",
    "target",
    "vendor",
    ".aws-sam",
    ".terraform",
    ".serverless",
    ".terragrunt-cache",
    ".gradle",
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ".DS_Store",
    "package.json"
}

EXCLUDED_FILE_PATTERNS = {
    "*.log",
    "*.tmp",
    "*.pyc",
    "*.yaml",
    "*.yml",
    "*.sh",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*credentials*.csv",
    "*accessKeys*.csv",
    "package*.json",
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
