from pathlib import Path

DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "laptop_search"

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

SUPPORTED_EXTENSIONS = {
    "image": {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tiff",
    },
    "audio": {
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".flac",
        ".aac",
    },
    "video": {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    },
    "document": {
        ".pdf",
    },
    "text": {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".html",
        ".py",
        ".js",
        ".ts",
        ".go",
        ".rs",
        ".sh",
    },
}

ALL_EXTENSIONS = set()

for extensions in SUPPORTED_EXTENSIONS.values():
    ALL_EXTENSIONS.update(extensions)
    
def get_media_category(extension: str) -> str | None:
    extension = extension.lower()

    for category, extensions in SUPPORTED_EXTENSIONS.items():
        if extension in extensions:
            return category

    return None