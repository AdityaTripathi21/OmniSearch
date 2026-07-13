import os
from fnmatch import fnmatchcase
from pathlib import Path

import store
import embeddings
import utils
import config


# embed a file and store it in chromaDB, return a dict of status
def ingest_file(path: str | Path, source: str = "manual", description: str = "") -> dict:
    path = Path(path).resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not utils.is_supported(path):
        raise ValueError(f"Unsupported file type: {path.suffix}")

    category = utils.get_media_category(path)

    doc_id = utils.file_hash(path)
    
    if store.exists(doc_id):
        return {
            "status": "skipped",
            "reason": "already embedded",
            "id": doc_id,
            "path": str(path),
        }
    if category == "text":
        text = path.read_text(errors="replace") # replace characters if unable to identify

        if len(text) > 32000:
            text = text[:32000]

        embedding = embeddings.embed_text(text)
        document = text[:500]
    elif category == "image":
        embedding = embeddings.embed_image(path)
        document = description or f"Image: {path.name}"
    elif category == "document":
        embedding = embeddings.embed_pdf(path)
        document = description or f"PDF: {path.name}"
    elif category == "audio":
        embedding = embeddings.embed_audio(path)
        document = description or f"Audio: {path.name}"
    elif category == "video":
        embedding = embeddings.embed_video(path)
        document = description or f"Video: {path.name}"
    else:
        raise ValueError(f"Unsupported media category: {category}")
    
    metadata = {
        "file_path": str(path),
        "file_name": path.name,
        "file_type": utils.mime_type(path),
        "media_category": category,
        "timestamp": utils.now_iso(),
        "source": source,
        "description": description,
        "file_size": path.stat().st_size,
    }
    
    store.add(doc_id, embedding, metadata, document=document)
    
    return {
        "status": "embedded",
        "id": doc_id,
        "path": str(path),
        "category": category,
    }

# ingest every file in a directory recursively
# if recursive = False, only ingest files inside folder and not nested folders    
def ingest_directory(path: str | Path, source: str = "manual", 
                     recursive: bool = True) -> list[dict]:
    path = Path(path).resolve()

    files = []

    # dirnames is immediate child directories
    for directory, dirnames, filenames in os.walk(path, topdown=True):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in config.EXCLUDED_DIR_NAMES
        )

        for filename in sorted(filenames):
            if filename in config.EXCLUDED_FILE_NAMES:
                continue

            if any(
                fnmatchcase(filename, pattern)
                for pattern in config.EXCLUDED_FILE_PATTERNS
            ):
                continue

            file_path = Path(directory) / filename
            if utils.is_supported(file_path):
                files.append(file_path)

        if not recursive:
            dirnames.clear()

    files.sort()

    results = []

    for file_path in files:
        try:
            result = ingest_file(file_path, source=source)
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "status": "error",
                    "path": str(file_path),
                    "error": str(e),
                }
            )

    return results
    
    
        
        



    

