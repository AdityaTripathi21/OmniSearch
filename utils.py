import hashlib
from pathlib import Path
import mimetypes
from datetime import datetime, timezone

import config


# read raw binary contents of file and generate a hash for ids
def file_hash(path: str | Path) -> str:
    path = Path(path)
    
    h = hashlib.sha256()
    
    # read raw binary contents of file
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
        
    return h.hexdigest()
    
# generate hash for text    
def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()    

# basically just return type/extensions -> ex: Path("photo.jpeg") -> "image/jpeg"
def mime_type(path: str | Path) -> str:
    path = Path(path)

    detected_type, _ = mimetypes.guess_type(str(path))

    # if type is unknown, application/octet-stream is used to convey unknown binary data
    return detected_type or "application/octet-stream"
    
# return timestamp, useful for metadata    
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()    

# check if file type is supported    
def is_supported(path: str | Path) -> bool:
    path = Path(path)

    # hidden resource files on MacOS should not be embeddings
    if path.name.startswith("._"):
        return False

    return path.suffix.lower() in config.ALL_EXTENSIONS    

# return category like image/document/audio    
def get_media_category(path: str | Path) -> str | None:
    path = Path(path)
    suffix = path.suffix.lower()

    # supported extensions is map like -> category -> set of extensions
    for category, extensions in config.SUPPORTED_EXTENSIONS.items():
        if suffix in extensions:
            return category

    return None