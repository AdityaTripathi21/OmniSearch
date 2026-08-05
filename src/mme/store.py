import chromadb
from pathlib import Path

from . import config

config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))


collection = client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},  # cosine distance -> lower = higher similarity
)

# add or update record 
def add(doc_id: str, embedding: list[float], metadata: dict, document: str = "") -> None:
     collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[document],
    )     
     
def add_many(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    documents: list[str],
) -> None:
    """Add or update multiple Chroma records."""

    lengths = {
        len(ids),
        len(embeddings),
        len(metadatas),
        len(documents),
    }

    if len(lengths) != 1:
        raise ValueError(
            "ids, embeddings, metadatas, and documents "
            "must have equal lengths"
        )

    if not ids:
        return

    collection.upsert(
        ids=ids,
        embeddings=embeddings, # type: ignore
        metadatas=metadatas, # type: ignore
        documents=documents,
    )

# search for top k results, where specifices metadata filters like image or pdf     
def search(query_embedding: list[float], n_results: int = 5, 
           where: dict | None = None) -> dict:   
    total = count() 
    if total == 0:
        return {
            "ids": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "distances": [[]],
        }

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, total),
        "include": ["metadatas", "documents", "distances"],
    }

    if where is not None:
        kwargs["where"] = where

    return collection.query(**kwargs) # type: ignore

# only checks for id, and not if a file changed, so a good id like a file hash is important  
def exists(doc_id: str) -> bool:
    result = collection.get(ids=[doc_id])
    return len(result["ids"]) > 0

# return count of records
def count() -> int:
    return collection.count()

def get_file_chunk_ids(path: str | Path) -> list[str]:
    """Return the IDs of all Chroma records belonging to a file."""
    path = Path(path).expanduser().resolve()

    result = collection.get(
        where={
            "file_path": str(path),
        },
        include=[],
    )

    return sorted(result["ids"])
    
def delete_ids(ids: list[str]) -> None:
    """Delete specific Chroma records."""
    if not ids:
        return

    collection.delete(ids=ids)
