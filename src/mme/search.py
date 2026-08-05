import embeddings
import store

def search(query: str, n_results: int = 5, media_type: str | None = None) -> list[dict]:
    query_embedding = embeddings.embed_query(query)

    where = None
    if media_type:
        where = {"media_category": media_type}

    raw = store.search(query_embedding, n_results=n_results, where=where)
    
    results = []

    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]

        similarity = 1 - distance

        results.append({
            "id": raw["ids"][0][i],
            "similarity": round(similarity, 4),
            "file_path": meta.get("file_path", ""),
            "file_name": meta.get("file_name", ""),
            "media_category": meta.get("media_category", ""),
            "timestamp": meta.get("timestamp", ""),
            "description": meta.get("description", ""),
            "source": meta.get("source", ""),
            "preview": raw["documents"][0][i][:200] if raw["documents"][0][i] else "",
        })

    return results
