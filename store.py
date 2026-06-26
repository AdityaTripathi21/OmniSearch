import chromadb

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(
    name = "laptop_search",
    embedding_function=None
    )


# add or update record 
def add (doc_id, embedding, metadata, document = ""):
     collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[document],
    )


# search for top k results, where specifices metadata type like image or pdf     
def search(query_embedding, n_results=5, where=None):
    if count() == 0:
        return {
            "ids": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "distances": [[]],
        }

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, count()),
        "include": ["metadatas", "documents", "distances"],
    }

    if where is not None:
        kwargs["where"] = where

    return collection.query(**kwargs)

# only checks for id, and not if a file changed, so a good id like a file hash is important  
def exists(doc_id):
    result = collection.get(ids=[doc_id])
    return len(result["ids"]) > 0

# return count of records
def count():
    return collection.count()
