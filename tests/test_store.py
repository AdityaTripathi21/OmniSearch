from uuid import uuid4

from mme import config, store


prefix = f"test-{uuid4()}"
ids = [
    f"{prefix}-chunk-0",
    f"{prefix}-chunk-1",
]

embeddings = [
    [0.1] * config.EMBEDDING_DIMENSIONS,
    [0.2] * config.EMBEDDING_DIMENSIONS,
]

metadatas = [
    {
        "file_path": "/temporary/test.txt",
        "content_hash": "test-hash",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 4,
    },
    {
        "file_path": "/temporary/test.txt",
        "content_hash": "test-hash",
        "chunk_index": 1,
        "start_char": 3,
        "end_char": 7,
    },
]

documents = [
    "ABCD",
    "DEFG",
]

try:
    store.add_many(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    result = store.collection.get(
        ids=ids,
        include=["documents", "metadatas"],
    )

    assert set(result["ids"]) == set(ids)
    assert len(result["documents"]) == 2 # type: ignore
    assert len(result["metadatas"]) == 2 # type: ignore

    records = {
        record_id: {
            "document": document,
            "metadata": metadata,
        }
        for record_id, document, metadata in zip(
            result["ids"],
            result["documents"], # type: ignore
            result["metadatas"], # type: ignore
        )
    }

    assert records[ids[0]]["document"] == "ABCD"
    assert records[ids[1]]["document"] == "DEFG"
    assert records[ids[0]]["metadata"]["chunk_index"] == 0
    assert records[ids[1]]["metadata"]["chunk_index"] == 1

    print("add_many test passed")

finally:
    # Prevent test records from remaining in the real collection.
    store.collection.delete(ids=ids)
