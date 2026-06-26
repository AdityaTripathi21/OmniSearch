import chromadb

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(
    name = "laptop_search",
    embedding_function=None
    )


