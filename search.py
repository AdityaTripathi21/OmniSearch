import math
import os
from pathlib import Path
import mimetypes
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from db import collection

load_dotenv() 

MODEL = "gemini-embedding-2"


client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
)

def embed_image(path: str) -> list[float]:
    image_path = Path(path)
    mime_type, _ = mimetypes.guess_type(image_path)
    
    if mime_type is None:
        raise ValueError(f"Could not detect MIME type for {path}")
    
    # read entire file as raw bytes
    image_bytes = image_path.read_bytes() # type: ignore
    
    # create embedding
    res = client.models.embed_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(  # 
                data=image_bytes,
                mime_type=mime_type,
            )
        ],
    )
    
    return res.embeddings[0].values # type: ignore


def embed_query(query: str) -> list[float]:
    res = client.models.embed_content(
        model=MODEL,
        contents=f"task: search result | query: {query}",   # contents -> actual input
    )
    
    return res.embeddings[0].values # type: ignore


# get images from a folder, embed them, and update/insert into chromaDB
def index_images(folder: str):
    folder_path = Path(folder)
    
    image_paths = [
        *folder_path.glob("*.jpg"),
        *folder_path.glob("*.jpeg"),
        *folder_path.glob("*.png"),
    ]
    
    for path in image_paths:
        embedding = embed_image(str(path))

        # id and metadata share values
        collection.upsert(
            ids=[f"image:{path.resolve()}"],
            embeddings=[embedding],
            metadatas=[
                {
                    "path": str(path.resolve()),
                    "type": "image",
                }
            ],
            documents=[path.name],
        )
        
    
    

def search_index(query, top_k):
    query_vec = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )

    return results


    

    