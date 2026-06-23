import math
import os
from pathlib import Path
import mimetypes
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

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

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b)

def save_index(index: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(index, f)

def load_index(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def index_images(folder: str):
    folder_path = Path(folder)
    
    image_paths = [
        *folder_path.glob("*.jpg"),
        *folder_path.glob("*.jpeg"),
        *folder_path.glob("*.png"),
    ]
    
    index = []

    for path in image_paths:
        embedding = embed_image(str(path))

        record = {
            "path": str(path),
            "type": "image",
            "embedding": embedding,
        }
        
        index.append(record)
        
    return index

def search_index(index, query, top_k):
    query_vec = embed_query(query)
    
    res = []
    
    for record in index:
        score = cosine(query_vec, record["embedding"])
        
        res.append({
            "path": record["path"],
            "score": score,
        })
        
    res.sort(key=lambda item: item["score"], reverse=True)
    
    
    return res[:top_k]


    

    