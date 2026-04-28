import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv() 

client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
)

with open("tiger.jpeg", "rb") as f:
    image_bytes = f.read()
    
res = client.models.embed_content(
    model="gemini-embedding-2",
    contents=types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg"
    )
)

print(res.embeddings[0].values[:5]) # type: ignore