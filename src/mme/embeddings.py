import os
from pathlib import Path

import src.mme.config as config
import src.mme.utils as utils

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv() 

MODEL = config.EMBEDDING_MODEL

client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
)


def embed_query(query: str) -> list[float]:
    res = client.models.embed_content(
        model=MODEL,
        contents=f"task: search result | query: {query}",   # contents -> actual input
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )
    
    return res.embeddings[0].values # type: ignore

def embed_image(path: str | Path) -> list[float]:
    image_path = Path(path)
    mime_type = utils.mime_type(image_path)
    
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
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )
    
    return res.embeddings[0].values # type: ignore

def embed_pdf(path: str | Path) -> list[float]:
    pdf_path = Path(path)
    pdf_bytes = pdf_path.read_bytes()

    res = client.models.embed_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            )
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )

    return res.embeddings[0].values  # type: ignore

def embed_audio(path: str | Path) -> list[float]:
    audio_path = Path(path)
    mime_type = utils.mime_type(audio_path)
    audio_bytes = audio_path.read_bytes()

    res = client.models.embed_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            )
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )

    return res.embeddings[0].values  # type: ignore


def embed_video(path: str | Path) -> list[float]:
    video_path = Path(path)
    mime_type = utils.mime_type(video_path)
    video_bytes = video_path.read_bytes()

    res = client.models.embed_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=video_bytes,
                mime_type=mime_type,
            )
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )

    return res.embeddings[0].values  # type: ignore

def embed_text(text: str) -> list[float]:
    res = client.models.embed_content(
        model=MODEL,
        contents=f"title: text document | text: {text}",
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )

    return res.embeddings[0].values  # type: ignore

def embed_text_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one request while preserving input order."""

    if not texts:
        return []

    contents = [
        types.Content(
            parts=[
                types.Part.from_text(
                    text=f"title: text document | text: {text}"
                )
            ]
        )
        for text in texts
    ]

    res = client.models.embed_content(
        model=MODEL,
        contents=contents,
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBEDDING_DIMENSIONS,
        ),
    )

    embeddings = [
        embedding.values
        for embedding in res.embeddings # type: ignore
    ]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Embedding response count does not match input count"
        )

    return embeddings  # type: ignore