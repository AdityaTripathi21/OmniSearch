import os
from pathlib import Path
from typing import Any


from . import config, utils
from .errors import RateLimitError


from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

load_dotenv() 

MODEL = config.EMBEDDING_MODEL

client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
)


def _embed_content(contents: Any) -> Any:
    """Request embeddings and translate Gemini rate-limit errors."""

    try:
        return client.models.embed_content(
            model=MODEL,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=config.EMBEDDING_DIMENSIONS,
            ),
        )

    except genai_errors.ClientError as error:
        if error.code == 429:
            raise RateLimitError(
                "Gemini embedding rate limit reached"
            ) from error

        raise


def embed_query(query: str) -> list[float]:
    res = _embed_content(
        f"task: search result | query: {query}"
    )
    
    return res.embeddings[0].values # type: ignore

def embed_image(path: str | Path) -> list[float]:
    image_path = Path(path)
    mime_type = utils.mime_type(image_path)
    
    # read entire file as raw bytes
    image_bytes = image_path.read_bytes() # type: ignore
    
    # create embedding
    res = _embed_content(
        [
            types.Part.from_bytes(  # 
                data=image_bytes,
                mime_type=mime_type,
            )
        ]
    )
    
    return res.embeddings[0].values # type: ignore

def embed_pdf(path: str | Path) -> list[float]:
    pdf_path = Path(path)
    pdf_bytes = pdf_path.read_bytes()

    res = _embed_content(
        [
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            )
        ]
    )

    return res.embeddings[0].values  # type: ignore

def embed_audio(path: str | Path) -> list[float]:
    audio_path = Path(path)
    mime_type = utils.mime_type(audio_path)
    audio_bytes = audio_path.read_bytes()

    res = _embed_content(
        [
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type,
            )
        ]
    )

    return res.embeddings[0].values  # type: ignore


def embed_video(path: str | Path) -> list[float]:
    video_path = Path(path)
    mime_type = utils.mime_type(video_path)
    video_bytes = video_path.read_bytes()

    res = _embed_content(
        [
            types.Part.from_bytes(
                data=video_bytes,
                mime_type=mime_type,
            )
        ]
    )

    return res.embeddings[0].values  # type: ignore

def embed_text(text: str) -> list[float]:
    res = _embed_content(
        f"title: text document | text: {text}"
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

    res = _embed_content(contents)

    embeddings = [
        embedding.values
        for embedding in res.embeddings # type: ignore
    ]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Embedding response count does not match input count"
        )

    return embeddings  # type: ignore
