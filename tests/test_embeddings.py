import unittest
from unittest.mock import patch

from google.genai import errors as genai_errors

from mme import embeddings
from mme.errors import RateLimitError


class EmbeddingErrorTests(unittest.TestCase):
    def test_embed_content_translates_rate_limit_error(self) -> None:
        provider_error = genai_errors.ClientError(
            429,
            {"error": {"message": "quota reached", "status": "RESOURCE_EXHAUSTED"}},
        )

        with (
            patch.object(
                embeddings.client.models,
                "embed_content",
                side_effect=provider_error,
            ),
            self.assertRaises(RateLimitError) as raised,
        ):
            embeddings._embed_content("test text")

        self.assertIs(raised.exception.__cause__, provider_error)

    def test_embed_content_preserves_non_rate_limit_client_error(self) -> None:
        provider_error = genai_errors.ClientError(
            400,
            {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}},
        )

        with (
            patch.object(
                embeddings.client.models,
                "embed_content",
                side_effect=provider_error,
            ),
            self.assertRaises(genai_errors.ClientError) as raised,
        ):
            embeddings._embed_content("test text")

        self.assertIs(raised.exception, provider_error)


if __name__ == "__main__":
    unittest.main()
