from google import genai
from google.genai import types

from config.settings import (
    GEMINI_API_KEY,
    LLM_MODEL,
)

from src.llm.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider using the official Google Gen AI SDK.
    """

    def __init__(self):
        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        if not response.text:
            return ""

        return response.text.strip()