from openai import OpenAI

from config.settings import (
    LLM_MODEL,
    OPENAI_API_KEY,
)

from src.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT provider.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.responses.create(
            model=LLM_MODEL,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        if not response.output_text:
            return ""

        return response.output_text.strip()