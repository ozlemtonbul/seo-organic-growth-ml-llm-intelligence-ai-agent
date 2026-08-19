from anthropic import Anthropic

from config.settings import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
)

from src.llm.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.
    """

    def __init__(self):
        self.client = Anthropic(
            api_key=ANTHROPIC_API_KEY
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> str:

        response = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        if not response.content:
            return ""

        return response.content[0].text.strip()