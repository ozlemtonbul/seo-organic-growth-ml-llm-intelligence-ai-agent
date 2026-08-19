from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Common interface for all supported LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Generate a text response from the selected LLM provider.
        """
        raise NotImplementedError