"""
File: _provider_factory.py

Description: Create validated provider clients

@author Derek Garcia
"""
from typing import TypeVar

from ._anthropic import AnthropicClient, AsyncAnthropicClient
from ._ollama import OllamaClient, AsyncOllamaClient
from ._openai import OpenAIClient, AsyncOpenAIClient
from ..core import ModelClient, AsyncModelClient

S = TypeVar("S", bound=ModelClient)
A = TypeVar("A", bound=AsyncModelClient)


class ProviderFactory:
    """Factory for creating validated model clients"""

    @staticmethod
    def anthropic(model: str) -> AnthropicClient:
        """
        Create new Anthropic client

        :param model: Name of Anthropic `model <https://platform.claude.com/docs/en/about-claude/models/overview>`_
        :return: :class:`AnthropicClient`
        """
        return _validate(AnthropicClient(model))

    @staticmethod
    async def async_anthropic(model: str) -> AsyncAnthropicClient:
        """
        Create new async Anthropic client

        :param model: Name of Anthropic `model <https://platform.claude.com/docs/en/about-claude/models/overview>`_
        :return: :class:`AsyncAnthropicClient`
        """
        return await _async_validate(AsyncAnthropicClient(model))

    @staticmethod
    def openai(model: str, *, api_key: str | None = None, base_url: str | None = None) -> OpenAIClient:
        """
        Create new OpenAI client

        :param model: Name of OpenAI `model <https://developers.openai.com/api/docs/models>`_
        :param api_key: API key to use (Default: OPENAI_API_KEY env var)
        :param base_url: URL of api server (Default: OpenAI)
        :return: :class:`OpenAIClient`
        """
        return _validate(OpenAIClient(model, api_key, base_url))

    @staticmethod
    async def async_openai(model: str, *, api_key: str | None = None, base_url: str | None = None) -> AsyncOpenAIClient:
        """
        Create new async OpemAI client

        :param model: Name of OpenAI `model <https://developers.openai.com/api/docs/models>`_
        :param api_key: API key to use (Default: OPENAI_API_KEY env var)
        :param base_url: URL of api server (Default: OpenAI)
        :return: :class:`AsyncOpenAIClient`
        """
        return await _async_validate(AsyncOpenAIClient(model, api_key, base_url))

    @staticmethod
    def ollama(model_name: str,
               model_tag: str | None = None,
               *,
               server_url: str | None = None,
               defer_download: bool = False) -> OllamaClient:
        """
        Create new Ollama client

        :param model_name: Name of Ollama `model <https://ollama.com/search>`
        :param model_tag: Optional model tag (Default: latest)
        :param server_url: Optional URL of the ollama server (Default: http://localhost:11434)
        :param defer_download: Defer model download if not already downloaded (Default: False)
        :return: :class:`OllamaClient`
        """
        client = _validate(OllamaClient(model_name, model_tag, server_url))
        # download if requested
        if not defer_download:
            client.download_model()
        return client

    @staticmethod
    async def async_ollama(model_name: str,
                           model_tag: str | None = None,
                           *,
                           server_url: str | None = None,
                           defer_download: bool = False) -> AsyncOllamaClient:
        """
        Create new Ollama client

        :param model_name: Name of Ollama `model <https://ollama.com/search>`
        :param model_tag: Optional model tag (Default: latest)
        :param server_url: Optional URL of the ollama server (Default: http://localhost:11434)
        :param defer_download: Defer model download if not already downloaded (Default: False)
        :return: :class:`AsyncOllamaClient`
        """
        client = await _async_validate(AsyncOllamaClient(model_name, model_tag, server_url))
        # download if requested
        if not defer_download:
            await client.download_model()
        return client


def _validate(client: S) -> S:
    """
    Execute a client's validation function

    :param client: Client to validate
    :return: Subclass of :class:`ModelClient`
    """
    client.validate()  # raises errors on failure
    return client


async def _async_validate(client: A) -> A:
    """
    Execute an async client's validation function

    :param client: Async client to validate
    :return: Subclass of :class:`AsyncModelClient`
    """
    await client.validate()  # raises errors on failure
    return client
