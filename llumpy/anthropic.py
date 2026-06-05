"""
File: anthropic.py

Description: Client for interacting with Anthropic models

@author Derek Garcia
"""

from openai import AuthenticationError

from llumpy.exeception import InvalidAPIKeyException
from llumpy.model_client import ModelClient, _load_api_key, AsyncModelClient

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"


class AnthropicClient(ModelClient):
    """
    Interface for using Anthropic models
    """

    def __init__(self, model_name: str):
        """
        Create new client for use with Anthropic LLMs

        :param model_name: Name of Anthropic model to use
        :raises EnvironmentError: If the 'ANTHROPIC_API_KEY' env var is not defined
        """
        super().__init__(model_name, _load_api_key(ANTHROPIC_API_KEY_ENV), ANTHROPIC_BASE_URL)

    def verify_api_key(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If Anthropic key does not have access to requested model
        """
        try:
            self._model_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyException('Anthropic') from e


class AsyncAnthropicClient(AsyncModelClient):
    """
    Async interface for using Anthropic models
    """

    def __init__(self, model_name: str):
        """
        Create new async client for use with Anthropic LLMs

        :param model_name: Name of Anthropic model to use
        :raises EnvironmentError: If the 'ANTHROPIC_API_KEY' env var is not defined
        """
        super().__init__(model_name, _load_api_key(ANTHROPIC_API_KEY_ENV), ANTHROPIC_BASE_URL)

    async def verify_api_key(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If Anthropic key does not have access to requested model
        """
        try:
            await self._model_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyException('Anthropic') from e
