"""
File: openai.py

Description: Client for interacting with OpenAI models

@author Derek Garcia
"""

from openai import AuthenticationError

from llumpy.exeception import InvalidAPIKeyException
from llumpy.model_client import ModelClient, AsyncModelClient, _load_api_key

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient(ModelClient):
    """
    Interface for using OpenAI models
    """

    def __init__(self, model_name: str):
        """
        Create new client for use with OpenAI LLMs

        :param model_name: Name of OpenAI model to use
        :raises EnvironmentError: If the 'OPENAI_API_KEY' env var is not defined
        """
        super().__init__(model_name, _load_api_key(OPENAI_API_KEY_ENV), OPENAI_BASE_URL)

    def verify_api_key(self) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If OpenAI key does not have access to requested model
        """
        try:
            self._model_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyException('OpenAI') from e


class AsyncOpenAIClient(AsyncModelClient):
    """
    Async interface for using OpenAI models
    """

    def __init__(self, model_name: str):
        """
        Create new async client for use with OpenAI LLMs

        :param model_name: Name of OpenAI model to use
        :raises EnvironmentError: If the 'OPENAI_API_KEY' env var is not defined
        """
        super().__init__(model_name, _load_api_key(OPENAI_API_KEY_ENV), OPENAI_BASE_URL)

    async def verify_api_key(self) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If OpenAI key does not have access to requested model
        """
        try:
            await self._model_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyException('OpenAI') from e
