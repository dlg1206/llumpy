"""
File: anthropic.py

Description: Client for interacting with Anthropic models

@author Derek Garcia
"""
from openai import AuthenticationError, NotFoundError, PermissionDeniedError

from llumpy.exeception import InvalidAPIKeyError, ModelNotFoundError
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

    def validate(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If the Anthropic key is invalid
        :raises ModelNotFoundExecution: If requested model does not exist
        :raises PermissionDeniedError: If key does not have access to requested model
        """
        try:
            self.prompt([{"role": "user", "content": "hi"}], max_tokens=1)
        except AuthenticationError as e:
            raise InvalidAPIKeyError('Anthropic') from e
        except NotFoundError as e:
            raise ModelNotFoundError('Anthropic', self.model) from e
        except PermissionDeniedError as e:
            raise e


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

    async def validate(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If the Anthropic key is invalid
        :raises ModelNotFoundExecution: If requested model does not exist
        :raises PermissionDeniedError: If key does not have access to requested model
        """
        try:
            await self.prompt([{"role": "user", "content": "hi"}], max_tokens=1)
        except AuthenticationError as e:
            raise InvalidAPIKeyError('Anthropic') from e
        except NotFoundError as e:
            raise ModelNotFoundError('Anthropic', self.model) from e
        except PermissionDeniedError as e:
            raise e
