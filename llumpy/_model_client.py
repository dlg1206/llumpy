"""
File: _model_client.py

Description: Generic models for clients

@author Derek Garcia
"""
import os
from abc import ABC, abstractmethod
from typing import Any, List


class ModelClient(ABC):
    """Placeholder class for synchronous clients"""

    def __init__(self, model: str):
        """
        Create new client

        :param model: Model to use
        """
        self._model = model

    @abstractmethod
    def prompt(self, messages: List[Any], **prompt_kwargs: Any) -> str:
        """Simple prompt to get LLM text response"""
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate the client is ready to use"""
        pass

    @property
    def model(self) -> str:
        return self._model


class AsyncModelClient(ABC):
    """Placeholder class for asynchronous clients"""

    def __init__(self, model: str):
        """
        Create new client

        :param model: Name of model to use
        """
        self._model = model

    @abstractmethod
    async def prompt(self, messages: List[Any], **prompt_kwargs: Any) -> str:
        """Simple prompt to get LLM text response"""
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate the client is ready to use"""
        pass

    @property
    def model(self) -> str:
        return self._model


def _load_api_key(env_var: str):
    """
    Load API key from env variable

    :param env_var: Environment variable to attempt load key from
    :raises EnvironmentError: If the env var is not defined
    :returns: API key
    """
    # ensure OpenAPI key is present
    api_key = os.getenv(env_var)
    if not api_key:
        raise EnvironmentError(f"Missing API key in environment variable: {env_var}")
    return api_key
