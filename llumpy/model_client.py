"""
File: model_client.py

Description: Generic model client for connecting and authenticating with an OpenAI compatible servers

@author Derek Garcia
"""
import os
from abc import ABC, abstractmethod
from typing import Any, List

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion


class _ModelClientBase(ABC):
    """Shared connection details for API connection"""

    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use
        :param base_url: Url of api server
        """
        self._model = model
        self._params = {}
        if api_key:
            self._params['api_key'] = api_key
        if base_url:
            self._params['base_url'] = base_url

    @property
    def model(self) -> str:
        return self._model


class ModelClient(_ModelClientBase, ABC):
    """Abstract model client for interacting with different LLM APIs"""

    def __init__(self, model: str, api_key: str, base_url: str):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use
        :param base_url: Url of api server
        """
        super().__init__(model, api_key, base_url)
        self._model_client = OpenAI(**self._params)

    @abstractmethod
    def verify_api_key(self) -> None:
        """Verify API key"""
        pass

    def prompt(self, messages: List[Any], **prompt_kwargs: Any) -> ChatCompletion:
        """
        Prompt a model

        :param messages: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat object
        """
        completion = self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            **prompt_kwargs
        )
        return completion


class AsyncModelClient(_ModelClientBase, ABC):
    """Abstract async model client for interacting with different LLM APIs"""

    def __init__(self, model: str, api_key: str, base_url: str):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use
        :param base_url: Url of api server
        """
        super().__init__(model, api_key, base_url)
        self._model_client = AsyncOpenAI(**self._params)

    @abstractmethod
    async def verify_api_key(self) -> None:
        """Verify API key"""
        pass

    async def prompt(self, messages: List[Any], **prompt_kwargs: Any) -> ChatCompletion:
        """
        Prompt a model

        :param messages: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat object and timer
        """
        completion = await self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            **prompt_kwargs
        )
        return completion


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
