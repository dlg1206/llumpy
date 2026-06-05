"""
File: model_client.py

Description: Generic model client for connecting and authenticating with an OpenAI compatible servers

@author Derek Garcia
"""

from abc import ABC
from typing import Any, List

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion


class _ModelClientBase(ABC):
    """
    Shared connection details for API connection
    """

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


class ModelClient(_ModelClientBase):
    """
    Abstract model client for interacting with different LLM APIs
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use
        :param base_url: Url of api server
        """
        super().__init__(model, api_key, base_url)
        self._model_client = OpenAI(**self._params)

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


class AsyncModelClient(_ModelClientBase):
    """
    Abstract async model client for interacting with different LLM APIs
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use
        :param base_url: Url of api server
        """
        super().__init__(model, api_key, base_url)
        self._model_client = AsyncOpenAI(**self._params)

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
