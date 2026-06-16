"""
File: _model_client.py

Description: Generic models for clients

@author Derek Garcia
"""
import os
from abc import ABC, abstractmethod
from typing import Any

from ._message import ConversationBuilder, Conversation
from ..utils import RetryHandler, AsyncRetryHandler, DEFAULT_MAX_RETRIES


class ModelClient(ABC):
    """Placeholder class for synchronous clients"""

    def __init__(self, model: str):
        """
        Create new client

        :param model: Model to use
        """
        self._model = model

    @abstractmethod
    def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response object"""

    @abstractmethod
    def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response stream object"""

    @abstractmethod
    def extract_text(self, response: Any) -> str | None:
        """Extract text from vendor-specific response object"""

    def prompt_one(self,
                   message: str,
                   *,
                   handler: RetryHandler = None,
                   retries: int = DEFAULT_MAX_RETRIES,
                   **prompt_kwargs: Any) -> Any:
        """
        Prompt a model for simple text return

        :param message: Message to send to LLM
        :param handler: Optional handler to ensure the response is valid (Default: None)
        :param retries: Number of retries allowed (Default: 5)
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text or parsed object from retry handler
        """
        conversation = ConversationBuilder().user(message).build()
        return self.prompt_many(conversation, handler=handler, retries=retries, **prompt_kwargs)

    def prompt_many(self,
                    conversation: Conversation,
                    *,
                    handler: RetryHandler = None,
                    retries: int = DEFAULT_MAX_RETRIES,
                    **prompt_kwargs: Any) -> Any:
        """
        Prompt a model for simple text return

        :param conversation: Conversation with prompt to send to LLM
        :param handler: Optional handler to ensure the response is valid (Default: None)
        :param retries: Number of retries allowed (Default: 5)
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text or parsed object from retry handler
        """
        # wrap with handler if provided
        if handler:
            return handler.try_prompt(lambda: self.vendor_prompt(conversation, **prompt_kwargs), self.extract_text,
                                      retries)
        # else just prompt
        return self.extract_text(self.vendor_prompt(conversation, **prompt_kwargs))

    @abstractmethod
    def validate(self) -> None:
        """Validate the client is ready to use"""

    @property
    def model(self) -> str:
        """
        :return: Name of the model
        """
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
    async def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response object"""

    @abstractmethod
    async def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response stream object"""

    @abstractmethod
    def extract_text(self, response: Any) -> str | None:
        """Extract text from vendor-specific response object"""

    async def prompt_one(self,
                         message: str,
                         *,
                         handler: AsyncRetryHandler = None,
                         retries: int = DEFAULT_MAX_RETRIES,
                         **prompt_kwargs: Any) -> Any:
        """
        Prompt a model for simple text return

        :param message: Message to send to LLM
        :param handler: Optional handler to ensure the response is valid (Default: None)
        :param retries: Number of retries allowed (Default: 5)
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text or parsed object from retry handler
        """
        conversation = ConversationBuilder().user(message).build()
        return await self.prompt_many(conversation, handler=handler, retries=retries, **prompt_kwargs)

    async def prompt_many(self,
                          conversation: Conversation,
                          *,
                          handler: AsyncRetryHandler = None,
                          retries: int = DEFAULT_MAX_RETRIES,
                          **prompt_kwargs: Any) -> Any:
        """
        Prompt a model for simple text return

        :param conversation: Conversation with prompt to send to LLM
        :param handler: Optional handler to ensure the response is valid (Default: None)
        :param retries: Number of retries allowed (Default: 5)
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text or parsed object from retry handler
        """
        # wrap with handler if provided
        if handler:
            return await handler.try_prompt(lambda: self.vendor_prompt(conversation, **prompt_kwargs),
                                            self.extract_text,
                                            retries)
        # else just prompt
        return self.extract_text(await self.vendor_prompt(conversation, **prompt_kwargs))

    @abstractmethod
    async def validate(self) -> None:
        """Validate the client is ready to use"""

    @property
    def model(self) -> str:
        """
        :return: Name of the model
        """
        return self._model


def load_api_key(env_var: str):
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
