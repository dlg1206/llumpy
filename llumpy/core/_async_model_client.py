"""
File: _async_model_client.py

Description: Generic models for asynchronous clients

@author Derek Garcia
"""
from abc import ABC, abstractmethod
from typing import Any

from ._conversation_builder import ConversationBuilder
from ._models import Conversation
from ..retry import DEFAULT_MAX_RETRIES, AsyncRetryHandler


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
                         handler: AsyncRetryHandler | None = None,
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
                          handler: AsyncRetryHandler | None = None,
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
