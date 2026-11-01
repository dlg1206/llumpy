"""
File: _chat.py

Description: Preserve chat history with model

@author Derek Garcia
"""
import copy
from abc import ABC
from typing import Any

from retry import AsyncRetryHandler
from . import AsyncModelClient
from ._conversation_builder import _UserStep, _UserOrBuildStep, ConversationBuilder
from ._model_client import ModelClient
from ._models import History
from ..retry import RetryHandler, DEFAULT_MAX_RETRIES

type Context = _UserStep | _UserOrBuildStep


class _Chat(ABC):
    """Chat methods to track history"""

    def __init__(self, context: Context | None):
        """
        Create a new chat

        :param context: Optional context to start chat with (Default: None)
        """
        self._context = copy.deepcopy(context) if context else None
        self._chat = copy.deepcopy(self._context) if self._context else ConversationBuilder()

    def clear_chat(self) -> None:
        """
        Clear the chat history to the starting context
        """
        self._chat = copy.deepcopy(self._context) if self._context else ConversationBuilder()

    @property
    def history(self) -> History:
        """
        :return: Conversation history
        """
        return self._chat.history


class Chat(_Chat):
    """Synchronous chat"""

    def __init__(self, client: ModelClient, context: Context | None = None):
        """
        Create a new synchronous chat

        :param client: LLM to chat with
        :param context: Optional context to start chat with (Default: None)
        """
        super().__init__(context)
        self._client = client

    def message(self,
                content: str | None = None,
                *,
                file: str | None = None,
                handler: RetryHandler | None = None,
                retries: int = DEFAULT_MAX_RETRIES,
                **prompt_kwargs: Any) -> str:
        """
        Send a message to the chat

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :param handler: Optional handler to ensure the response is valid (Default: None)
        :param retries: Number of retries allowed (Default: 5)
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text or parsed object from retry handler
        """
        # send a user message with previous messages
        self._chat = self._chat.user(content, file=file)
        reply = self._client.prompt_many(self._chat.build(), handler=handler, retries=retries, **prompt_kwargs)
        self._chat = self._chat.assistant(reply)
        return reply


class AsyncChat(_Chat):
    """Asynchronous chat"""

    def __init__(self, client: AsyncModelClient, context: Context | None = None):
        """
        Create a new asynchronous chat

        :param client: LLM to chat with
        :param context: Optional context to start chat with (Default: None)
        """
        super().__init__(context)
        self._client = client

    async def message(self,
                      content: str | None = None,
                      *,
                      file: str | None = None,
                      handler: AsyncRetryHandler | None = None,
                      retries: int = DEFAULT_MAX_RETRIES,
                      **prompt_kwargs: Any) -> str:
        """
        Send a message to the chat

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :param handler: Optional handler to ensure the response is valid (Default: None)
        :param retries: Number of retries allowed (Default: 5)
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text or parsed object from retry handler
        """
        # send a user message with previous messages
        self._chat = self._chat.user(content, file=file)
        reply = await self._client.prompt_many(self._chat.build(), handler=handler, retries=retries, **prompt_kwargs)
        self._chat = self._chat.assistant(reply)
        return reply
