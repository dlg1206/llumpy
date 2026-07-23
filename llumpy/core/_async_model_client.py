"""
File: _async_model_client.py

Description: Generic models for asynchronous clients

@author Derek Garcia
"""
from abc import ABC, abstractmethod
from typing import Any, List

from ._args import _validate_args
from ._conversation_builder import ConversationBuilder, _MessagesHolder
from ._models import Conversation, Message, Role
from ..retry import DEFAULT_MAX_RETRIES, AsyncRetryHandler


class _AsyncPromptMixIn:
    """Mixin that allows a step to prompt the model that started this chain"""
    _messages: List[Message]
    _client: "AsyncModelClient"

    async def prompt(self,
                     *,
                     handler: AsyncRetryHandler | None = None,
                     retries: int = DEFAULT_MAX_RETRIES,
                     **prompt_kwargs: Any) -> Any:
        """
        Prompt the model with the conversation built so far

        :param handler: Optional handler to ensure the response is valid (Default: None)
        :param retries: Number of retries allowed (Default: 5)
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text or parsed object from retry handler
        """
        return await self._client.prompt_many(Conversation(self._messages), handler=handler, retries=retries,
                                              **prompt_kwargs)


class _UserStep(_MessagesHolder):
    """First user turn - prevent prompting with just a system prompt"""

    def __init__(self, messages: List[Message], client: "AsyncModelClient"):
        """
        Create a new User step

        :param messages: List of messages of the current conversation
        :param client: Client to eventually prompt
        """
        super().__init__(messages)
        self._client = client

    def user(self, content: str | None = None, *, file: str | None = None) -> "_AssistantOrAsyncPromptStep":
        """
        Add a user message

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :return: Assistant or prompt step
        """
        return _AssistantOrAsyncPromptStep(self._add(Role.USER, content, file), self._client)


class _UserOrAsyncPromptStep(_UserStep, _AsyncPromptMixIn):
    """User turn in the conversation - allow prompting"""


class _AssistantOrAsyncPromptStep(_MessagesHolder, _AsyncPromptMixIn):
    """Assistant or prompt step in the conversation"""

    def __init__(self, messages: List[Message], client: "AsyncModelClient"):
        """
        Create a new Assistant step

        :param messages: List of messages of the current conversation
        :param client: Client to eventually prompt
        """
        super().__init__(messages)
        self._client = client

    def assistant(self, content: str | None = None, *, file: str | None = None) -> _UserOrAsyncPromptStep:
        """
        Add an assistant message

        :param content: Content of assistant message (Default: None)
        :param file: File to read content from (Default: None)
        :return: User step or prompt step
        """
        return _UserOrAsyncPromptStep(self._add(Role.ASSISTANT, content, file), self._client)


class _SingleUseConversationMixIn:
    """MixIn to prompt client with a single use conversations"""

    def system(self, content: str | None = None, *, file: str | None = None) -> _UserStep:
        """
        Init conversation with a system message

        :param content: Content of system message (Default: None)
        :param file: File to read content from (Default: None)
        :raises ValueError: If neither or both content and file are provided
        :return: User step
        """
        validated = _validate_args(content, file)
        return _UserStep([Message(Role.SYSTEM, validated)], self)

    def user(self, content: str | None = None, *, file: str | None = None) -> _AssistantOrAsyncPromptStep:
        """
        Init conversation with a user message

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :return: Assistant or prompt step
        """
        validated = _validate_args(content, file)
        return _AssistantOrAsyncPromptStep([Message(Role.USER, validated)], self)


class AsyncModelClient(_SingleUseConversationMixIn, ABC):
    """Abstract base class for asynchronous model clients"""

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
        Prompt a model with a single user message

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
        Prompt a model with a full conversation

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

    async def prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """
        Prompt a model and stream the response

        :param conversation: Messages to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Chat stream
        """
        # wrapper for vendor prompt
        return await self.vendor_prompt_stream(conversation, **prompt_kwargs)

    @abstractmethod
    async def validate(self) -> None:
        """Validate the client is ready to use"""

    @property
    def model(self) -> str:
        """
        :return: Name of the model
        """
        return self._model
