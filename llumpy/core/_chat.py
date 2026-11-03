"""
File: _chat.py

Description: Model client wrapper to preserve chat history

@author Derek Garcia
"""
import copy
from abc import ABC, abstractmethod
from typing import Any, Iterator, AsyncIterator

from ._async_model_client import AsyncModelClient
from ._conversation_builder import _UserStep, _UserOrBuildStep, ConversationBuilder, _AssistantOrBuildStep
from ._model_client import ModelClient
from ._models import History
from ..retry import RetryHandler, AsyncRetryHandler, DEFAULT_MAX_RETRIES

type Context = _UserStep | _UserOrBuildStep


class ExceedContextWindowError(RuntimeError):
    """Raised when a request needs more tokens than the model's context window allows."""

    def __init__(self, requested_tokens: int, context_window: int) -> None:
        """
        Create new ExceedContextWindowError

        :param requested_tokens: Tokens requested
        :param context_window: Max context window
        """
        self.requested_tokens = requested_tokens
        self.context_window = context_window
        super().__init__(
            f"Exceeded context window: requested {requested_tokens} tokens, "
            f"but only {context_window} are allowed"
        )


class Tokenizer(ABC):
    """Base class for token counters"""

    @abstractmethod
    def count_tokens(self, text: str | History) -> int:
        """Estimate the amount of tokens of text"""


class ContextWindow:
    """Context window to restrict the chat"""

    def __init__(self, tokenizer: Tokenizer, max_tokens: int, reserved_output_tokens: int = 0) -> None:
        """
        Create new Context Window

        :param tokenizer: Tokenizer to estimate calculate the token
        :param max_tokens: Max tokens allowed for this context window
        :param reserved_output_tokens: Optional buffer output tokens (Default: 0)
        """
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._reserved_output_tokens = reserved_output_tokens

    def validate(self, text: str | History) -> None:
        """
        Validate the text or conversation does not exceed the context window

        :param text: Text to estimate tokens for
        :raises ExceedContextWindowError: If the request tokens exceed the max context window
        """
        requested = self._tokenizer.count_tokens(text) + self._reserved_output_tokens
        if requested > self._max_tokens:
            raise ExceedContextWindowError(requested, self._max_tokens)

    def get_remaining_tokens(self, text: str | History) -> int:
        """
        Get the remaining tokens in the current context window

        :param text: Text to estimate tokens for
        :return: Remaining tokens
        """
        return self._max_tokens - (self._tokenizer.count_tokens(text) + self._reserved_output_tokens)


class _Chat(ABC):
    """Chat methods to track history"""

    def __init__(self, context: Context | None, context_window: ContextWindow | None):
        """
        Create a new chat

        :param context: Optional context to start chat with (Default: None)
        """
        self._context = copy.deepcopy(context) if context else None
        self._chat = self._fresh_chat()
        self._context_window = context_window

    def _fresh_chat(self) -> _UserStep | _UserOrBuildStep | ConversationBuilder:
        """
        Create a fresh chat with the original context

        :return: Start of the context
        """
        return copy.deepcopy(self._context) if self._context else ConversationBuilder()

    def _prepare(self, content: str | None, file: str | None) -> _AssistantOrBuildStep:
        """
        Build the next conversation state and validate it, without committing it.

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :return: The candidate conversation step
        """
        candidate = self._chat.user(content, file=file)
        if self._context_window:
            self._context_window.validate(candidate.history)
        return candidate

    def clear_chat(self) -> None:
        """
        Clear the chat history to the starting context
        """
        self._chat = self._fresh_chat()

    @property
    def history(self) -> History:
        """
        :return: Conversation history
        """
        return self._chat.history

    @property
    def remaining_tokens(self) -> int | None:
        """
        :return: The tokens remaining in the current context window, None if not using context window
        """
        if not self._context_window:
            return None
        return self._context_window.get_remaining_tokens(self._chat.history)


class Chat(_Chat):
    """Synchronous chat"""

    def __init__(self,
                 client: ModelClient,
                 *,
                 context: Context | None = None,
                 context_window: ContextWindow | None = None):
        """
        Create a new synchronous chat

        :param client: LLM to chat with
        :param context: Optional context to start chat with (Default: None)
        """
        self._client = client
        super().__init__(context, context_window)

    def _stream(self, candidate: _AssistantOrBuildStep, **prompt_kwargs: Any) -> Iterator[Any]:
        """
        Internal stream method to track responses

        :param candidate: Candidate step to send to client
        :param prompt_kwargs: kwargs for chat
        :return: Stream
        """
        chunks: list[str] = []
        for chunk in self._client.prompt_stream(candidate.build(), **prompt_kwargs):
            if text := self._client.extract_text(chunk):
                chunks.append(text)
            yield chunk
        # only reached if the stream was fully consumed without error
        self._chat = candidate.assistant("".join(chunks))

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
        candidate = self._prepare(content, file)
        reply = self._client.prompt_many(
            candidate.build(), handler=handler, retries=retries, **prompt_kwargs
        )
        self._chat = candidate.assistant(reply)  # commit only on success
        return reply

    def stream(self,
               content: str | None = None,
               *,
               file: str | None = None,
               **prompt_kwargs: Any) -> Iterator[Any]:
        """
        Prompt a model and stream the response

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :param prompt_kwargs: kwargs for chat
        :return: Chat stream
        """
        candidate = self._prepare(content, file)
        return self._stream(candidate, **prompt_kwargs)


class AsyncChat(_Chat):
    """Asynchronous chat"""

    def __init__(self,
                 client: AsyncModelClient,
                 *,
                 context: Context | None = None,
                 context_window: ContextWindow | None = None):
        """
        Create a new asynchronous chat

        :param client: LLM to chat with
        :param context: Optional context to start chat with (Default: None)
        """
        self._client = client
        super().__init__(context, context_window)

    async def _stream(self, candidate: _AssistantOrBuildStep, **prompt_kwargs: Any) -> AsyncIterator[Any]:
        """
        Internal stream method to track responses

        :param candidate: Candidate step to send to client
        :param prompt_kwargs: kwargs for chat
        :return: Async stream
        """
        chunks: list[str] = []
        async for chunk in await self._client.prompt_stream(candidate.build(), **prompt_kwargs):
            if text := self._client.extract_text(chunk):
                chunks.append(text)
            yield chunk
        # only reached if the stream was fully consumed without error
        self._chat = candidate.assistant("".join(chunks))

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
        candidate = self._prepare(content, file)
        reply = await self._client.prompt_many(
            candidate.build(), handler=handler, retries=retries, **prompt_kwargs
        )
        self._chat = candidate.assistant(reply)  # commit only on success
        return reply

    def stream(self,
               content: str | None = None,
               *,
               file: str | None = None,
               **prompt_kwargs: Any) -> AsyncIterator[Any]:
        """
        Prompt a model and stream the response

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :param prompt_kwargs: kwargs for chat
        :return: Async chat stream
        """
        candidate = self._prepare(content, file)
        return self._stream(candidate, **prompt_kwargs)
