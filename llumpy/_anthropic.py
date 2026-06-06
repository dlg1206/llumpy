"""
File: _anthropic.py

Description: Client for interacting with Anthropic models

@author Derek Garcia
"""
from typing import List, Any

from anthropic import AuthenticationError, NotFoundError, PermissionDeniedError, Anthropic, Stream, AsyncAnthropic
from anthropic.types import Message, RawContentBlockDeltaEvent, RawContentBlockStartEvent, RawContentBlockStopEvent, \
    RawMessageDeltaEvent, RawMessageStartEvent, RawMessageStopEvent

from llumpy._exception import InvalidAPIKeyError, ModelNotFoundError
from llumpy._message import Conversation, ConversationBuilder
from llumpy._model_client import ModelClient, _load_api_key, AsyncModelClient

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_MAX_TOKENS = 8096


class AnthropicClient(ModelClient):
    """Client interface for interacting with Anthropic API"""

    def __init__(self, model: str):
        """
        Initialize connection to Anthropic

        :param model: LLM to use
        :raises EnvironmentError: If the 'ANTHROPIC_API_KEY' env var is not defined
        """
        super().__init__(model)
        self._model_client = Anthropic(api_key=_load_api_key(ANTHROPIC_API_KEY_ENV))

    def prompt_message(self, conversation: Conversation, **prompt_kwargs: Any) \
            -> Message | Stream[
                RawMessageStartEvent | RawMessageDeltaEvent | RawMessageStopEvent | RawContentBlockStartEvent | RawContentBlockDeltaEvent | RawContentBlockStopEvent]:
        """
        Prompt a model for Anthropic chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion or stream
        """
        messages: List[Any] = conversation.to_dicts()
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        non_system = [m for m in messages if m['role'] != 'system']

        return self._model_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=non_system,
            **({"system": system} if system else {}),
            **prompt_kwargs
        )

    def prompt_one(self, message: str, **prompt_kwargs: Any) -> str:
        """
        Prompt a model for simple text return

        :param message: Message to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        msg = self.prompt_message(ConversationBuilder().user(message).build(), stream=False, **prompt_kwargs)
        return msg.content[0].text

    def prompt_many(self, conversation: Conversation, **prompt_kwargs: Any) -> str | None:
        """
        Prompt a model for simple text return

        :param conversation: Conversation with prompt to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        msg = self.prompt_message(conversation, stream=False, **prompt_kwargs)
        return msg.content[0].text

    def validate(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If the Anthropic key is invalid
        :raises ModelNotFoundExecution: If requested model does not exist
        :raises PermissionDeniedError: If key does not have access to requested model
        """
        try:
            self.prompt_one('hi', max_tokens=1)
        except AuthenticationError as e:
            raise InvalidAPIKeyError('Anthropic') from e
        except NotFoundError as e:
            raise ModelNotFoundError('Anthropic', self.model) from e
        except PermissionDeniedError as e:
            raise e


class AsyncAnthropicClient(AsyncModelClient):
    """Client interface for interacting with Anthropic API"""

    def __init__(self, model: str):
        """
        Initialize connection to Anthropic

        :param model: LLM to use
        :raises EnvironmentError: If the 'ANTHROPIC_API_KEY' env var is not defined
        """
        super().__init__(model)
        self._model_client = AsyncAnthropic(api_key=_load_api_key(ANTHROPIC_API_KEY_ENV))

    async def prompt_message(self, conversation: Conversation, **prompt_kwargs: Any) \
            -> Message | Stream[
                RawMessageStartEvent | RawMessageDeltaEvent | RawMessageStopEvent | RawContentBlockStartEvent | RawContentBlockDeltaEvent | RawContentBlockStopEvent]:
        """
        Prompt a model for Anthropic chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion or stream
        """
        messages: List[Any] = conversation.to_dicts()
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        non_system = [m for m in messages if m['role'] != 'system']

        return await self._model_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=non_system,
            **({"system": system} if system else {}),
            **prompt_kwargs
        )

    async def prompt_one(self, message: str, **prompt_kwargs: Any) -> str:
        """
        Prompt a model for simple text return

        :param message: Message to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        msg = await self.prompt_message(ConversationBuilder().user(message).build(), stream=False, **prompt_kwargs)
        return msg.content[0].text

    async def prompt_many(self, conversation: Conversation, **prompt_kwargs: Any) -> str | None:
        """
        Prompt a model for simple text return

        :param conversation: Conversation with prompt to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        msg = await self.prompt_message(conversation, stream=False, **prompt_kwargs)
        return msg.content[0].text

    async def validate(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If the Anthropic key is invalid
        :raises ModelNotFoundExecution: If requested model does not exist
        :raises PermissionDeniedError: If key does not have access to requested model
        """
        try:
            await self.prompt_one('hi', max_tokens=1)
        except AuthenticationError as e:
            raise InvalidAPIKeyError('Anthropic') from e
        except NotFoundError as e:
            raise ModelNotFoundError('Anthropic', self.model) from e
        except PermissionDeniedError as e:
            raise e
