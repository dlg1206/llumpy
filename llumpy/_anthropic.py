"""
File: _anthropic.py

Description: Client for interacting with Anthropic models

@author Derek Garcia
"""
from typing import List, Any

from anthropic import AuthenticationError, NotFoundError, PermissionDeniedError, Anthropic, Stream, AsyncAnthropic
from anthropic.types import Message, RawContentBlockDeltaEvent, RawContentBlockStartEvent, RawContentBlockStopEvent, \
    RawMessageDeltaEvent, RawMessageStartEvent, RawMessageStopEvent

from llumpy.exeception import InvalidAPIKeyError, ModelNotFoundError
from llumpy.model_client import ModelClient, _load_api_key, AsyncModelClient

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

    def prompt_message(self, messages: List[Any], **prompt_kwargs: Any) \
            -> Message | Stream[
                RawMessageStartEvent | RawMessageDeltaEvent | RawMessageStopEvent | RawContentBlockStartEvent | RawContentBlockDeltaEvent | RawContentBlockStopEvent]:
        """
        Prompt a model for Anthropic chat completion

        :param messages: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion or stream
        """
        return self._model_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=messages,
            **prompt_kwargs
        )

    def prompt(self, messages: List[Any], **prompt_kwargs: Any) -> str:
        """
        Prompt a model for simple text return

        :param messages: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        message = self.prompt_message(messages, stream=False, **prompt_kwargs)
        return message.content[0].text

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
    """Client interface for interacting with Anthropic API"""

    def __init__(self, model: str):
        """
        Initialize connection to Anthropic

        :param model: LLM to use
        :raises EnvironmentError: If the 'ANTHROPIC_API_KEY' env var is not defined
        """
        super().__init__(model)
        self._model_client = AsyncAnthropic(api_key=_load_api_key(ANTHROPIC_API_KEY_ENV))

    async def prompt_message(self, messages: List[Any], **prompt_kwargs: Any) \
            -> Message | Stream[
                RawMessageStartEvent | RawMessageDeltaEvent | RawMessageStopEvent | RawContentBlockStartEvent | RawContentBlockDeltaEvent | RawContentBlockStopEvent]:
        """
        Prompt a model for Anthropic chat completion

        :param messages: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion or stream
        """
        return self._model_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=messages,
            **prompt_kwargs
        )

    async def prompt(self, messages: List[Any], **prompt_kwargs: Any) -> str:
        """
        Prompt a model for simple text return

        :param messages: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        message = await self.prompt_message(messages, stream=False, **prompt_kwargs)
        return message.content[0].text

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
