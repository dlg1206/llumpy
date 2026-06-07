"""
File: _anthropic.py

Description: Client for interacting with Anthropic models

@author Derek Garcia
"""
from typing import List, Any

from anthropic import AuthenticationError, NotFoundError, PermissionDeniedError, Anthropic, Stream, AsyncAnthropic
from anthropic.types import Message, RawContentBlockDeltaEvent, RawContentBlockStartEvent, RawContentBlockStopEvent, \
    RawMessageDeltaEvent, RawMessageStartEvent, RawMessageStopEvent, TextBlock

from llumpy._exception import InvalidAPIKeyError, ModelNotFoundError
from llumpy._message import Conversation
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

    def prompt(self, conversation: Conversation, **prompt_kwargs: Any) \
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

    def extract_text(self, response: Message) -> str | None:
        """
        Extract LLM response text from Anthropic object

        :param response: Anthropic chat response
        :return: Text message if present, else None
        """
        text_block = next((block for block in response.content if isinstance(block, TextBlock)), None)
        return text_block.text if text_block else None

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

    async def prompt(self, conversation: Conversation, **prompt_kwargs: Any) \
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

    def extract_text(self, response: Message) -> str | None:
        """
        Extract LLM response text from Anthropic object

        :param response: Anthropic chat response
        :return: Text message if present, else None
        """
        text_block = next((block for block in response.content if isinstance(block, TextBlock)), None)
        return text_block.text if text_block else None

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
