"""
File: _anthropic.py

Description: Client for interacting with Anthropic models

@author Derek Garcia
"""
from typing import List, Any, cast

from anthropic import AuthenticationError, NotFoundError, PermissionDeniedError, Anthropic, Stream, AsyncAnthropic, \
    AsyncStream
from anthropic.types import Message, TextBlock, RawMessageStreamEvent, RawContentBlockDeltaEvent, TextDelta, \
    MessageStreamEvent, MessageParam

from ..core import ModelClient, AsyncModelClient, Conversation, load_api_key, InvalidAPIKeyError, ModelNotFoundError, \
    Tokenizer, History

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_MAX_TOKENS = 8096


class AnthropicTokenizer(Tokenizer):
    """"Estimate the amount of tokens of text"""

    def __init__(self, model: str, token_client: Anthropic):
        """
        Create new Anthropic tokenizer

        :param model: Name of model
        :param token_client: Sync Anthropic client to request token estimates
        """
        self._model = model
        self._token_client = token_client

    def count_tokens(self, text: str | History) -> int:
        """
        Calculate the number of tokens a text or chat history

        :param text: Text or history
        :return: Number of tokens
        """
        messages = [{"role": "user", "content": text}] if isinstance(text, str) else text.to_dicts()
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        non_system = cast(list[MessageParam], cast(object, [m for m in messages if m['role'] != 'system']))
        result = self._token_client.messages.count_tokens(
            model=self._model,
            messages=non_system,
            **({"system": system} if system else {})
        )
        return result.input_tokens


class AnthropicClient(ModelClient):
    """Client interface for interacting with Anthropic API"""

    # placeholder for model client
    _vendor_client: Anthropic

    def __init__(self, model: str):
        """
        Internal initialize connection to Anthropic

        :param model: LLM to use
        :raises EnvironmentError: If the 'ANTHROPIC_API_KEY' env var is not defined
        """
        vendor_client = Anthropic(api_key=load_api_key(ANTHROPIC_API_KEY_ENV))
        self._tokenizer = AnthropicTokenizer(model, vendor_client)
        super().__init__(model, vendor_client)

    def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> Message:
        """
        Prompt a model for Anthropic chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion
        """
        messages: List[Any] = conversation.to_dicts()
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        non_system = [m for m in messages if m['role'] != 'system']
        prompt_kwargs.pop("stream", None)  # guard against true stream
        return self._vendor_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=non_system,
            stream=False,
            **({"system": system} if system else {}),
            **prompt_kwargs
        )

    def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Stream[RawMessageStreamEvent]:
        """
        Prompt a model for a streamed Anthropic chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion stream
        """
        messages: List[Any] = conversation.to_dicts()
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        non_system = [m for m in messages if m['role'] != 'system']
        prompt_kwargs.pop("stream", None)  # guard against false stream
        return self._vendor_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=non_system,
            stream=True,
            **({"system": system} if system else {}),
            **prompt_kwargs
        )

    def extract_text(self, response: Message | RawMessageStreamEvent) -> str | None:
        """
        Extract LLM response text from Anthropic object

        :param response: Anthropic chat response or stream chunk
        :return: Text message if present, else None
        """
        # if message passed is a chunk
        if isinstance(response, Message):
            texts = [b.text for b in response.content if isinstance(b, TextBlock)]
            return "".join(texts) if texts else None
        # if message passed is a completed message
        if isinstance(response, RawContentBlockDeltaEvent) and isinstance(response.delta, TextDelta):
            return response.delta.text

        return None

    def validate(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyError: If the Anthropic key is invalid
        :raises ModelNotFoundError: If requested model does not exist
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

    @property
    def tokenizer(self) -> AnthropicTokenizer:
        """
        :return: Anthropic tokenizer
        """
        return self._tokenizer


class AsyncAnthropicClient(AsyncModelClient):
    """Async client interface for interacting with Anthropic API"""

    # placeholder for model client
    _vendor_client: AsyncAnthropic

    def __init__(self, model: str):
        """
        Initialize connection to Anthropic

        :param model: LLM to use
        :raises EnvironmentError: If the 'ANTHROPIC_API_KEY' env var is not defined
        """
        # sync model purely for token estimate
        self._tokenizer = AnthropicTokenizer(model, Anthropic(api_key=load_api_key(ANTHROPIC_API_KEY_ENV)))
        super().__init__(model, AsyncAnthropic(api_key=load_api_key(ANTHROPIC_API_KEY_ENV)))

    async def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> Message:
        """
        Prompt a model for Anthropic chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion
        """
        messages: List[Any] = conversation.to_dicts()
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        non_system = [m for m in messages if m['role'] != 'system']
        prompt_kwargs.pop("stream", None)  # guard against true stream
        return await self._model_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=non_system,
            stream=False,
            **({"system": system} if system else {}),
            **prompt_kwargs
        )

    async def vendor_prompt_stream(self,
                                   conversation: Conversation,
                                   **prompt_kwargs: Any) -> AsyncStream[RawMessageStreamEvent]:
        """
        Prompt a model for a streamed Anthropic chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion stream
        """
        messages: List[Any] = conversation.to_dicts()
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        non_system = [m for m in messages if m['role'] != 'system']
        prompt_kwargs.pop("stream", None)  # guard against false stream
        return await self._model_client.messages.create(
            model=self._model,
            max_tokens=prompt_kwargs.pop('max_tokens', ANTHROPIC_MAX_TOKENS),
            messages=non_system,
            stream=True,
            **({"system": system} if system else {}),
            **prompt_kwargs
        )

    def extract_text(self, response: Message | MessageStreamEvent) -> str | None:
        """
        Extract LLM response text from Anthropic object

        :param response: Anthropic chat response
        :return: Text message if present, else None
        """
        # if message passed is a chunk
        if isinstance(response, Message):
            texts = [b.text for b in response.content if isinstance(b, TextBlock)]
            return "".join(texts) if texts else None
        # if message passed is a completed message
        if isinstance(response, RawContentBlockDeltaEvent) and isinstance(response.delta, TextDelta):
            return response.delta.text

        return None

    async def validate(self) -> None:
        """
        Verify Anthropic key is valid and has access to the requested model

        :raises InvalidAPIKeyError: If the Anthropic key is invalid
        :raises ModelNotFoundError: If requested model does not exist
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

    @property
    def tokenizer(self) -> AnthropicTokenizer:
        """
        :return: Anthropic tokenizer
        """
        return self._tokenizer
