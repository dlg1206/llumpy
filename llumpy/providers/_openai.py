"""
File: _openai.py

Description: Client for interacting with OpenAI API

@author Derek Garcia
"""
from typing import List, Any, cast

from openai import AuthenticationError, NotFoundError, PermissionDeniedError, Stream, AsyncOpenAI, OpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..core import Conversation, ModelClient, AsyncModelClient, load_api_key, InvalidAPIKeyError, ModelNotFoundError

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient(ModelClient):
    """Client interface for interacting with OpenAI API"""

    def __init__(self, model: str, api_key: str | None = None, base_url: str = OPENAI_BASE_URL):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use (Default: OPENAI_API_KEY_ENV)
        :param base_url: URL of api server (Default: OpenAI)
        :raises EnvironmentError: api_key is none and the 'OPENAI_API_KEY_ENV' env var is not defined
        """
        super().__init__(model)
        params = {'api_key': api_key if api_key else load_api_key(OPENAI_API_KEY_ENV)}
        if base_url:
            params['base_url'] = base_url
        self._model_client = OpenAI(**params)

    def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> ChatCompletion:
        """
        Prompt a model for OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion or stream
        """
        messages: List[Any] = conversation.to_dicts()
        return cast(ChatCompletion, self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            **prompt_kwargs
        ))

    def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Stream[ChatCompletionChunk]:
        """
        Prompt a model for OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat stream
        """
        messages: List[Any] = conversation.to_dicts()
        return cast(Stream[ChatCompletionChunk], self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            **prompt_kwargs
        ))

    def extract_text(self, response: ChatCompletion | ChatCompletionChunk) -> str | None:
        """
        Extract LLM response text from OpenAI object

        :param response: OpenAI chat response
        :return: Text message if present, else None
        """
        return response.choices[0].delta.content \
            if isinstance(response, ChatCompletionChunk) else response.choices[0].message.content

    def validate(self) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If the OpenAI key is invalid
        :raises ModelNotFoundExecution: If requested model does not exist
        :raises PermissionDeniedError: If key does not have access to requested model
        """
        try:
            self._model_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyError('OpenAI') from e
        except NotFoundError as e:
            raise ModelNotFoundError('OpenAI', self.model) from e
        except PermissionDeniedError as e:
            raise e


class AsyncOpenAIClient(AsyncModelClient):
    """Async client interface for interacting with OpenAI API"""

    def __init__(self, model: str, api_key: str | None = None, base_url: str = OPENAI_BASE_URL):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use (Default: None)
        :param base_url: URL of api server (Default: OpenAI)
        :raises EnvironmentError: api_key is none and the 'OPEN_AI_API_KEY' env var is not defined
        """
        super().__init__(model)
        params = {'api_key': api_key if api_key else load_api_key(OPENAI_API_KEY_ENV)}
        if base_url:
            params['base_url'] = base_url
        self._model_client = AsyncOpenAI(**params)

    async def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> ChatCompletion:
        """
        Prompt a model for OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion or stream
        """
        messages: List[Any] = conversation.to_dicts()
        return await self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            **prompt_kwargs
        )

    async def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> AsyncStream[
        ChatCompletionChunk]:
        """
        Prompt a model for OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat stream
        """
        messages: List[Any] = conversation.to_dicts()
        return cast(AsyncStream[ChatCompletionChunk], self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            **prompt_kwargs
        ))

    def extract_text(self, response: ChatCompletion) -> str | None:
        """
        Extract LLM response text from OpenAI object

        :param response: OpenAI chat response
        :return: Text message if present, else None
        """
        return response.choices[0].message.content

    async def validate(self) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model

        :raises InvalidAPIKeyException: If the OpenAI key is invalid
        :raises ModelNotFoundExecution: If requested model does not exist
        :raises PermissionDeniedError: If key does not have access to requested model
        """
        try:
            await self._model_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyError('OpenAI') from e
        except NotFoundError as e:
            raise ModelNotFoundError('OpenAI', self.model) from e
        except PermissionDeniedError as e:
            raise e
