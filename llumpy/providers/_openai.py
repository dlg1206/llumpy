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

    _vendor_client: OpenAI

    def __init__(self, model: str, api_key: str | None = None, base_url: str = OPENAI_BASE_URL):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use (Default: OPENAI_API_KEY env var)
        :param base_url: URL of api server (Default: OpenAI)
        :raises EnvironmentError: api_key is none and the 'OPENAI_API_KEY' env var is not defined
        """
        params = {'api_key': api_key if api_key else load_api_key(OPENAI_API_KEY_ENV)}
        if base_url:
            params['base_url'] = base_url
        super().__init__(model, OpenAI(**params))

    def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> ChatCompletion:
        """
        Prompt a model for OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion
        """
        messages: List[Any] = conversation.to_dicts()
        prompt_kwargs.pop("stream", None)  # guard against true stream
        return cast(ChatCompletion, self._vendor_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            **prompt_kwargs
        ))

    def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Stream[ChatCompletionChunk]:
        """
        Prompt a model for a streamed OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat stream
        """
        messages: List[Any] = conversation.to_dicts()
        prompt_kwargs.pop("stream", None)  # guard against false stream
        return cast(Stream[ChatCompletionChunk], self._vendor_client.chat.completions.create(
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

        :raises InvalidAPIKeyError: If the OpenAI key is invalid
        :raises ModelNotFoundError: If requested model does not exist
        :raises PermissionDeniedError: If key does not have access to requested model
        """
        try:
            self._vendor_client.models.retrieve(self._model)
        except AuthenticationError as e:
            raise InvalidAPIKeyError('OpenAI') from e
        except NotFoundError as e:
            raise ModelNotFoundError('OpenAI', self.model) from e
        except PermissionDeniedError as e:
            raise e


class AsyncOpenAIClient(AsyncModelClient):
    """Async client interface for interacting with OpenAI API"""

    _vendor_client: AsyncOpenAI

    def __init__(self, model: str, api_key: str | None = None, base_url: str = OPENAI_BASE_URL):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use (Default: None)
        :param base_url: URL of api server (Default: OpenAI)
        :raises EnvironmentError: api_key is none and the 'OPENAI_API_KEY' env var is not defined
        """
        params = {'api_key': api_key if api_key else load_api_key(OPENAI_API_KEY_ENV)}
        if base_url:
            params['base_url'] = base_url
        super().__init__(model, AsyncOpenAI(**params))

    async def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> ChatCompletion:
        """
        Prompt a model for OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion
        """
        messages: List[Any] = conversation.to_dicts()
        prompt_kwargs.pop("stream", None)  # guard against true stream
        return await self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=False,
            **prompt_kwargs
        )

    async def vendor_prompt_stream(self,
                                   conversation: Conversation,
                                   **prompt_kwargs: Any) -> AsyncStream[ChatCompletionChunk]:
        """
        Prompt a model for a streamed OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat stream
        """
        messages: List[Any] = conversation.to_dicts()
        prompt_kwargs.pop("stream", None)  # guard against false stream
        return await self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            **prompt_kwargs
        )

    def extract_text(self, response: ChatCompletion | ChatCompletionChunk) -> str | None:
        """
        Extract LLM response text from OpenAI object

        :param response: OpenAI chat response
        :return: Text message if present, else None
        """
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content
        if not response.choices:
            return None
        return response.choices[0].delta.content

    async def validate(self) -> None:
        """
        Verify OpenAI key is valid and has access to the requested model

        :raises InvalidAPIKeyError: If the OpenAI key is invalid
        :raises ModelNotFoundError: If requested model does not exist
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
