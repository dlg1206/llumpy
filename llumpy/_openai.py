"""
File: _openai.py

Description: Client for interacting with OpenAI API

@author Derek Garcia
"""
from typing import List, Any

from openai import AuthenticationError, NotFoundError, PermissionDeniedError, Stream, AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llumpy._exception import InvalidAPIKeyError, ModelNotFoundError
from llumpy._message import Conversation, ConversationBuilder
from llumpy._model_client import ModelClient, AsyncModelClient, _load_api_key

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient(ModelClient):
    """Client interface for interacting with OpenAI API"""

    def __init__(self, model: str, api_key: str = None, base_url: str = OPENAI_BASE_URL):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use (Default: OPENAI_API_KEY_ENV)
        :param base_url: URL of api server (Default: OpenAI)
        :raises EnvironmentError: api_key is none and the 'OPEN_AI_API_KEY' env var is not defined
        """
        super().__init__(model)
        params = {'api_key': api_key if api_key else _load_api_key(OPENAI_API_KEY_ENV)}
        if base_url:
            params['base_url'] = base_url
        self._model_client = OpenAI(**params)

    def prompt_completion(self,
                          conversation: Conversation,
                          **prompt_kwargs: Any) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Prompt a model for OpenAI chat completion

        :param conversation: Messages to send to llm
        :param prompt_kwargs: kwargs for chat
        :return: Chat completion or stream
        """
        messages: List[Any] = conversation.to_dicts()
        return self._model_client.chat.completions.create(
            model=self._model,
            messages=messages,
            **prompt_kwargs
        )

    def prompt_one(self, message: str, **prompt_kwargs: Any) -> str | None:
        """
        Prompt a model for simple text return

        :param message: Message to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        completion = self.prompt_completion(ConversationBuilder().user(message).build(), stream=False, **prompt_kwargs)
        return completion.choices[0].message.content

    def prompt_many(self, conversation: Conversation, **prompt_kwargs: Any) -> str | None:
        """
        Prompt a model for simple text return

        :param conversation: Conversation with prompt to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        completion = self.prompt_completion(conversation, stream=False, **prompt_kwargs)
        return completion.choices[0].message.content

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

    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        """
        Initialize connection to OpenAI compatible server

        :param model: LLM to use
        :param api_key: API key to use (Default: None)
        :param base_url: URL of api server (Default: OpenAI)
        """
        super().__init__(model)
        params = {'api_key': api_key if api_key else _load_api_key(OPENAI_API_KEY_ENV)}
        if base_url:
            params['base_url'] = base_url
        self._model_client = AsyncOpenAI(**params)

    async def prompt_completion(self,
                                conversation: Conversation,
                                **prompt_kwargs: Any) -> ChatCompletion | Stream[ChatCompletionChunk]:
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
            **prompt_kwargs
        )

    async def prompt_one(self, message: str, **prompt_kwargs: Any) -> str | None:
        """
        Prompt a model for simple text return

        :param message: Message to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        completion = await self.prompt_completion(ConversationBuilder().user(message).build(), stream=False,
                                                  **prompt_kwargs)
        return completion.choices[0].message.content

    async def prompt_many(self, conversation: Conversation, **prompt_kwargs: Any) -> str | None:
        """
        Prompt a model for simple text return

        :param conversation: Conversation with prompt to send to LLM
        :param prompt_kwargs: kwargs for chat
        :return: Completed chat response text
        """
        completion = await self.prompt_completion(conversation, stream=False, **prompt_kwargs)
        return completion.choices[0].message.content

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
