"""
File: _retry_handler.py

Description: Retry handler for formating prompts

@author Derek Garcia
"""
import json
import re
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Dict, Tuple, Type

from openai.types.beta.realtime.conversation_created_event import Conversation

from llumpy._model_client import ModelClient, AsyncModelClient

JSON_RE = re.compile(r'{[\w\W]*}')

DEFAULT_MAX_RETRIES = 5


class ExceededRetriesError(RuntimeError):
    """Exceed the number of retries"""

    def __init__(self, model: str, attempts: int):
        """
        Create new error

        :param model: Model prompted
        :param attempts: Number of attempts made
        """
        super().__init__(f"Failed to generate valid response from {model} after {attempts} attempts")
        self.model = model
        self.attempts = attempts


class _FormatMixIn(ABC):
    @property
    def _retry_on(self) -> Tuple[Type[Exception], ...]:
        """
        Handled executions to retry on

        :return: Generic exception
        """
        return (Exception,)

    @abstractmethod
    def _format(self, response: str) -> Any:
        """Attempt to format the response to validate it and raise an exception"""
        pass


class RetryHandler(_FormatMixIn, ABC):

    def try_prompt(self, model: ModelClient, conversation: Conversation, retries: int, **prompt_kwargs) -> Any:
        """
        Prompt an LLM to get a valid response

        :param model: LLM to prompt
        :param conversation: Conversation with prompt to send to LLM
        :param retries: Number of retries allowed
        :param prompt_kwargs: kwargs for chat
        :raises ExceededRetriesError: Exceed the number of permitted retries
        :return: Validated chat response text
        """
        # attempt generation
        for attempt in range(retries):
            try:
                # prompt the model
                response = model.extract_text(model.prompt(conversation, stream=False, **prompt_kwargs))
                if response:
                    return self._format(response)
            except self._retry_on:
                # try again
                continue
        # failed to prompt
        raise ExceededRetriesError(model.model, retries)


class AsyncRetryHandler(_FormatMixIn, ABC):

    async def try_prompt(self, model: AsyncModelClient, conversation: Conversation, retries: int,
                         **prompt_kwargs) -> Any:
        """
        Prompt an LLM to get a valid response

        :param model: LLM to prompt
        :param conversation: Conversation with prompt to send to LLM
        :param retries: Number of retries allowed
        :param prompt_kwargs: kwargs for chat
        :raises ExceededRetriesError: Exceed the number of permitted retries
        :return: Validated chat response text
        """
        # attempt generation
        for attempt in range(retries):
            try:
                # prompt the model
                response = model.extract_text(await model.prompt(conversation, stream=False, **prompt_kwargs))
                if response:
                    return self._format(response)
            except self._retry_on:
                # try again
                continue
        # failed to prompt
        raise ExceededRetriesError(model.model, retries)


class JSONRetryHandler(RetryHandler, AsyncRetryHandler):
    """Handler for parsing JSON from LLM responses"""

    @property
    def _retry_on(self) -> Tuple[Type[Exception], ...]:
        """
        Retry on JSON decode

        :return JSON decode errors
        """
        return (JSONDecodeError,)

    def _format(self, response: str) -> Dict[str, Any]:
        """
        Validate the response contains a valid JSON object

        :param response: Response text to validate contains JSON
        :raises JSONDecodeError: If failed to parse JSON from text
        :return: JSON object
        """
        match = JSON_RE.search(response.strip())
        if not match:
            raise JSONDecodeError("Text contains no JSON", response, 0)
        return json.loads(match.group())
