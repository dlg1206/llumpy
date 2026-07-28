"""
File: _base.py

Description: Retry handler for formating prompts

@author Derek Garcia
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple, Type, Callable, Awaitable

from ._exceptions import ExceededRetriesError

DEFAULT_MAX_RETRIES = 5


class _FormatMixIn(ABC):
    @property
    def _retry_on(self) -> Tuple[Type[Exception], ...]:
        """
        Exceptions to retry on

        :return: Tuple of exception types to retry on
        """
        return (Exception,)

    @abstractmethod
    def _format(self, response: str) -> Any:
        """Attempt to format the response to validate it and raise an exception"""


class RetryHandler(_FormatMixIn, ABC):
    """Retry handler to validate LLM responses"""

    def try_prompt(self,
                   prompt_fn: Callable[[], Any],
                   extract_fn: Callable[[Any], str | None],
                   retries: int) -> Any:
        """
        Prompt an LLM to get a valid response

        :param prompt_fn: Prompt callback function
        :param extract_fn: Prompt text extraction callback function
        :param retries: Number of retries allowed
        :raises ExceededRetriesError: Exceed the number of permitted retries
        :return: Validated chat response text
        """
        # attempt generation
        last_exc = None
        for _ in range(retries):
            try:
                # prompt the model
                response = extract_fn(prompt_fn())
                if response:
                    return self._format(response)
            except self._retry_on as e:
                # try again
                last_exc = e
        # failed to prompt
        raise ExceededRetriesError(retries) from last_exc


class AsyncRetryHandler(_FormatMixIn, ABC):
    """Async retry handler to validate LLM responses"""

    async def try_prompt(self,
                         async_prompt_fn: Callable[[], Awaitable[Any]],
                         extract_fn: Callable[[Any], str | None],
                         retries: int) -> Any:
        """
        Prompt an LLM to get a valid response

        :param async_prompt_fn: Async prompt callback function
        :param extract_fn: Prompt text extraction callback function
        :param retries: Number of retries allowed
        :raises ExceededRetriesError: Exceed the number of permitted retries
        :return: Validated chat response text
        """
        # attempt generation
        last_exc = None
        for _ in range(retries):
            try:
                # prompt the model
                response = extract_fn(await async_prompt_fn())
                if response:
                    return self._format(response)
            except self._retry_on as e:
                # try again
                last_exc = e
        # failed to prompt
        raise ExceededRetriesError(retries) from last_exc
