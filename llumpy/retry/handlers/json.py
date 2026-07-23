"""
File: json.py

Description: JSON handlers for parsing JSON from LLM response

@author Derek Garcia
"""
import json
import re
from abc import ABC
from json import JSONDecodeError
from typing import Tuple, Type, Dict, Any

from .._base import RetryHandler, AsyncRetryHandler

JSON_RE = re.compile(r'{[\w\W]*}')


class _JSONRetryHandlerBase(ABC):
    @property
    def _retry_on(self) -> Tuple[Type[Exception], ...]:
        """
        Retry on JSON decode errors

        :return: Tuple of exception types to retry on
        """
        return (JSONDecodeError,)

    @staticmethod
    def _format(response: str) -> Dict[str, Any]:
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


class JSONRetryHandler(_JSONRetryHandlerBase, RetryHandler):
    """Handler for parsing JSON from LLM responses"""


class AsyncJSONRetryHandler(_JSONRetryHandlerBase, AsyncRetryHandler):
    """Async handler for parsing JSON from LLM responses"""
