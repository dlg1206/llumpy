"""
File: exeception.py

Description: Exceptions for llumpy operations

@author Derek Garcia
"""


class InvalidAPIKeyException(ValueError):
    """
    Failed to validate LLM API key
    """

    def __init__(self, provider: str):
        """
        Failed to validate API key
        """
        super().__init__(f"Failed to validate {provider} API key")
        self._provider = provider
