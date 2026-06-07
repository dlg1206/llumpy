"""
File: _exception.py

Description: Exceptions for llumpy operations

@author Derek Garcia
"""


class InvalidAPIKeyError(ValueError):
    """Failed to validate LLM API key"""

    def __init__(self, provider: str):
        """
        Failed to validate API key

        :param provider: Name of LLM provider
        """
        super().__init__(f"Failed to validate {provider} API key")
        self._provider = provider


class ModelNotFoundError(ValueError):
    """Requested model not found"""

    def __init__(self, provider: str, model_name: str):
        """
        Could not find requested model

        :param provider: Name of LLM provider
        :param model_name: Name of requested model
        """
        super().__init__(f"Could not find model '{model_name}' from {provider}")
        self._provider = provider
        self._model_name = model_name


class ExceededRetriesError(RuntimeError):
    """Exceed the number of retries"""

    def __init__(self, attempts: int):
        """
        Create new error

        :param attempts: Number of attempts made
        """
        self.attempts = attempts
