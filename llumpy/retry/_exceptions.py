"""
File: _exceptions.py

Description: Exceptions for retry handlers

@author Derek Garcia
"""


class ExceededRetriesError(RuntimeError):
    """Exceed the number of retries"""

    def __init__(self, attempts: int):
        """
        Create new error

        :param attempts: Number of attempts made
        """
        self.attempts = attempts
