# Custom Retry Handlers

Retry handlers validate the LLM response and automatically reprompts if fails.

```python
from llumpy.providers import OllamaClient
from llumpy.retry import JSONRetryHandler

llama3_8b = OllamaClient('llama3', '8b')

print(llama3_8b.prompt_one("Hello!", handler=JSONRetryHandler(), retries=2))
```

<img src="../_static/assets/retry_handler_fail.png" alt="terminal failure when exceed retries">

```python
from llumpy.providers import OllamaClient
from llumpy.retry import JSONRetryHandler

llama3_8b = OllamaClient('llama3', '8b')

print(llama3_8b.system("Only reply in JSON").user("Hello!").prompt(handler=JSONRetryHandler()))
```

<img src="../_static/assets/retry_handler_pass.png" alt="terminal success with handler">

## Creating Custom Handler

Inherit the {py:class}`~llumpy.retry.RetryHandler` or {py:class}`~llumpy.retry.AsyncRetryHandler` class and methods. See
{py:class}`~llumpy.retry.JSONRetryHandler` for an example implementation.

```python
from typing import Any, Tuple, Type
from llumpy.retry import RetryHandler, AsyncRetryHandler


class MyRetryHandler(RetryHandler):
    def _format(self, response: str) -> Any:
        """Attempt to format the response to validate it and raise an exception"""
        pass

    @property
    def _retry_on(self) -> Tuple[Type[Exception], ...]:
        """(Optional) Return the tuple of exceptions thrown by _format"""
        pass


class MyAsyncRetryHandler(AsyncRetryHandler):
    def _format(self, response: str) -> Any:
        """Attempt to format the response to validate it and raise an exception"""
        pass

    @property
    def _retry_on(self) -> Tuple[Type[Exception], ...]:
        """(Optional) Return the tuple of exceptions thrown by _format"""
        pass
```