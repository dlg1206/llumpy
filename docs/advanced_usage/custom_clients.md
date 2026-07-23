# Custom Clients

Inherit the {py:class}`~llumpy.core.ModelClient` or {py:class}`~llumpy.core.AsyncModelClient` classes and methods.

```python
from typing import Any
from llumpy.core import ModelClient, Conversation, AsyncModelClient


class MyLLMClient(ModelClient):
    def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response object"""
        pass

    def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response stream object"""
        pass

    def extract_text(self, response: Any) -> str | None:
        """Extract text from vendor-specific response object"""
        pass

    def validate(self) -> None:
        """Validate the client is ready to use"""
        pass


class MyAsyncLLMClient(AsyncModelClient):
    async def vendor_prompt(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response object"""
        pass

    async def vendor_prompt_stream(self, conversation: Conversation, **prompt_kwargs: Any) -> Any:
        """Raw API call, returns vendor-specific response stream object"""
        pass

    def extract_text(self, response: Any) -> str | None:
        """Extract text from vendor-specific response object"""
        pass

    async def validate(self) -> None:
        """Validate the client is ready to use"""
        pass
```
