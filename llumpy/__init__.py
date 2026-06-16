"""
File: __init__.py

Description:

@author Derek Garcia
"""
from .client import (AnthropicClient as AnthropicClient,
                     AsyncAnthropicClient as AsyncAnthropicClient,
                     OllamaClient as OllamaClient,
                     AsyncOllamaClient as AsyncOllamaClient,
                     OpenAIClient as OpenAIClient,
                     AsyncOpenAIClient as AsyncOpenAIClient,
                     InvalidAPIKeyError as InvalidAPIKeyError,
                     ModelNotFoundError as ModelNotFoundError, )

from .core import (Conversation as Conversation,
                   Message as Message,
                   ConversationBuilder as ConversationBuilder,
                   ModelClient as ModelClient,
                   AsyncModelClient as AsyncModelClient)

from .retry import (ExceededRetriesError as ExceededRetriesError,
                    RetryHandler as RetryHandler,
                    AsyncRetryHandler as AsyncRetryHandler,
                    JSONRetryHandler as JSONRetryHandler,
                    AsyncJSONRetryHandler as AsyncJSONRetryHandler)

__all__ = [
    # clients
    "OpenAIClient", "AsyncOpenAIClient",
    "AnthropicClient", "AsyncAnthropicClient",
    "OllamaClient", "AsyncOllamaClient",
    "InvalidAPIKeyError", "ModelNotFoundError",
    # core
    "Conversation", "Message", "ConversationBuilder",
    "ModelClient", "AsyncModelClient",
    # retry
    "ExceededRetriesError",
    "RetryHandler", "AsyncRetryHandler",
    "JSONRetryHandler", "AsyncJSONRetryHandler"
]
