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
                     AsyncOpenAIClient as AsyncOpenAIClient)

from .core import (Role as Role,
                   Message as Message,
                   Conversation as Conversation,
                   ConversationBuilder as ConversationBuilder,
                   ModelClient as ModelClient,
                   AsyncModelClient as AsyncModelClient,
                   InvalidAPIKeyError as InvalidAPIKeyError,
                   ModelNotFoundError as ModelNotFoundError)

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
    # core
    "ConversationBuilder", "Role", "Message", "Conversation",
    "ModelClient", "AsyncModelClient",
    "InvalidAPIKeyError", "ModelNotFoundError",
    # retry
    "ExceededRetriesError",
    "RetryHandler", "AsyncRetryHandler",
    "JSONRetryHandler", "AsyncJSONRetryHandler"
]
