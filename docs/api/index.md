# API Reference

## Providers

> Prebuilt clients for LLM providers

### Anthropic

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~llumpy.providers.AnthropicClient
   ~llumpy.providers.AsyncAnthropicClient
```

### OpenAI

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~llumpy.providers.OpenAIClient
   ~llumpy.providers.AsyncOpenAIClient
```

### Ollama

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~llumpy.providers.OllamaClient
   ~llumpy.providers.AsyncOllamaClient
   ~llumpy.providers.InvalidOllamaServerError
```

## Core

> Core components for creating client interfaces and model chats

### Conversation

> Building blocks for creating prompts to LLMs

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~llumpy.core.ConversationBuilder
   ~llumpy.core.Conversation
   ~llumpy.core.Message
   ~llumpy.core.Role
```

### Base Clients

> Base clients for LLM clients

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~llumpy.core.ModelClient
   ~llumpy.core.AsyncModelClient
   ~llumpy.core.load_api_key
   ~llumpy.core.InvalidAPIKeyError
   ~llumpy.core.ModelNotFoundError
```

## Retry Handlers

> Automatic response validation and reprompting

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~llumpy.retry.RetryHandler
   ~llumpy.retry.AsyncRetryHandler
   ~llumpy.retry.JSONRetryHandler
   ~llumpy.retry.AsyncJSONRetryHandler
   ~llumpy.retry.ExceededRetriesError
```
