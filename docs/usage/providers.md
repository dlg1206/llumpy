# Provider Clients

Anthropic, OpenAI, and Ollama clients have been developed and are ready to use out of the box. For custom clients,
see [here](../advanced_usage/custom_clients.md).

```{warning}
Clients will fail to be initialized if API keys are invalid, models do not exist, or the key does not have access to the requested model.
```

## Anthropic

```python
from llumpy.providers import AnthropicClient

claude = AnthropicClient('claude-sonnet-4-6')
```

The `ANTHROPIC_API_KEY` env variable **MUST** be set. Anthropic models can be
found [here](https://platform.claude.com/docs/en/about-claude/models/overview).

## OpenAI

```python
from llumpy.providers import OpenAIClient

gpt = OpenAIClient('gpt-5.4')
```

The `OPENAI_API_KEY` env variable **MUST** be set. OpenAI models can be
found [here](https://developers.openai.com/api/docs/models).

## Ollama

```python
from llumpy.providers import OllamaClient

ollama = OllamaClient('llama3', '8b')
```

The Ollama server url will be set using the following precedence:

1. `server_url` param
2. `OLLAMA_SERVER_URL` env variable
3. Default (http://localhost:11434)

Ollama models can be found [here](https://ollama.com/search).