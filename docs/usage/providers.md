# Provider Clients

Anthropic, OpenAI, and Ollama clients have been developed and are ready to use out of the box. For custom clients,
see [here](../advanced_usage/custom_clients.md).

```{warning}
Clients will fail to be initialized if API keys are invalid, models do not exist, or the key does not have access to the requested model.
```

The {py:class}`ProviderFactory` is the main means of generating clients and handles any validation or additional setup.
Clients instantiated without the factory, but do not have the same guarantees as ones created with the factory.

## Anthropic

### Factory

```python
from llumpy.providers import ProviderFactory

# API key env var: ANTHROPIC_API_KEY
claude = ProviderFactory.anthropic('claude-sonnet-4-6')
```

### Standalone

```python
from llumpy.providers import AnthropicClient

claude = AnthropicClient('claude-sonnet-4-6')
claude.validate()
```

The `ANTHROPIC_API_KEY` env variable **MUST** be set. Anthropic models can be
found [here](https://platform.claude.com/docs/en/about-claude/models/overview).

## OpenAI

### Factory

```python
from llumpy.providers import ProviderFactory

# API key env var: OPENAI_API_KEY
gpt = ProviderFactory.openai('gpt-5.4')
gpt.validate()
```

### Standalone

```python
from llumpy.providers import OpenAIClient

gpt = OpenAIClient('gpt-5.4')
gpt.validate()
```

The `OPENAI_API_KEY` env variable **MUST** be set. OpenAI models can be
found [here](https://developers.openai.com/api/docs/models).

## Ollama

### Factory

```python
from llumpy.providers import ProviderFactory

# Ollama server url env var: OLLAMA_SERVER_URL (Default: http://localhost:11434)
llama3_latest = ProviderFactory.ollama('llama3')  # default ':latest'
llama3_8b = ProviderFactory.ollama('llama3', '8b')
```

The {py:class}`ProviderFactory` also handles downloading the model if it has not been download locally. To skip this
step, use the `defer_download` param:

```python
from llumpy.providers import ProviderFactory

llama3_latest = ProviderFactory.ollama('llama3', defer_download=True)
```

If the model is not downloaded, the `download_model()` method **MUST** be called before prompting to ensure the model is
downloaded. Deferring is useful since models can take a while to download, so the download task can be done in the
background while other code is running.

The {py:class}`OllamaClient` also supports a `wake_up()` method that sends a short message to "warm up" the LLM so the
first actual prompt does not take a long time if this is a fresh running Ollama instance. To disable this, use the
`skip_wakeup` param:

```python
from llumpy.providers import ProviderFactory

llama3_latest = ProviderFactory.ollama('llama3', skip_wakeup=True)
```

This is often used with the `defer_download` param since if the model has not been downloaded, a 
{py:class}`ModelNotDownloadedError` will be thrown.

### Standalone

```python
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')
llama3_8b.validate()  # ensure the model exists
llama3_8b.download_model()  # download the model if it hasn't been downloaded already
```

The Ollama server url will be set using the following precedence:

1. `server_url` param
2. `OLLAMA_SERVER_URL` env variable
3. Default (http://localhost:11434)

Ollama models can be found [here](https://ollama.com/search).