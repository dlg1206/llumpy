<h1>
    <img alt="llumpy mascot" src="assets/llumpy.png" width="150">
    llumpy
</h1>

> Provider-agnostic client wrapper for prompting LLMs

## 🤖 Supported Providers

- [Anthropic](https://dlg1206.github.io/llumpy/usage/providers.html#anthropic)
- [OpenAI](https://dlg1206.github.io/llumpy/usage/providers.html#openai)
- [Ollama](https://dlg1206.github.io/llumpy/usage/providers.html#ollama)

## ✨ Features

- [One](https://dlg1206.github.io/llumpy/usage/prompting.html#one-shot)
  or [Few](https://dlg1206.github.io/llumpy/usage/prompting.html#few-shot) Shot Prompting
- [Streaming LLM Responses](https://dlg1206.github.io/llumpy/usage/prompting.html#streaming-response)
- [Load Prompts From File](https://dlg1206.github.io/llumpy/usage/prompting.html#file-param)
- [Asynchronous Variants](https://dlg1206.github.io/llumpy/async/index.html)
- [Automatic Retry Handlers](https://dlg1206.github.io/llumpy/advanced_usage/custom_retry_handlers.html)
- [Custom Provider Clients](https://dlg1206.github.io/llumpy/advanced_usage/custom_clients.html)

## Installation

```bash
pip install git+https://github.com/dlg1206/llumpy.git
```

or add to `requirements.txt`

```txt
llumpy @ git+https://github.com/dlg1206/llumpy.git
```

## Quickstart

1. Create a client

```python
from llumpy.providers import AnthropicClient, OpenAIClient, OllamaClient

# API key env var: OPENAI_API_KEY
gpt = OpenAIClient('gpt-5.4')

# API key env var: ANTHROPIC_API_KEY
claude = AnthropicClient('claude-sonnet-4-6')

# Ollama server url env var: OLLAMA_SERVER_URL (Default: http://localhost:11434)
llama3_latest = OllamaClient('llama3')  # default ':latest'
llama3_8b = OllamaClient('llama3', '8b')
```

> [!WARNING]
> Clients will fail to be initialized if API keys are bad, models do not exist, or key
> does not have access to that model.

2. Prompt

```python
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')
response = llama3_8b.prompt_one("Hello!")
print(response)
```

<img src="assets/prompt_one.png" alt="terminal output simple prompt">

See [documentation](https://dlg1206.github.io/llumpy) for additional usage.