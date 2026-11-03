# Chat Options

Options are available for both {py:class}`~llumpy.core.Chat` and {py:class}`~llumpy.core.AsyncChat`.

## Context

Context to start the chat with. The context must end on a `system` or an `assistant` message.

```python3
from llumpy.core import Chat, ConversationBuilder
from llumpy.providers import OllamaClient

chat = Chat(OllamaClient('llama3', '8b'), context=ConversationBuilder().system("Speak like a pirate"))

print(chat.message("Hello World!"))
print(chat.message("What was my last message"))
# Shiver me timbers! Yer last message be... "Hello World!" Arrr, don't ye worry about it not being a booty-ful greeting, matey! World was just fine with yer "hello"!

chat.clear_chat()  # reset the chat back to the starting context
```

## Context Window

A {py:class}`~llumpy.core.ContextWindow` can be provided to limit the number of tokens an entire chat is allowed to use.
If a context window is exceeded, a {py:class}`~llumpy.core.ExceedContextWindowError` is thrown.

```python3
from llumpy.core import Chat, ContextWindow
from llumpy.providers import AnthropicClient

claude = AnthropicClient('claude-sonnet-4-6')
chat = Chat(claude, context_window=ContextWindow(claude.tokenizer, max_tokens=1))

print(chat.message("Hello World!"))
# llumpy.core._chat.ExceedContextWindowError: Exceeded context window: requested 10 tokens, but only 1 are allowed
```

To check the remaining tokens, use the `remaining_tokens` property:

```python3
from llumpy.core import Chat, ContextWindow
from llumpy.providers import AnthropicClient

claude = AnthropicClient('claude-sonnet-4-6')
chat = Chat(claude, context_window=ContextWindow(claude.tokenizer, max_tokens=10))

print(chat.message("Hello World!"))
print(chat.remaining_tokens)
# Hello! 👋 Welcome! How are you doing today? Is there something I can help you with? 😊
# -26
```

### Custom Tokenizers

Only `Anthropic` and `OpenAI` clients have the built in `tokenizer` methods since Ollama embeddings can vary.
Implement the {py:class}`~llumpy.core.Tokenizer` class to use with the context window.

```python3
from llumpy.providers import OllamaClient
from llumpy.core import Chat, Tokenizer, History, ContextWindow


class Llama3Tokenizer(Tokenizer):
    def __init__(self):
        from transformers import AutoTokenizer  # optional dependency
        self._tok = AutoTokenizer.from_pretrained("llama3:8b")

    def count_tokens(self, text: str | History) -> int:
        if isinstance(text, str):
            return len(self._tok.encode(text))
        return len(self._tok.apply_chat_template(text.to_dicts(), tokenize=True, add_generation_prompt=True))


chat = Chat(
    OllamaClient('llama3', '8b'),
    context_window=ContextWindow(Llama3Tokenizer(), max_tokens=10)
)
```