# Chat

A chat for a continuous conversation with an LLM.

```python
from llumpy.core import Chat
from llumpy.providers import OllamaClient

chat = Chat(OllamaClient('llama3', '8b'))

print(chat.message("Hello World!"))
print(chat.message("What was my last message"))
# 'Your last message was: "Hello World!"'
```

To get the chat history, use the `history` property:

```python3
from llumpy.core import Chat
from llumpy.providers import OllamaClient

chat = Chat(OllamaClient('llama3', '8b'))
print(chat.history)
```

## Streaming Response

Stream token responses from LLM instead of waiting for complete response.

```python3
from llumpy.core import Chat
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')
chat = Chat(llama3_8b)
for chunk in chat.stream("Hello World"):
    print(llama3_8b.extract_text(chunk), end="_", flush=True)
```

See [Chat Options](../advanced_usage/chat_options.md) for advanced usage.