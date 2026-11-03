# Chat

A chat for a continuous conversation with an LLM.

```python
from llumpy.core import AsyncChat
from llumpy.providers import AsyncOllamaClient

chat = AsyncChat(AsyncOllamaClient('llama3', '8b'))

print((await chat.message("Hello World!")))
print((await chat.message("What was my last message")))
# 'Your last message was: "Hello World!"'
```

To get the chat history, use the `history` property:

```python3
from llumpy.core import AsyncChat
from llumpy.providers import AsyncOllamaClient

chat = AsyncChat(AsyncOllamaClient('llama3', '8b'))
print(chat.history)
```

## Streaming Response

Stream token responses from LLM instead of waiting for complete response.

```python3
from llumpy.core import AsyncChat
from llumpy.providers import AsyncOllamaClient

llama3_8b = AsyncOllamaClient('llama3', '8b')
chat = AsyncChat(llama3_8b)
async for chunk in chat.stream("Hello World"):
    print(llama3_8b.extract_text(chunk) , end="_", flush=True)
```

See [Chat Options](../advanced_usage/chat_options.md) for advanced usage.