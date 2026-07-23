# Vendor Responses

To access vendor specific LLM responses, use the `vendor_prompt()` or `vender_prompt_stream()` methods.

```python
from llumpy.providers import OllamaClient
from llumpy.retry import JSONRetryHandler
from llumpy.core import ConversationBuilder

llama3_8b = OllamaClient('llama3', '8b')

conversation = ConversationBuilder().user("Hello!").build()
response = llama3_8b.vendor_prompt(conversation)

# ollama uses OpenAI APIs
print(type(response))
```

<img src="../_static/assets/vendor_prompt.png" alt="vendor type">
