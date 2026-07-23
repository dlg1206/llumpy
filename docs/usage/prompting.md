# Prompting

## One-shot

Prompt an LLM with a single message.

```python
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')

llama3_8b.prompt_one("Hello!")
```

## Few-shot

Prompt an LLM with a multiple messages.

### Ephemeral Prompting

```python3
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')
(llama3_8b.system("You are a pirate.")
 .user("What is the weather like today?")
 .assistant("Arrr matey! Skies be grey!")
 .user("What should I wear?")
 .prompt())
```

The messages are ephemeral and will not be included in the next prompt. (See
[Conversation](../advanced_usage/conversation.md) to save prebuilt conversations). Ephemeral prompting also ensures
messages are always valid and follow the grammar below:

```text
base        ::= system_p | user_p
system_p    ::= <system prompt> user_p
user_p      ::= <user prompt> | <user prompt> assistant_p
assistant_p ::= <assistant prompt> | <assistant prompt> user_p
```

When the grammar terminates, then the prompt option is available. This also guarantees that:

1. A user prompt must follow a system or assistant prompt
2. An assistant prompt can only follow a user prompt
3. A conversation can be sent to an LLM only if the last message was a user or assistant prompt[^1]

### Conversation

Clients can also be prompted using a prebuilt {py:class}`~llumpy.core.Conversation` (See
[Conversation](../advanced_usage/conversation.md)) using `prompt_many()`.

```python
from llumpy.core import ConversationBuilder
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')
conversation = (
    ConversationBuilder()
    .system("You are a pirate.")
    .user("What is the weather like today?")
    .assistant("Arrr matey! Skies be grey!")
    .user("What should I wear?")
    .build()
)

llama3_8b.prompt_many(conversation)
```

### Streaming Response

Stream token responses from LLM instead of waiting for complete response. Not supported by `prompt_one()` or ephemeral
prompting.

```python
from llumpy.core import ConversationBuilder
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')
conversation = ConversationBuilder().user("Hello!").build()

for chunk in llama3_8b.prompt_stream(conversation):
    print(llama3_8b.extract_text(chunk), end="_", flush=True)
```

<img src="../_static/assets/prompt_stream.png" alt="terminal output for streamed response">

## Additional Params

### File Param

`prompt_one()` and ephemeral prompts (`system()`, `user()`, `assistant()`) also support a `file` param to read text
directly from a file.

```python
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')

# prompt one
llama3_8b.prompt_one("Hello!", file="prompts/user.prompt")
# ephemeral prompt
llama3_8b.user(file="prompts/user.prompt").prompt()
```

`content` and `file` are mutually exclusive, using both (`user("foo", file="prompts/user.prompt")`) will result in a
{py:class}`ValueError`.

### Retry Handler Param

If the LLM response should pass a test to ensure it is formatted correctly, includes required data, etc, the {py:class}
`~llumpy.retry.RetryHandler` can be used to automatically reprompt the LLM with same conversation (See
[Custom Retry Handlers](../advanced_usage/custom_retry_handlers.md)). The `retries` param can also be used if requested
(Default: 5).

```python
from llumpy.providers import OllamaClient
from llumpy.retry import JSONRetryHandler
from llumpy.core import Conversation

llama3_8b = OllamaClient('llama3', '8b')

# prompt one
llama3_8b.prompt_one("Hello!", handler=JSONRetryHandler())
llama3_8b.prompt_one("Hello!", handler=JSONRetryHandler(), retries=2)
# ephemeral prompt
llama3_8b.user("Hello!").prompt(handler=JSONRetryHandler())
llama3_8b.user("Hello!").prompt(handler=JSONRetryHandler(), retries=2)
# prompt many
conversation: Conversation = None
llama3_8b.prompt_many(conversation, handler=JSONRetryHandler())
llama3_8b.prompt_many(conversation, handler=JSONRetryHandler(), retries=2)
```

Currently, the {py:class}`~llumpy.retry.JSONRetryHandler` is the only handler that parses the LLM response into a JSON
object.

### LLM API Params

Additional LLM params supported by the underlying provider API can be passed as a kwargs argument and will be used in
the actual request, for example `temperature`:

```python
from llumpy.providers import OllamaClient
from llumpy.core import Conversation

llama3_8b = OllamaClient('llama3', '8b')

# prompt one
llama3_8b.prompt_one("Hello!", temperature=1.0)
# ephemeral prompt
llama3_8b.user("Hello!").prompt(temperature=1.0)
# prompt many
conversation: Conversation = None
llama3_8b.prompt_many(conversation, temperature=1.0)
# stream
for chunk in llama3_8b.prompt_stream(conversation, temperature=1.0):
    print(llama3_8b.extract_text(chunk), end="_", flush=True)
```

[^1]: Ending on an assistant prompt to support 'prefill' technique where the starting response is provided to the LLM.