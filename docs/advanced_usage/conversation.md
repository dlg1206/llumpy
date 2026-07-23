# Conversation

To build up a conversation that can be resued, use the {py:class}`~llumpy.core.ConversationBuilder`.

```python
from llumpy.core import ConversationBuilder

conversation = (
    ConversationBuilder()
    .system("You are a pirate.")
    .user("What is the weather like today?")
    .assistant("Arrr matey! Skies be grey!")
    .user("What should I wear?")
    .build()
)
```

The {py:class}`~llumpy.core.ConversationBuilder` ensures messages are always valid and follow the grammar below:

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

## File Param

`system()`, `user()`, and `assistant()` also support a `file` param to read text directly from a file

```python
from llumpy.core import ConversationBuilder

conversation = ConversationBuilder().user(file="prompts/user.prompt").build()
```

`content` and `file` are mutually exclusive, using both (`user("foo", file="prompts/user.prompt")`) will result in a
{py:class}`ValueError`.

## Ephemeral Endings

The {py:class}`~llumpy.core.ConversationBuilder` supports ephemeral endings, allowing for a root conversation to be
reused with only the final prompt swapped out like so:

```python
from llumpy.core import ConversationBuilder

builder = ConversationBuilder().system("Foo")
for tail in ['bar', 'baz']:
    print(builder.build_with_user(tail))
```

<img src="../_static/assets/build_with_usage.png" alt="terminal output build with">

`build_with_assistant()` is also available if ending on a user prompt:

```python
from llumpy.core import ConversationBuilder

builder = ConversationBuilder().system("Foo").user("Bar")

buz_convo = builder.build_with_assistant("buz")
baf_convo = builder.build_with_assistant("baf")
```

`build_with_user()` and `build_with_assistant()`also support a `file` param to read text directly from a file

```python
from llumpy.providers import OllamaClient

llama3_8b = OllamaClient('llama3', '8b')

llama3_8b.user(file="prompts/user.prompt").prompt()
```

`content` and `file` are mutually exclusive, using both (`build_with_user("foo", file="prompts/user.prompt")`) will
result in a {py:class}`ValueError`.

[^1]: Ending on an assistant prompt to support 'prefill' technique where the starting response is provided to the LLM.