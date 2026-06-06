"""
File: _message.py

Description: Model for messages to send to LLMs

@author Derek Garcia
"""
from dataclasses import dataclass, asdict
from enum import StrEnum
from typing import List, Dict


class Role(StrEnum):
    """Roles for LLM conversation"""
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


@dataclass(frozen=True)
class Message:
    """DTO for LLM messages"""
    role: Role
    content: str

    def to_dict(self) -> Dict:
        """
        :return: DTO as dict
        """
        return asdict(self)


@dataclass(frozen=True)
class Conversation:
    """Immutable conversation for LLM"""
    messages: List[Message]

    def to_dicts(self) -> List[Dict]:
        return [m.to_dict() for m in self.messages]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)


class ConversationBuilder:
    """Conversation builder to ensure messages are in a valid order"""

    def __init__(self):
        """
        Create new builder
        """
        self._messages: List[Message] = []

    def system(self, content: str) -> "ConversationBuilder._UserStep":
        """
        Add a system message

        :param content: Content of system message
        :return: User step
        """
        self._messages.append(Message(Role.SYSTEM, content))
        return ConversationBuilder._UserStep(self)

    def system_from_file(self, file: str) -> "ConversationBuilder._UserStep":
        """
        Add a system message from file

        :param file: File to read content from
        :return: User step
        """
        return self.system(_read_file(file))

    def user(self, content: str) -> "ConversationBuilder._AssistantOrBuildStep":
        """
        Add a user message

        :param content: Content of user message
        :return: Assistant or build step
        """
        self._messages.append(Message(Role.USER, content))
        return ConversationBuilder._AssistantOrBuildStep(self)

    def user_from_file(self, file: str) -> "ConversationBuilder._AssistantOrBuildStep":
        """
        Add a user message from file

        :param file: File to read content from
        :return: Assistant or build step
        """
        return self.user(_read_file(file))

    def build(self) -> Conversation:
        """
        :return: Immutable conversation
        """
        return Conversation(self._messages)

    class _UserStep:
        """User turn in the conversation"""

        def __init__(self, builder: "ConversationBuilder"):
            """
            Create the user turn

            :param builder: Conversation builder
            """
            self._builder = builder

        def user(self, content: str) -> "ConversationBuilder._AssistantOrBuildStep":
            """
            Add a user message

            :param content: Content of user message
            :return: Assistant or build step
            """
            return self._builder.user(content)

        def user_from_file(self, file: str) -> "ConversationBuilder._AssistantOrBuildStep":
            """
            Add a user message from file

            :param file: File to read content from
            :return: Assistant or build step
            """
            return self.user(_read_file(file))

    class _AssistantOrBuildStep:
        """Assistant or build step in the conversation"""

        def __init__(self, builder: "ConversationBuilder"):
            """
            Create the assistant or build turn
            :param builder: Conversation builder
            """
            self._builder = builder

        def assistant(self, content: str) -> "ConversationBuilder._UserStep":
            """
            Add an assistant message

            :param content: Content of assistant message
            :return: User step
            """
            self._builder._messages.append(Message(Role.ASSISTANT, content))
            return ConversationBuilder._UserStep(self._builder)

        def assistant_from_file(self, file: str) -> "ConversationBuilder._UserStep":
            """
            Add an assistant message from file

            :param file: File to read content from
            :return: User step
            """
            return self.assistant(_read_file(file))

        def build(self) -> Conversation:
            """
            :return: List of messages
            """
            return self._builder.build()


def _read_file(file: str) -> str:
    """
    Read content of a file

    :param file: Path to file
    :return: File content
    """
    with open(file, 'r') as f:
        return f.read()
