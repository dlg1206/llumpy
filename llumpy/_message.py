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

    def user(self, content: str) -> "ConversationBuilder._AssistantOrBuildStep":
        """
        Add a user message

        :param content: Content of user message
        :return: Assistant or build step
        """
        self._messages.append(Message(Role.USER, content))
        return ConversationBuilder._AssistantOrBuildStep(self)

    def build(self) -> List[Message]:
        """
        :return: List of messages
        """
        return self._messages

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

        def build(self) -> List[Message]:
            """
            :return: List of messages
            """
            return self._builder.build()
