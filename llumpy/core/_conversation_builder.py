"""
File: _conversation_builder.py

Description: Builder to ensure conversations to LLMs are in the correct order

@author Derek Garcia
"""
from typing import List, Literal

from ._args import _validate_args
from ._models import Role, Message, Conversation


class _BuildMixIn:
    """Mixin that allows a step to build a conversation"""
    _messages: List[Message]  # assert self._messages will be available

    def build(self) -> Conversation:
        """
        :return: Immutable conversation
        """
        return Conversation(self._messages)


class _MessagesHolder:
    """Holds the message list and shares the validate and append logic"""

    def __init__(self, messages: List[Message] | None = None):
        """
        Create a new container for messages

        :param messages: List of messages to append to (Default: None)
        """
        self._messages: List[Message] = messages if messages is not None else []

    def _add(self, role: Role, content: str | None, file: str | None) -> List[Message]:
        """
        Validate args, append the message, and return the message list

        :param role: Role of the message
        :param content: Content of system message (Default: None)
        :param file: File to read content from (Default: None)
        :return: List with updated method
        """
        validated = _validate_args(content, file)
        self._messages.append(Message(role, validated))
        return self._messages


class _UserStep(_MessagesHolder):
    """First user turn - prevent building with just a system prompt"""

    def user(self, content: str | None = None, *, file: str | None = None) -> "_AssistantOrBuildStep":
        """
        Add a user message

        :param content: Content of system message (Default: None)
        :param file: File to read content from (Default: None)
        :return: Assistant or build step
        """
        return _AssistantOrBuildStep(self._add(Role.USER, content, file))

    def build_with_user(self, content: str | None = None, *, file: str | None = None) -> Conversation:
        """
        Build with a temporary user message

        :param content: Content of system message (Default: None)
        :param file: File to read content from (Default: None)
        :return: List of messages
        """
        return _build_with(self._messages, Role.USER, content, file)


class _UserOrBuildStep(_UserStep, _BuildMixIn):
    """User turn in the conversation - allow build"""


class _AssistantOrBuildStep(_MessagesHolder, _BuildMixIn):
    """Assistant or build step in the conversation"""

    def assistant(self, content: str | None = None, *, file: str | None = None) -> "_UserOrBuildStep":
        """
        Add an assistant message

        :param content: Content of system message (Default: None)
        :param file: File to read content from (Default: None)
        :return: User step
        """
        return _UserOrBuildStep(self._add(Role.ASSISTANT, content, file))

    def build_with_assistant(self, content: str | None = None, *, file: str | None = None) -> Conversation:
        """
        Build with a temporary assistant message

        :param content: Content of system message (Default: None)
        :param file: File to read content from (Default: None)
        :return: List of messages
        """
        return _build_with(self._messages, Role.ASSISTANT, content, file)


class ConversationBuilder(_MessagesHolder):
    """Conversation builder to ensure messages are in a valid order"""

    def system(self, content: str | None = None, *, file: str | None = None) -> _UserStep:
        """
        Init conversation with a system message

        :param content: Content of system message (Default: None)
        :param file: File to read content from (Default: None)
        :raises ValueError: If nether or both content or file provided
        :return: User step
        """
        return _UserStep(self._add(Role.SYSTEM, content, file))

    def user(self, content: str | None = None, *, file: str | None = None) -> _AssistantOrBuildStep:
        """
        Init conversation with a user message

        :param content: Content of user message (Default: None)
        :param file: File to read content from (Default: None)
        :return: Assistant step
        """
        return _AssistantOrBuildStep(self._add(Role.USER, content, file))


def _build_with(messages: List[Message],
                role: Literal[Role.USER, Role.ASSISTANT],
                content: str | None = None,
                file: str | None = None) -> Conversation:
    """
    Build a conversation without appending the last message to the current conversation

    :param messages: Messages in the current conversation
    :param role: Either temporary user or assistant role to the conversation
    :param content: Content of system message (Default: None)
    :param file: File to read content from (Default: None)
    :return: Immutable conversation
    """
    content = _validate_args(content, file)
    return Conversation(messages + [Message(role, content)])
