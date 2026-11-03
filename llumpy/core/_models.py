"""
File: _models.py

Description: Datastructures for messages to send to LLMs

@author Derek Garcia
"""
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """Roles for LLM conversation"""
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Message:
    """DTO for LLM messages"""
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        """
        :return: DTO as dict
        """
        return {'role': self.role.value, 'content': self.content}


@dataclass(frozen=True)
class History:
    """Immutable history with LLM"""
    messages: list[Message]

    def to_dicts(self) -> list[dict[str, str]]:
        """
        :return: Messages as dicts for API
        """
        return [m.to_dict() for m in self.messages]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)


@dataclass(frozen=True)
class Conversation(History):
    """Immutable conversation for LLM"""

    def __post_init__(self):
        """
        Validate the conversation order is correct

        :raises ValueError: If no messages or in invalid order
        """
        msgs = self.messages
        if not msgs:
            raise ValueError("Conversation must contain at least one message")

        # base ::= system_p | user_p
        # system_p ::= <system> user_p   (system only allowed first)
        start = 1 if msgs[0].role == Role.SYSTEM else 0

        if start >= len(msgs):
            raise ValueError("A system message must be followed by a user message")

        # user_p / assistant_p: strictly alternate user, assistant, user, ...
        for i, msg in enumerate(msgs[start:], start=start):
            expected = Role.USER if (i - start) % 2 == 0 else Role.ASSISTANT
            if msg.role != expected:
                raise ValueError(
                    f"Invalid role at position {i}: expected '{expected}', got '{msg.role}'"
                )
