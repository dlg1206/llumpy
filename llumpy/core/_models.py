"""
File: _models.py

Description: Datastructures for messages to send to LLMs

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

    def to_dicts(self) -> List[Dict[str, str]]:
        """
        :return: Messages as dicts for API
        """
        return [m.to_dict() for m in self.messages]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)
