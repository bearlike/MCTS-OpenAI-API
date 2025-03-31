#!/usr/bin/env python3
# Pydantic Models for Chat Completion API
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional


class ReasoningEffort(Enum):
    NORMAL = "normal"
    MEDIUM = "medium"
    HIGH = "high"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str  # e.g.: "gpt-4o-mini"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    reasoning_effort: Optional[ReasoningEffort] = ReasoningEffort.NORMAL
