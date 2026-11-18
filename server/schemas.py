"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    # content may be null for assistant tool-call turns.
    content: Union[str, List[dict], None] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # DeepSeek web-chat options.
    thinking: bool = False
    search: bool = False
    # OpenAI-compatible prompt-emulated function tools. They are returned to the
    # caller as tool_calls; this server does not execute them.
    tools: Optional[List[dict]] = None
    tool_choice: Any = "auto"
    # Accepted for compatibility but not all are forwarded to DeepSeek.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
