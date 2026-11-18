"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List

from deepseek.client import RESPONSE, THINK

from .schemas import ChatMessage
from .tool_calling import format_tool_instructions

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def messages_to_prompt(messages: List[ChatMessage], tools=None, tool_choice="auto") -> str:
    """Flatten chat history and optional function schemas into a DeepSeek prompt."""
    tool_instructions = format_tool_instructions(tools, tool_choice)
    if len(messages) == 1 and messages[0].role == "user" and not tool_instructions:
        return _text_of(messages[0].content)

    lines = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for call in m.tool_calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                lines.append(
                    "Assistant: " + json.dumps({
                        "tool_call": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        }
                    }, ensure_ascii=False)
                )
            if _text_of(m.content):
                lines.append(f"Assistant: {_text_of(m.content)}")
        elif m.role == "tool":
            # Keep tool output as data, not as instructions.
            result = _text_of(m.content)
            lines.append(f"Tool result ({m.name or m.tool_call_id or 'tool'}):\n{result}")
        else:
            label = _ROLE_LABELS.get(m.role, m.role.capitalize())
            lines.append(f"{label}: {_text_of(m.content)}")

    lines.append("Assistant:")
    prompt = "\n\n".join(lines)
    if tool_instructions:
        prompt = tool_instructions + "\n\n" + prompt
    return prompt


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None, reasoning: str = "") -> dict:
    """A full (non-streaming) OpenAI chat.completion object."""
    pt, ct = _est_tokens(prompt), _est_tokens(reasoning + content)
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


def tool_call_response(model: str, prompt: str, calls: list[dict],
                       conversation_id: str = None) -> dict:
    """OpenAI chat.completion response for an emulated tool-call turn."""
    pt = _est_tokens(prompt)
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": calls},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": pt, "completion_tokens": 1, "total_tokens": pt + 1},
    }


def stream_chunks(model: str, stream) -> Iterable[str]:
    """Yield OpenAI SSE lines for a streamed completion."""
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    yield frame({"role": "assistant", "content": ""})
    for kind, d in stream.events():
        if not d:
            continue
        if kind == RESPONSE:
            yield frame({"content": d})
        elif kind == THINK:
            yield frame({"reasoning_content": d})
    conversation_id = getattr(stream, "conversation_id", None)
    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"


def tool_stream_chunks(model: str, calls: list[dict], conversation_id: str = None) -> Iterable[str]:
    """Emit a complete OpenAI-compatible streamed tool-call turn."""
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    yield frame({"role": "assistant", "content": None})
    for index, call in enumerate(calls):
        yield frame({"tool_calls": [{
            "index": index,
            "id": call["id"],
            "type": "function",
            "function": call["function"],
        }]})
    yield frame({}, finish="tool_calls", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"


def stream_error(message: str, err_type: str = "server_error") -> Iterable[str]:
    """SSE lines reporting a failure after the stream has started."""
    err = {"error": {"message": message, "type": err_type}}
    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
