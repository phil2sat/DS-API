"""Prompt-based adapter for OpenAI function tools.

DeepSeek's web chat has no native function-call channel, so this module teaches
it a compact wire format and converts only explicit model output into OpenAI
``tool_calls``. It never executes a tool; execution remains the client's job.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from typing import Any

MAX_TOOL_OUTPUT_CHARS = 256 * 1024


def _function(tool: dict) -> dict | None:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn
    # Also accept Responses-style function tools.
    if tool.get("type") == "function" and isinstance(tool.get("name"), str):
        return tool
    return None


def tool_names(tools: list[dict] | None) -> set[str]:
    return {
        fn["name"] for tool in (tools or [])
        if (fn := _function(tool)) and isinstance(fn.get("name"), str)
    }


def format_tool_instructions(tools: list[dict] | None, tool_choice: Any = "auto") -> str:
    """Serialize the caller's tool schemas as instructions for DeepSeek."""
    if not tools:
        return ""
    available = []
    for tool in tools:
        fn = _function(tool)
        if not fn or not isinstance(fn.get("name"), str):
            continue
        spec = {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        available.append(spec)
    if not available:
        return ""

    choice_text = "Use a tool when it is useful for the user's request; otherwise answer normally."
    if tool_choice == "required":
        choice_text = "You MUST call one of the available tools before answering."
    elif isinstance(tool_choice, dict):
        forced = tool_choice.get("function", {}).get("name") if isinstance(tool_choice.get("function"), dict) else None
        if forced:
            choice_text = f'You MUST call the tool named "{forced}".'
    elif tool_choice == "none":
        return ""

    return (
        "\n\n[TOOL CALLING INSTRUCTIONS]\n"
        "You can request tools, but do not claim to execute them yourself. "
        f"{choice_text}\n"
        "When calling a tool, output ONLY one JSON object in this form, with no markdown or explanation:\n"
        '{"tool_call":{"name":"function_name","arguments":{"key":"value"}}}\n'
        "The gateway will return the actual tool result in a following message. "
        "Never invent or simulate tool results.\n"
        "Available tools (JSON Schema):\n"
        f"{json.dumps(available, ensure_ascii=False, separators=(',', ':'))}\n"
        "[END TOOL CALLING INSTRUCTIONS]"
    )


def _balanced_objects(text: str):
    """Yield top-level JSON object candidates without regex brace guessing."""
    start = depth = 0
    in_string = escaped = False
    for i, char in enumerate(text):
        if not depth:
            if char == "{":
                start, depth, in_string, escaped = i, 1, False, False
            continue
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string and char == "{":
            depth += 1
        elif not in_string and char == "}":
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def _to_call(candidate: Any, allowed: set[str]) -> dict | None:
    if not isinstance(candidate, dict):
        return None
    fn = candidate.get("function") if isinstance(candidate.get("function"), dict) else candidate
    name = fn.get("name")
    args = fn.get("arguments", fn.get("input", candidate.get("arguments", {})))
    if not isinstance(name, str) or name not in allowed:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    try:
        encoded = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    if len(encoded) > MAX_TOOL_OUTPUT_CHARS:
        return None
    return {
        "id": "call_" + uuid.uuid4().hex,
        "type": "function",
        "function": {"name": name, "arguments": encoded},
    }


def _candidate_from_json(obj: Any, allowed: set[str]) -> list[dict] | None:
    if not isinstance(obj, dict):
        return None
    candidate = obj.get("tool_call", obj.get("function_call"))
    if candidate is not None:
        call = _to_call(candidate, allowed)
        return [call] if call else None
    candidates = obj.get("tool_calls")
    if isinstance(candidates, list) and candidates:
        calls = [_to_call(item, allowed) for item in candidates]
        return calls if all(calls) else None
    return None


def _parse_dsml(text: str, allowed: set[str]) -> list[dict] | None:
    # DeepSeek emits both <｜DSML｜...> and ordinary XML-like variants.
    text = text.translate(str.maketrans({"＜": "<", "＞": ">", "｜": "|"}))
    text = re.sub(r"<\|+\s*DSML\s*\|+\s*", "<", text, flags=re.I)
    text = re.sub(r"</\|+\s*DSML\s*\|+\s*", "</", text, flags=re.I)
    text = re.sub(r"<\|+\|+\s*DSML\s*\|+\|+\s*", "<", text, flags=re.I)
    wrapper = re.search(r"<tool_calls\b[^>]*>(.*?)</tool_calls\s*>", text, flags=re.I | re.S)
    scope = wrapper.group(1) if wrapper else text
    invokes = list(re.finditer(r"<invoke\b([^>]*)>(.*?)</invoke\s*>", scope, flags=re.I | re.S))
    if len(invokes) != 1 or scope[:invokes[0].start()].strip() or scope[invokes[0].end():].strip():
        return None
    attrs, body = invokes[0].groups()
    name_match = re.search(r"\bname\s*=\s*([\"'])(.*?)\1", attrs, flags=re.I | re.S)
    if not name_match:
        return None
    args = {}
    params = list(re.finditer(r"<parameter\b([^>]*)>(.*?)</parameter\s*>", body, flags=re.I | re.S))
    cursor = 0
    for match in params:
        if body[cursor:match.start()].strip():
            return None
        p_name = re.search(r"\bname\s*=\s*([\"'])(.*?)\1", match.group(1), flags=re.I | re.S)
        if not p_name or p_name.group(2) in args:
            return None
        value = html.unescape(match.group(2).strip())
        is_string = re.search(r"\bstring\s*=\s*([\"'])false\1", match.group(1), flags=re.I)
        if is_string:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        args[p_name.group(2)] = value
        cursor = match.end()
    if body[cursor:].strip():
        return None
    call = _to_call({"name": name_match.group(2), "arguments": args}, allowed)
    return [call] if call else None


def _schema_props(tools: list[dict] | None) -> dict[str, dict]:
    out = {}
    for tool in tools or []:
        fn = _function(tool)
        if fn and isinstance(fn.get("name"), str):
            params = fn.get("parameters")
            props = params.get("properties") if isinstance(params, dict) else None
            out[fn["name"]] = props if isinstance(props, dict) else {}
    return out


def _coerce(value: str, spec: Any) -> Any:
    """Convert an XML attribute/text value to the type its JSON Schema asks for."""
    kind = spec.get("type") if isinstance(spec, dict) else None
    if kind in ("integer", "number", "boolean", "array", "object"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


_TAG_ATTR = re.compile(r"([A-Za-z_][\w.:-]*)\s*=\s*(\"(.*?)\"|'(.*?)')", re.S)


def _parse_xml_tags(text: str, tools: list[dict] | None) -> list[dict] | None:
    """Fallback for models that improvise calls as ``<read path="x"/>`` or
    ``<tool><arg>value</arg></tool>``. Only accepted when the whole reply (minus
    an optional code fence) consists of tags naming known tools."""
    props = _schema_props(tools)
    if not props:
        return None
    body = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())
    names = "|".join(re.escape(n) for n in sorted(props, key=len, reverse=True))
    pattern = re.compile(
        rf"<(?P<name>{names})\b(?P<attrs>[^<>]*?)(?:/>|>(?P<inner>.*?)</(?P=name)\s*>)",
        re.S,
    )
    calls, cursor = [], 0
    for m in pattern.finditer(body):
        if body[cursor:m.start()].strip():
            return None
        name, spec = m.group("name"), props[m.group("name")]
        args = {
            a.group(1): _coerce(html.unescape(a.group(3) if a.group(3) is not None else a.group(4)),
                                spec.get(a.group(1)))
            for a in _TAG_ATTR.finditer(m.group("attrs"))
        }
        inner = m.group("inner")
        if inner and inner.strip():
            children = list(re.finditer(r"<([A-Za-z_][\w.:-]*)>(.*?)</\1\s*>", inner, re.S))
            rest = re.sub(r"<([A-Za-z_][\w.:-]*)>(.*?)</\1\s*>", "", inner, flags=re.S).strip()
            if children and not rest:
                for c in children:
                    args[c.group(1)] = _coerce(html.unescape(c.group(2).strip()), spec.get(c.group(1)))
            else:
                return None
        call = _to_call({"name": name, "arguments": args}, set(props))
        if not call:
            return None
        calls.append(call)
        cursor = m.end()
    if not calls or body[cursor:].strip():
        return None
    return calls


def parse_tool_calls(text: str, tools: list[dict] | None) -> list[dict] | None:
    """Parse strict JSON, legacy TOOL_CALL, fenced JSON, DeepSeek DSML and bare XML tags."""
    if not isinstance(text, str) or len(text) > MAX_TOOL_OUTPUT_CHARS:
        return None
    allowed = tool_names(tools)
    if not allowed:
        return None

    dsml = _parse_dsml(text, allowed)
    if dsml:
        return dsml

    legacy = re.search(r"TOOL_CALL:\s*([A-Za-z0-9_.:-]+)", text, flags=re.I)
    if legacy:
        tail = text[legacy.end():]
        for raw in _balanced_objects(tail):
            try:
                call = _to_call({"name": legacy.group(1), "arguments": json.loads(raw)}, allowed)
                if call:
                    return [call]
            except json.JSONDecodeError:
                continue

    for raw in _balanced_objects(text):
        try:
            calls = _candidate_from_json(json.loads(raw), allowed)
            if calls:
                return calls
        except json.JSONDecodeError:
            continue
    return _parse_xml_tags(text, tools)
