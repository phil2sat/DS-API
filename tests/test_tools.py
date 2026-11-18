import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server import api
from server.tool_calling import parse_tool_calls

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "read file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "execute_command", "description": "run",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
]


@pytest.mark.parametrize("text,name,args", [
    ('<read path="/a/package.json"/>', "read", {"path": "/a/package.json"}),
    ('<read path="/a" limit="50"/>', "read", {"path": "/a", "limit": 50}),
    ('<execute_command><command>ls -la</command></execute_command>', "execute_command", {"command": "ls -la"}),
    ('{"tool_call":{"name":"read","arguments":{"path":"x"}}}', "read", {"path": "x"}),
    ('TOOL_CALL: read {"path":"x"}', "read", {"path": "x"}),
    ('```json\n{"tool_call":{"name":"read","arguments":{"path":"x"}}}\n```', "read", {"path": "x"}),
    ('<｜DSML｜tool_calls><｜DSML｜invoke name="read"><｜DSML｜parameter name="path" string="true">x</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>',
     "read", {"path": "x"}),
])
def test_parse(text, name, args):
    calls = parse_tool_calls(text, TOOLS)
    assert calls and calls[0]["function"]["name"] == name
    assert json.loads(calls[0]["function"]["arguments"]) == args


@pytest.mark.parametrize("text", [
    "Ich würde <read path='x'/> benutzen, um die Datei zu lesen.",
    '<unknown path="x"/>',
    "Hallo!",
])
def test_plain_text_not_parsed(text):
    assert parse_tool_calls(text, TOOLS) is None


class FakeClient:
    def __init__(self, text):
        self.text, self.prompt = text, None

    def chat(self, prompt, cid, model_type, thinking, search):
        self.prompt = prompt
        return SimpleNamespace(text=self.text, conversation_id="c:1", reasoning="")


def _post(monkeypatch, text, stream):
    fake = FakeClient(text)
    monkeypatch.setattr(api, "get_client", lambda: fake)
    r = TestClient(api.app).post("/v1/chat/completions", json={
        "model": "deepseek-chat", "stream": stream, "tools": TOOLS,
        "messages": [{"role": "user", "content": "lies package.json"}]})
    return fake, r


def test_nonstream_tool_call(monkeypatch):
    fake, r = _post(monkeypatch, '<read path="/p/package.json"/>', False)
    ch = r.json()["choices"][0]
    assert ch["finish_reason"] == "tool_calls"
    assert ch["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert "TOOL CALLING INSTRUCTIONS" in fake.prompt


def test_stream_tool_call(monkeypatch):
    _, r = _post(monkeypatch, '{"tool_call":{"name":"read","arguments":{"path":"x"}}}', True)
    frames = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
    assert frames[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read"
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_plain_text(monkeypatch):
    _, r = _post(monkeypatch, "Hallo!", True)
    frames = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: {")]
    assert any(f["choices"][0]["delta"].get("content") == "Hallo!" for f in frames)
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
