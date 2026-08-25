#!/usr/bin/env python3
"""A minimal MCP server (stdio, JSON-RPC) exposing the Claude Code memory
notes as a RAG: rag_search greps the markdown notes, rag_read returns one
note whole. No dependencies beyond the standard library.

Point any MCP host at:  python3 memory-rag-mcp.py [memory_dir]
"""
import json
import os
import re
import sys

MEMORY = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.claude/projects/-Users-themoretheless-Documents-Sources-allpaka/memory"
)

TOOLS = [
    {
        "name": "rag_search",
        "description": "Search the personal engineering knowledge base "
        "(Metal/GPU optimisation notes for the allpaka LLM engine). "
        "Returns note names and matching lines.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "words to look for"}},
            "required": ["query"],
        },
    },
    {
        "name": "rag_read",
        "description": "Read one knowledge-base note in full by its file name "
        "(as returned by rag_search).",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "note file name, e.g. engine-status.md"}},
            "required": ["name"],
        },
    },
]


def notes():
    for f in sorted(os.listdir(MEMORY)):
        if f.endswith(".md"):
            yield f


def search(query):
    words = [w for w in re.split(r"\W+", query.lower()) if w]
    if not words:
        return "empty query"
    out = []
    for f in notes():
        path = os.path.join(MEMORY, f)
        text = open(path, encoding="utf-8", errors="replace").read()
        low = text.lower()
        score = sum(low.count(w) for w in words)
        if score == 0:
            continue
        lines = [
            l.strip()
            for l in text.splitlines()
            if any(w in l.lower() for w in words)
        ][:6]
        out.append((score, f, lines))
    out.sort(reverse=True)
    if not out:
        return "no notes match: " + query
    parts = []
    for score, f, lines in out[:5]:
        parts.append(f"## {f} (hits: {score})\n" + "\n".join("- " + l for l in lines))
    return "\n\n".join(parts)


def read(name):
    base = os.path.basename(name)
    path = os.path.join(MEMORY, base)
    if not os.path.isfile(path):
        return f"no such note: {base}. Available: " + ", ".join(notes())
    return open(path, encoding="utf-8", errors="replace").read()


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        msg["error"] = {"code": -32000, "message": error}
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method", "")
        id_ = req.get("id")
        if method == "initialize":
            reply(id_, {
                "protocolVersion": req["params"].get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "memory-rag", "version": "0.1.0"},
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            reply(id_, {"tools": TOOLS})
        elif method == "tools/call":
            name = req["params"]["name"]
            args = req["params"].get("arguments", {})
            try:
                if name == "rag_search":
                    text = search(args.get("query", ""))
                elif name == "rag_read":
                    text = read(args.get("name", ""))
                else:
                    reply(id_, error=f"unknown tool {name}")
                    continue
                reply(id_, {"content": [{"type": "text", "text": text}]})
            except Exception as e:  # noqa: BLE001 - surface, not crash
                reply(id_, error=str(e))
        elif id_ is not None:
            reply(id_, error=f"unsupported method {method}")


if __name__ == "__main__":
    main()
