"""
Minimal JARVES client for Claude.
Usage:
    from j import J
    J.ask("写一个排序函数")
    J.run("ls ~/Desktop")
    J.read("~/Desktop/jarves/jarves.py")
    J.summarize("~/Desktop/jarves/jarves.py", focus="endpoints")
    J.note("key", "value")
    J.batch([("run","ls ~"), ("ask","当前时间")])
    J.clear()
"""
import requests, json

BASE = "http://localhost:7860"

def _post(path, data):
    r = requests.post(f"{BASE}{path}", json=data, timeout=120)
    return r.json()

class _J:
    def ask(self, task, max_tokens=600):
        d = _post("/ask", {"task": task, "from": "claude", "max_tokens": max_tokens})
        return d.get("result") or d.get("error")

    def run(self, cmd):
        d = _post("/run", {"cmd": cmd})
        return d.get("output") or d.get("error")

    def read(self, path, limit=8000):
        d = _post("/read", {"path": path, "limit": limit})
        return d.get("content") or d.get("error")

    def summarize(self, path_or_text, focus="key logic and structure"):
        key = "path" if "/" in path_or_text or "~" in path_or_text else "text"
        d = _post("/summarize", {key: path_or_text, "focus": focus})
        return d.get("summary") or d.get("error")

    def note(self, key, value):
        d = _post("/note", {"key": key, "value": value})
        return d.get("saved") or d.get("error")

    def batch(self, tasks):
        """tasks: list of (type, payload) tuples
        type='run' → payload=cmd string
        type='read' → payload=path string
        type='ask' → payload=task string
        """
        items = []
        for typ, payload in tasks:
            if typ == "run":   items.append({"type": "run",  "cmd": payload})
            elif typ == "read": items.append({"type": "read", "path": payload})
            else:               items.append({"type": "ask",  "task": payload})
        d = _post("/batch", {"tasks": items})
        return [r.get("result") or r.get("error") for r in d.get("results", [])]

    def clear(self):
        d = requests.post(f"{BASE}/memory/clear", timeout=10).json()
        return d.get("cleared")

    def status(self):
        return requests.get(f"{BASE}/status", timeout=5).json()

J = _J()
