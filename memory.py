"""
Semantic memory with embedding-based retrieval.

Ported from the JARVES memory system. Stores past broadcasts and
retrieves relevant context for agent prompts.

In the GWT framework, this serves as "long-term memory" — the
sediment of past conscious broadcasts that shapes future attention.
"""

import json
import math
import threading
from pathlib import Path

import requests as req

MEMORY_FILE = Path(__file__).parent / "memory.json"
EMBED_FILE = Path(__file__).parent / "embeddings.json"
EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


class SemanticMemory:
    def __init__(self):
        self.notes: dict = {}
        self.history: list = []
        self.embeddings: list = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                d = json.loads(MEMORY_FILE.read_text())
                self.notes = d.get("notes", {})
                self.history = d.get("history", [])
            except Exception:
                pass
        if EMBED_FILE.exists():
            try:
                self.embeddings = json.loads(EMBED_FILE.read_text())
            except Exception:
                pass

    def _save(self):
        MEMORY_FILE.write_text(json.dumps(
            {"notes": self.notes, "history": self.history[-100:]},
            ensure_ascii=False, indent=2,
        ))

    def _save_embeddings(self):
        EMBED_FILE.write_text(json.dumps(self.embeddings[-100:], ensure_ascii=False))

    def _embed(self, text: str) -> list:
        try:
            r = req.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=10)
            return r.json().get("embedding", [])
        except Exception:
            return []

    def _cosine(self, a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def store(self, key: str, content: str):
        with self._lock:
            self._load()
            self.history.append({"key": key, "content": content[:200]})
            self._save()

        def _do_embed():
            emb = self._embed(content)
            if emb:
                with self._lock:
                    self.embeddings.append({
                        "key": key, "content": content[:200], "emb": emb,
                    })
                    self._save_embeddings()

        threading.Thread(target=_do_embed, daemon=True).start()

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k semantically similar memories."""
        if not self.embeddings:
            return []
        q_emb = self._embed(text)
        if not q_emb:
            return self.history[-top_k:]
        scored = sorted(
            self.embeddings,
            key=lambda e: self._cosine(q_emb, e["emb"]),
            reverse=True,
        )
        return [{"key": e["key"], "content": e["content"]} for e in scored[:top_k]]

    def context(self, query_text: str = "") -> str:
        """Build a context string of relevant memories for agent prompts."""
        if query_text:
            memories = self.query(query_text)
            if memories:
                return "Relevant memories:\n" + "\n".join(
                    f"  [{m['key']}] {m['content'][:100]}" for m in memories
                )
        if self.history:
            recent = self.history[-3:]
            return "Recent:\n" + "\n".join(
                f"  [{h['key']}] {h['content'][:100]}" for h in recent
            )
        return ""

    def clear(self):
        with self._lock:
            self.history = []
            self.embeddings = []
            self._save()
            if EMBED_FILE.exists():
                EMBED_FILE.unlink()

    def summary(self) -> dict:
        return {
            "notes": len(self.notes),
            "history_entries": len(self.history),
            "embeddings": len(self.embeddings),
        }
