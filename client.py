"""
Noesis client — for external agents (Claude Code, etc.) to interact with
the Noesis consciousness simulation server.

Usage:
    from client import N

    # Run a single experiment cycle
    result = N.run("A cat is sitting on a mat. The cat looks hungry.")
    print(result["narrative"])  # first-person report

    # Run a batch
    results = N.batch(["stimulus 1", "stimulus 2", "stimulus 3"])

    # Check system state
    print(N.status())
"""

import requests

BASE = "http://localhost:7860"


def _post(path, data, timeout=180):
    r = requests.post(f"{BASE}{path}", json=data, timeout=timeout)
    return r.json()


class _N:
    """Noesis client — N for Noesis."""

    def run(self, stimulus: str) -> dict:
        """Run one GWT+IIT cycle with the given stimulus."""
        return _post("/experiment/run", {"stimulus": stimulus})

    def batch(self, stimuli: list[str]) -> dict:
        """Run multiple stimuli in sequence."""
        return _post("/experiment/batch", {"stimuli": stimuli})

    def reset(self) -> dict:
        """Reset experiment state."""
        return _post("/experiment/reset", {})

    def status(self) -> dict:
        """Get current experiment status and Φ trace."""
        return requests.get(f"{BASE}/status", timeout=5).json()

    def memory(self) -> dict:
        """View semantic memory contents."""
        return requests.get(f"{BASE}/memory", timeout=5).json()


N = _N()
