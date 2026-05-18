"""
Noesis client — for external agents (Claude Code, etc.) to interact with
the Noesis consciousness simulation server.

Usage:
    from client import N

    # Run a single experiment cycle
    result = N.run("A cat is sitting on a mat. The cat looks hungry.")
    print(result["narrative"])
    print(result["profile"])  # consciousness profile

    # Run with different modes
    result_comp = N.run("Some stimulus", mode="competitive")
    result_rand = N.run("Some stimulus", mode="random")
    result_none = N.run("Some stimulus", mode="no_broadcast")

    # Run full comparison experiment
    comparison = N.compare(
        stimuli=["stim1", "stim2", "stim3"],
        modes=["competitive", "random", "no_broadcast"],
    )
    print(comparison["analysis"])  # hypothesis test results

    # Check system state
    print(N.status())
"""

import requests

BASE = "http://localhost:7860"


def _post(path, data, timeout=300):
    r = requests.post(f"{BASE}{path}", json=data, timeout=timeout)
    return r.json()


def _get(path, timeout=10):
    r = requests.get(f"{BASE}{path}", timeout=timeout)
    return r.json()


class _N:
    """Noesis client — N for Noesis."""

    def run(self, stimulus: str, mode: str = "competitive") -> dict:
        """Run one GWT+IIT cycle with the given stimulus."""
        return _post("/experiment/run", {"stimulus": stimulus, "mode": mode})

    def batch(self, stimuli: list[str], mode: str = "competitive",
              cycles_per_stimulus: int = 1) -> dict:
        """Run multiple stimuli in sequence."""
        return _post("/experiment/batch", {
            "stimuli": stimuli,
            "mode": mode,
            "cycles_per_stimulus": cycles_per_stimulus,
        })

    def compare(self, stimuli: list[str],
                modes: list[str] = None,
                cycles_per_stimulus: int = 2) -> dict:
        """
        Run full comparison experiment — same stimuli across all modes.

        Returns statistical analysis with hypothesis tests.
        """
        if modes is None:
            modes = ["competitive", "random", "no_broadcast"]
        return _post("/experiment/compare", {
            "stimuli": stimuli,
            "modes": modes,
            "cycles_per_stimulus": cycles_per_stimulus,
        })

    def reset(self) -> dict:
        """Reset experiment state."""
        return _post("/experiment/reset", {})

    def status(self) -> dict:
        """Get current experiment status, Φ trace, and consciousness profile."""
        return _get("/status")

    def profile(self) -> dict:
        """Get full consciousness profile trace."""
        return _get("/profile")

    def memory(self) -> dict:
        """View semantic memory contents."""
        return _get("/memory")


N = _N()
