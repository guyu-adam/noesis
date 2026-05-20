"""
Hardware-aware configuration for neural experiments.

Scales automatically based on detected hardware (GPU VRAM, RAM).
Override via environment variables or direct import.
"""

import os

# ── GPU detection ──────────────────────────────────────────────────

try:
    import cupy as cp
    _XP = cp
    _GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # fallback: use numpy on CPU
    _GPU_AVAILABLE = False


def get_backend():
    """Return the array backend (cupy if GPU available, else numpy)."""
    return _XP


def is_gpu_available() -> bool:
    return _GPU_AVAILABLE


def get_gpu_info() -> dict:
    if not _GPU_AVAILABLE:
        return {"gpu": False}
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "gpu": True,
            "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "vram_mb": props["totalGlobalMem"] // (1024 * 1024),
        }
    except Exception:
        return {"gpu": True, "name": "unknown", "vram_mb": 0}


# ── Scale factors based on hardware ─────────────────────────────────

_VRAM_GB = get_gpu_info().get("vram_mb", 0) / 1024 if _GPU_AVAILABLE else 0
_RAM_GB = 32  # conservative default


def _default_neurons() -> int:
    """Scale neuron count based on GPU VRAM."""
    vram = os.environ.get("NOESIS_N_NEURONS")
    if vram:
        return int(vram)
    if _VRAM_GB >= 14:     # RTX 5060 Ti 16GB, RTX 4090 24GB, etc.
        return 512
    if _VRAM_GB >= 10:     # RTX 3080 12GB, RTX 4070 12GB, etc.
        return 384
    if _VRAM_GB >= 6:      # GTX 1660 Ti 6GB, etc.
        return 256
    if _VRAM_GB >= 4:
        return 128
    return 64


def _default_processors() -> int:
    return int(os.environ.get("NOESIS_N_PROCESSORS",
                              os.environ.get("NOESIS_N_AGENTS", "5")))


def _default_unroll() -> int:
    return int(os.environ.get("NOESIS_N_UNROLL", "25"))


def _default_batch() -> int:
    """Default batch size — how many cycles to parallelize."""
    vram = os.environ.get("NOESIS_BATCH_SIZE")
    if vram:
        return int(vram)
    if _VRAM_GB >= 14:
        return 8
    if _VRAM_GB >= 10:
        return 6
    if _VRAM_GB >= 6:
        return 4
    return 2


def _default_clusters() -> int:
    """Default number of state clusters for TPM discretization."""
    vram = os.environ.get("NOESIS_N_CLUSTERS")
    if vram:
        return int(vram)
    if _VRAM_GB >= 14:
        return 50
    if _VRAM_GB >= 10:
        return 40
    if _VRAM_GB >= 6:
        return 30
    return 20


# ── Experiment configuration ────────────────────────────────────────


class Config:
    """Hardware-aware experiment configuration."""

    def __init__(self):
        self.n_neurons: int = _default_neurons()
        self.n_input: int = int(os.environ.get("NOESIS_N_INPUT", "32"))
        self.n_unroll: int = _default_unroll()
        self.n_processors: int = _default_processors()
        self.batch_size: int = _default_batch()
        self.noise_std: float = float(os.environ.get("NOESIS_NOISE", "0.01"))
        self.max_history: int = int(os.environ.get("NOESIS_MAX_HISTORY", "500"))
        self.n_state_clusters: int = int(os.environ.get(
            "NOESIS_N_CLUSTERS",
            str(_default_clusters())
        ))
        self.dtype: str = "float32"
        self.gpu: bool = _GPU_AVAILABLE

        # Derived
        self.total_neurons = self.n_neurons * self.n_processors
        self.weights_per_processor = self.n_neurons * self.n_neurons + self.n_neurons * self.n_input

    def summary(self) -> dict:
        return {
            "gpu": get_gpu_info(),
            "n_neurons": self.n_neurons,
            "n_input": self.n_input,
            "n_unroll": self.n_unroll,
            "n_processors": self.n_processors,
            "batch_size": self.batch_size,
            "total_neurons": self.total_neurons,
            "weights_per_processor": self.weights_per_processor,
            "dtype": self.dtype,
        }

    def print(self):
        info = self.summary()
        gpu = info["gpu"]
        print(f"=== Noesis Config ===")
        if gpu.get("gpu"):
            print(f"GPU: {gpu.get('name', '?')} ({gpu.get('vram_mb', 0)}MB VRAM)")
        else:
            print(f"GPU: none (CPU-only)")
        print(f"Neurons: {info['n_neurons']}/processor × {info['n_processors']} processors = {info['total_neurons']} total")
        print(f"Input dim: {info['n_input']}, unroll steps: {info['n_unroll']}")
        print(f"Batch size: {info['batch_size']}")
        print(f"FP precision: {info['dtype']}")
        print(f"====================")


# Module-level default config
_default_config = None


def get_config() -> Config:
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config
