"""Optional torch.compile Inductor (Triton-backed kernels on CUDA) for RuGPT-3 XL."""

from typing import Optional

import torch
from torch import nn


def triton_runtime_available() -> bool:
    """True if CUDA and the triton package are importable."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def compile_rugpt3xl_for_triton(
    model: nn.Module,
    mode: str = "max-autotune",
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
) -> nn.Module:
    """Apply torch.compile with Inductor backend (generates Triton for many ops).

    Does not change the mathematical definition of the model; only the
    implementation on GPU. Requires PyTorch 2.x with CUDA.
    """
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0+")
    if dynamic is None:
        dynamic = True
    try:
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraph_trees = False
    except Exception:
        pass
    return torch.compile(
        model,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
