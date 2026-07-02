"""Centralised PyTorch device selection and compilation strategy.

Import both ``device`` and ``paradis_compile`` from here so that hardware
detection and compilation policy are resolved once at import time.

Compilation backends
--------------------
``paradis_compile`` is the package-wide decorator applied to all hot tensor
functions (dispersal kernel, matrix solves, …).  It selects the best
available backend automatically:

``inductor`` (full ``torch.compile`` default)
    TorchInductor generates SIMD-vectorised native code by lowering the
    captured computation graph to C++ / OpenMP.  Fastest option, but
    requires a C++ compiler: MSVC (``cl.exe``) on Windows, gcc or clang
    on Linux/macOS.

``aot_eager`` (portable fallback)
    Ahead-of-Time Autograd traces and freezes the computation graph using
    PyTorch's own dispatcher — no external compiler is invoked.  Subsequent
    calls bypass the Python interpreter entirely and run through the
    pre-traced graph, giving real (though more modest) acceleration.
    Used automatically when no C++ compiler is found.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C++ compiler detection
# ---------------------------------------------------------------------------

def _compiler_available() -> bool:
    """Return ``True`` if TorchInductor can locate a usable C++ compiler."""
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler
        compiler = get_cpp_compiler()
        if sys.platform == "win32":
            from torch._inductor.cpp_builder import check_compiler_exist_windows
            check_compiler_exist_windows(compiler)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Compilation strategy
# ---------------------------------------------------------------------------

def _triton_available() -> bool:
    """Return ``True`` if a working Triton installation is found."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def _select_backend() -> str:
    """Return the name of the best available torch.compile backend."""
    if torch.cuda.is_available():
        # GPU path: inductor generates CUDA kernels via Triton.
        # Triton is not distributed for Windows on PyPI, so fall back to
        # aot_eager (graph-traced, no Triton required) when it is absent.
        if _triton_available():
            return "inductor"
        return "aot_eager"
    if _compiler_available():
        # CPU + C++ compiler: inductor generates AVX / AVX-512 vectorised code.
        return "inductor"
    # CPU without a compiler: aot_eager captures the graph and eliminates
    # Python overhead without emitting any native code.
    return "aot_eager"


_BACKEND = _select_backend()

if _BACKEND == "aot_eager":
    if torch.cuda.is_available():
        logger.info(
            "paradis: Triton not found — torch.compile will use the 'aot_eager' "
            "backend on GPU (graph tracing, no Triton kernels).  Triton is not "
            "currently distributed for Windows via PyPI; GPU tensors will still "
            "run on CUDA, just without kernel fusion."
        )
    else:
        logger.info(
            "paradis: no C++ compiler found — torch.compile will use the 'aot_eager' "
            "backend (graph tracing, no native-code generation).  Install a C++ "
            "compiler (e.g. MSVC Build Tools on Windows) for maximum CPU performance."
        )


def paradis_compile(fn: Callable | None = None, **kwargs) -> Callable:
    """Decorate *fn* with the best available ``torch.compile`` backend.

    Can be used as a plain decorator or with keyword arguments::

        @paradis_compile
        def my_fn(x): ...

        @paradis_compile(dynamic=True)
        def my_fn(x): ...

    The backend is chosen once at package import time (see module docstring).
    """
    kwargs.setdefault("backend", _BACKEND)
    if fn is None:                          # called with arguments: @paradis_compile(...)
        return lambda f: torch.compile(f, **kwargs)
    return torch.compile(fn, **kwargs)      # called bare: @paradis_compile


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
