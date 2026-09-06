"""Array backend selection: NumPy on the CPU, CuPy on an NVIDIA GPU.

CuPy implements the NumPy API on CUDA. Almost every call in
`src/scratch/` works unchanged against either module, so the from scratch
network can run on a GPU without giving up the hand written derivatives.

Selection happens once, before `src.scratch` is imported, because those
modules bind the array module at import time:

    from src import backend
    backend.select("cuda")
    from src.scratch.network import MLP

Data still loads on the CPU. `asarray` moves it to the device, `to_numpy`
brings results back for metrics and plotting, which stay on NumPy.
"""

from __future__ import annotations

import numpy as _numpy

xp = _numpy
name = "numpy"
_cupy = None
_locked = False


def available() -> bool:
    """True when CuPy is installed and a GPU answers."""
    try:
        import cupy
    except ImportError:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def select(device: str = "auto") -> str:
    """Choose the array module. Returns "numpy" or "cupy".

    "cuda" is a request and fails loudly if it cannot be honoured, because a
    silent fallback is how a run ends up far slower than expected without
    anyone noticing. "auto" falls back to NumPy and says why.
    """
    global xp, name, _cupy

    device = device.lower()
    if device in ("cpu", "numpy"):
        _set_numpy()
        return name

    if device in ("cuda", "gpu", "cupy"):
        try:
            import cupy
        except ImportError:
            raise SystemExit(
                "--device cuda needs CuPy, which is not installed.\n"
                "  pip install cupy-cuda13x        (CUDA 13, matches torch cu130)\n"
                "  pip install cupy-cuda12x        (CUDA 12)\n"
                "Check your CUDA version with nvidia-smi, then pick the matching wheel.\n"
                "The PyTorch trainer does not need CuPy: python train_torch.py --device cuda"
            ) from None
        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise SystemExit("CuPy is installed but no CUDA device is visible. Run nvidia-smi.")
        _smoke_test(cupy)
        _set_cupy(cupy)
        return name

    if device != "auto":
        raise ValueError(f"unknown device {device!r}, use cpu, cuda or auto")

    if available():
        import cupy

        _set_cupy(cupy)
    else:
        _set_numpy()
        print("CuPy not available, the from scratch network runs on the CPU")
    return name


def _smoke_test(cupy) -> None:
    """Exercise the two libraries a run depends on, before training starts.

    CuPy builds its elementwise kernels at runtime with NVRTC, so it needs
    the CUDA headers rather than only the driver. Matrix products go through
    cuBLAS, which is a separate library loaded separately and can be the only
    broken piece. Both failures otherwise arrive mid epoch as a traceback
    from inside CuPy.
    """
    try:
        float((cupy.zeros(4, dtype=cupy.float32) + 1).sum())
    except Exception as exc:
        message = str(exc)
        if "CUDA headers" in message or "CUDA_PATH" in message or "nvrtc" in message.lower():
            raise SystemExit(
                "CuPy sees the GPU but cannot compile kernels: it has no CUDA headers.\n"
                "Install them alongside CuPy:\n"
                '  pip install "cupy-cuda13x[ctk]"      (or cupy-cuda12x[ctk] on CUDA 12)\n'
                "That pulls the toolkit through pip, so no separate CUDA install is needed.\n"
                "The alternative is the full CUDA Toolkit with CUDA_PATH pointing at it.\n"
                f"\nOriginal error: {message.strip().splitlines()[-1]}"
            ) from None
        raise

    try:
        a = cupy.ones((8, 16), dtype=cupy.float32)
        b = cupy.ones((16, 4), dtype=cupy.float32)
        float((a @ b).sum())
    except Exception as exc:
        raise SystemExit(
            f"CuPy can run its own kernels but cannot multiply two matrices.\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "Matrix products go through cuBLAS, a separate library from CuPy's kernels,\n"
            "so it can be the only broken piece. Run\n"
            "  python check_gpu.py --probe\n"
            "which prints the build versions and finds the first operation that fails.\n"
            "The usual cause is two cuBLAS libraries on the search path, one from pip\n"
            "under site-packages/nvidia and one from a system CUDA install.\n\n"
            "Meanwhile the CPU path is unaffected: python train_scratch.py"
        ) from None


def _set_numpy() -> None:
    global xp, name, _cupy
    _guard()
    xp, name, _cupy = _numpy, "numpy", None


def _set_cupy(cupy) -> None:
    global xp, name, _cupy
    _guard()
    xp, name, _cupy = cupy, "cupy", cupy


def _guard() -> None:
    if _locked:
        raise RuntimeError(
            "src.scratch has already been imported and bound the array module. "
            "Call backend.select() before importing it."
        )


def lock() -> None:
    """Called by src.scratch modules once they have bound the array module."""
    global _locked
    _locked = True


def default_rng(seed: int):
    """A Generator from whichever array module is active."""
    return xp.random.default_rng(seed)


def permutation(rng, n: int):
    """A random permutation of range(n), from whichever Generator is active.

    CuPy's Generator implements a subset of the NumPy Generator API. As of
    CuPy 14 it has `random`, `uniform`, `standard_normal` and `integers`, but
    no `permutation`, `shuffle` or `choice`. Sorting n random keys gives the
    same thing: O(n log n) instead of O(n), which for 54000 elements is
    microseconds, and it keeps the indices on the device.

    NumPy keeps its own `permutation` so that existing runs stay
    reproducible against the numbers published in docs/.
    """
    if hasattr(rng, "permutation"):
        return rng.permutation(n)
    return rng.random(n).argsort()


# Generator methods CuPy 14 provides. Anything outside this set has to be
# reached through a helper above, or the GPU path breaks at run time.
CUPY_GENERATOR_METHODS = frozenset(
    {
        "random",
        "uniform",
        "integers",
        "standard_normal",
        "standard_exponential",
        "standard_gamma",
        "beta",
        "binomial",
        "chisquare",
        "dirichlet",
        "exponential",
        "f",
        "gamma",
        "geometric",
        "hypergeometric",
        "logseries",
        "poisson",
        "power",
    }
)


def asarray(a):
    """Move a NumPy array onto the active device. A no-op under NumPy."""
    return xp.asarray(a)


def array_module(a):
    """The array module that owns `a`, so a helper can serve both backends."""
    if _cupy is not None and isinstance(a, _cupy.ndarray):
        return _cupy
    return _numpy


def to_numpy(a):
    """Bring an array back to the host. A no-op under NumPy."""
    if _cupy is not None and isinstance(a, _cupy.ndarray):
        return _cupy.asnumpy(a)
    return _numpy.asarray(a)


def synchronize() -> None:
    """Wait for queued GPU work.

    CuPy calls are asynchronous. Timing a block without this measures how
    fast Python can queue kernels, not how fast they run.
    """
    if _cupy is not None:
        _cupy.cuda.runtime.deviceSynchronize()


def describe() -> str:
    if _cupy is None:
        return f"array backend numpy {_numpy.__version__} (CPU)"
    props = _cupy.cuda.runtime.getDeviceProperties(0)
    gpu = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    free, total = _cupy.cuda.runtime.memGetInfo()
    return (
        f"array backend cupy {_cupy.__version__} on {gpu}, "
        f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB"
    )


def info() -> dict:
    return {"backend": name, "cupy_installed": _import_ok("cupy"), "gpu_visible": available()}


def _import_ok(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True
