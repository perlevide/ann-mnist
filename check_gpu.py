"""Report what each backend can see, and time a matmul on every one available.

    python check_gpu.py
    python check_gpu.py --probe     when something fails and you need to know where

Two independent questions. PyTorch needs a CUDA build to use the GPU, and
`train_torch.py` depends on it. The from scratch trainer needs CuPy for
`--device cuda`, and runs on NumPy otherwise. Neither implies the other.

`--probe` runs CuPy through a staircase of operations, from allocating an
array to the matmul shapes this project actually uses, and reports the first
one that fails together with the build versions. It separates a broken
library load from a real problem with the arrays.
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="run the CuPy operation staircase")
    parser.add_argument(
        "--staircase",
        action="store_true",
        help="run only the staircase in this process, without importing torch",
    )
    parser.add_argument(
        "--with-torch", action="store_true", help="with --staircase, import torch first"
    )
    args = parser.parse_args()

    if args.staircase:
        if args.with_torch:
            import torch  # noqa: F401
        probe_cupy(show_config=not args.with_torch)
        return

    report_torch()
    print()
    report_cupy()

    if args.probe:
        print()
        compare_with_and_without_torch()
        return

    print()
    print("matmul, 4096 x 4096 float32, median of 5")
    for label, ms in timings():
        print(f"  {label:<14} {ms:>8.1f} ms")


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def compare_with_and_without_torch() -> None:
    """Run the staircase twice in fresh processes, once with torch loaded and once without.

    This matters because both libraries ship their own CUDA runtime. Whichever
    is imported first can decide which cuBLAS the process ends up using, so a
    program that imports both can work while one that imports only CuPy fails,
    or the reverse. `train_scratch.py` never imports torch; this script does,
    which is exactly the difference worth measuring.
    """
    import subprocess
    import sys

    variants = [("CuPy alone", [])]
    if has_torch():
        variants.append(("torch imported first", ["--with-torch"]))

    for label, extra in variants:
        print(f"=== {label} ===")
        result = subprocess.run(
            [sys.executable, __file__, "--staircase", *extra],
            capture_output=True,
            text=True,
        )
        print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip())
        print()

    if len(variants) == 1:
        print("PyTorch is not installed, so there is nothing to compare against.")
        return

    print(
        "If these two disagree, the CUDA libraries are being resolved differently\n"
        "depending on import order. `cupy.show_config()` above reports which CUDA\n"
        "root CuPy picked; when that points inside torch's package directory, CuPy\n"
        "is borrowing torch's libraries rather than the ones installed for it.\n"
        "Fixes, in order of preference:\n"
        "  1. Point CUDA_PATH at CuPy's own libraries, so import order stops mattering:\n"
        "       $env:CUDA_PATH = \"$PWD\\.venv\\Lib\\site-packages\\nvidia\\cu13\"\n"
        "     Set it permanently with setx once it works.\n"
        "  2. Keep CuPy and PyTorch in separate virtual environments.\n"
        "  3. Use train_torch.py --device cuda for GPU work and leave the from\n"
        "     scratch trainer on the CPU, where it is fast enough anyway."
    )


def probe_cupy(show_config: bool = True) -> None:
    """Run CuPy through increasingly demanding operations and stop at the first failure.

    A failure at "allocate" or "elementwise" points at the install. A failure
    that starts at the first matmul points at cuBLAS, which is a separate
    library from the CuPy kernels and is loaded separately, so it can be the
    only broken piece.
    """
    try:
        import cupy
    except ImportError:
        print("CuPy is not installed, nothing to probe")
        return

    if show_config:
        print("build configuration")
        print("-" * 60)
        cupy.show_config()
        print("-" * 60)
        print()

    steps = [
        ("allocate", lambda: cupy.zeros((128, 784), dtype=cupy.float32)),
        ("elementwise", lambda: cupy.zeros((128, 784), dtype=cupy.float32) + 1),
        ("reduction", lambda: cupy.zeros((128, 784), dtype=cupy.float32).sum()),
        ("rng", lambda: cupy.random.default_rng(0).standard_normal((128, 784), dtype=cupy.float32)),
        ("argsort", lambda: cupy.random.default_rng(0).random(1000).argsort()),
        ("fancy index", lambda: cupy.zeros((1000, 8), dtype=cupy.float32)[cupy.arange(128)]),
        ("matmul 4x4", lambda: _matmul(cupy, (4, 4), (4, 4))),
        ("matmul 128x784 @ 784x512", lambda: _matmul(cupy, (128, 784), (784, 512))),
        ("matmul float64", lambda: _matmul(cupy, (128, 784), (784, 512), cupy.float64)),
        ("matmul non-contiguous", lambda: _matmul_strided(cupy)),
    ]

    for label, call in steps:
        try:
            result = call()
            cupy.cuda.Device().synchronize()
            detail = ""
            if hasattr(result, "shape"):
                flags = "C" if result.flags.c_contiguous else "F" if result.flags.f_contiguous else "-"
                detail = f"  -> {result.shape} {result.dtype} {flags}"
            print(f"  ok    {label}{detail}")
        except Exception as exc:
            print(f"  FAIL  {label}")
            print(f"        {type(exc).__name__}: {exc}")
            print()
            print("Everything above this line works, so the problem is in this step.")
            if "matmul" in label:
                print(
                    "A failure that begins at the first matmul is cuBLAS, not CuPy's own\n"
                    "kernels. cuBLAS is a separate library loaded separately, so it can be\n"
                    "the only broken piece. Usual causes, in order:\n"
                    "  1. Two cuBLAS libraries on the search path, one from pip under\n"
                    "     site-packages/nvidia and one from a system CUDA install. Check\n"
                    "     CUDA_PATH and PATH, and prefer one source.\n"
                    "  2. A cuBLAS major version that does not match the CuPy wheel.\n"
                    "     cupy-cuda13x wants CUDA 13; cupy-cuda12x wants CUDA 12.\n"
                    "  3. A driver older than the toolkit the wheel was built against.\n"
                    "Reinstalling into a clean venv with only cupy-cuda13x[ctk] and no\n"
                    "system CUDA on PATH is the fastest way to rule out the first two."
                )
            return

    print()
    print("every step passed, so CuPy is healthy on this machine")


def _matmul(cupy, shape_a, shape_b, dtype=None):
    dtype = dtype or cupy.float32
    a = cupy.ones(shape_a, dtype=dtype)
    b = cupy.ones(shape_b, dtype=dtype)
    return a @ b


def _matmul_strided(cupy):
    """A slice with a gap, to check whether cuBLAS is being handed bad strides."""
    big = cupy.ones((256, 784), dtype=cupy.float32)
    return big[::2] @ cupy.ones((784, 64), dtype=cupy.float32)


def report_torch() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch    not installed")
        print("           pip install torch --index-url https://download.pytorch.org/whl/cu130")
        return

    from src.torchmlp.engine import backend_info, explain_no_cuda

    info = backend_info()
    print(f"PyTorch    {info['torch']}")
    print(f"           built with CUDA {info['cuda_build'] or 'no (CPU-only build)'}")
    print(f"           cuda available  {info['cuda_available']}")
    if info["cuda_available"]:
        free, total = torch.cuda.mem_get_info()
        print(f"           gpu 0 {info['gpu']}, compute capability {info['capability']}")
        print(f"           memory {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
    else:
        for line in explain_no_cuda(info).splitlines():
            print(f"           {line}")


def report_cupy() -> None:
    try:
        import cupy
    except ImportError:
        print("CuPy       not installed, so train_scratch.py runs on the CPU")
        print("           pip install cupy-cuda13x        (match your CUDA version)")
        return

    count = cupy.cuda.runtime.getDeviceCount()
    print(f"CuPy       {cupy.__version__}")
    print(f"           CUDA runtime {cupy.cuda.runtime.runtimeGetVersion()}")
    print(f"           devices visible {count}")
    if count:
        props = cupy.cuda.runtime.getDeviceProperties(0)
        gpu = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        free, total = cupy.cuda.runtime.memGetInfo()
        print(f"           gpu 0 {gpu}")
        print(f"           memory {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")


def timings(size: int = 4096, repeats: int = 5) -> list:
    """Median wall time of one large matmul per available backend.

    Both GPU libraries queue work rather than running it, so each timing
    synchronizes before stopping the clock. Without that you measure how fast
    Python can submit kernels and get an absurd speedup.
    """
    results = []

    import numpy

    results.append(("numpy cpu", _median(numpy, size, repeats, lambda: None)))

    try:
        import cupy
    except ImportError:
        pass
    else:
        if cupy.cuda.runtime.getDeviceCount():
            results.append(("cupy gpu", _median(cupy, size, repeats, cupy.cuda.Device().synchronize)))

    try:
        import torch
    except ImportError:
        return results

    a = torch.randn(size, size)
    b = torch.randn(size, size)
    results.append(("torch cpu", _median_torch(a, b, repeats, lambda: None)))
    if torch.cuda.is_available():
        a, b = a.cuda(), b.cuda()
        results.append(("torch gpu", _median_torch(a, b, repeats, torch.cuda.synchronize)))
    return results


def _median(module, size: int, repeats: int, sync) -> float:
    a = module.random.default_rng(0).standard_normal((size, size), dtype="float32")
    b = module.random.default_rng(1).standard_normal((size, size), dtype="float32")
    a @ b
    sync()
    return _time(lambda: a @ b, sync, repeats)


def _median_torch(a, b, repeats: int, sync) -> float:
    a @ b
    sync()
    return _time(lambda: a @ b, sync, repeats)


def _time(call, sync, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        sync()
        timings.append((time.perf_counter() - started) * 1000)
    return sorted(timings)[len(timings) // 2]


if __name__ == "__main__":
    main()
