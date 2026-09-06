"""Report what each backend can see, and time a matmul on every one available.

    python check_gpu.py

Two independent questions. PyTorch needs a CUDA build to use the GPU, and
`train_torch.py` depends on it. The from scratch trainer needs CuPy for
`--device cuda`, and runs on NumPy otherwise. Neither implies the other.
"""

import time


def main():
    report_torch()
    print()
    report_cupy()
    print()
    print("matmul, 4096 x 4096 float32, median of 5")
    for label, ms in timings():
        print(f"  {label:<14} {ms:>8.1f} ms")


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
