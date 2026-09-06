"""Report whether PyTorch can use your GPU, and say what to do if it cannot.

    python check_gpu.py

Run this first when training is slower than expected. The common case on
Windows is a working NVIDIA card and a PyTorch install that has no CUDA in
it at all, because `pip install torch` takes the CPU-only wheel from PyPI.
"""


def main():
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "PyTorch is not installed.\n"
            "CPU only:   pip install -r requirements-torch.txt\n"
            "With CUDA:  pip install torch --index-url https://download.pytorch.org/whl/cu130"
        ) from None

    from src.torchmlp.engine import backend_info, explain_no_cuda

    info = backend_info()
    print(f"torch            {info['torch']}")
    print(f"built with CUDA  {info['cuda_build'] or 'no (CPU-only build)'}")
    print(f"cuda available   {info['cuda_available']}")

    if not info["cuda_available"]:
        print()
        print(explain_no_cuda(info))
        return

    print(f"devices          {info['device_count']}")
    print(f"gpu 0            {info['gpu']} (compute capability {info['capability']})")
    free, total = torch.cuda.mem_get_info()
    print(f"memory           {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")

    print()
    print("matmul timing, 4096 x 4096 float32")
    for device in ("cpu", "cuda"):
        print(f"  {device:<5} {time_matmul(torch, device):>7.1f} ms")


def time_matmul(torch, device: str, size: int = 4096, repeats: int = 5) -> float:
    """Median wall time of one large matmul, after a warm up.

    A GPU kernel is queued rather than executed by the Python call, so timing
    it without `torch.cuda.synchronize()` measures how fast Python can submit
    work and reports an absurd speedup.
    """
    import time

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    for _ in range(2):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        a @ b
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)

    return sorted(timings)[len(timings) // 2]


if __name__ == "__main__":
    main()
