"""Step 1: prove PyTorch can see and use the GPU, and report free GPU memory."""

import json
import platform

import torch

import config


def main():
    assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
    free, total = torch.cuda.mem_get_info()
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(x @ x).all())
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_gb": round(free / 2**30, 2),
        "gpu_total_gb": round(total / 2**30, 2),
        "matmul_on_gpu_ok": ok,
        "enough_free_memory_for_training": free / 2**30 >= config.MIN_FREE_GPU_GB,
    }
    print(json.dumps(report, indent=2))
    (config.RESULTS_DIR / "step1").mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "step1" / "gpu_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
