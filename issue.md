# Draft: GitHub issue for pytorch/pytorch

## 🐛 Describe the bug

Under compiled autograd (`torch._dynamo.compiled_autograd._enable(...)`), an encoder LSTM's
backward silently returns a wrong gradient for one parameter: the last layer's
`weight_hh`, no exception or warning is generated and all the other gradients match eager

Three conditions must all hold:

1. LSTM runs on cuDNN (`torch.backends.cudnn.enabled = False` → all gradients correct).
2. A second cuDNN LSTM is chained after it, so gradients flow between the
   two backward nodes.
3. Backward runs under compiled autograd. Without it, gradients match eager to fp32 noise.

### Minimal repro

```python
import torch
import torch.nn as nn
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import same

B, T, F = 2, 4, 8  # batch, timesteps, features


def enc_weight_grads(compiled, dtype=torch.float32, cudnn=True):
    torch.backends.cudnn.enabled = cudnn
    torch.manual_seed(0)
    enc = nn.LSTM(F, F, batch_first=True).to(device="cuda", dtype=dtype)
    dec = nn.LSTM(F, F, batch_first=True).to(device="cuda", dtype=dtype)
    x = torch.randn(B, T, F, device="cuda").to(dtype)
    y = torch.randn(B, T, F, device="cuda").to(dtype)

    loss = nn.functional.mse_loss(dec(enc(x)[0])[0], y)

    if compiled:
        with compiled_autograd._enable(torch.compile(dynamic=False)):
            loss.backward()
    else:
        loss.backward()
    return {n: p.grad.clone() for n, p in enc.named_parameters()}


print(f"torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, {torch.cuda.get_device_name()}")

fp64 = enc_weight_grads(False, torch.float64)  # fp64 reference

for cudnn in (True, False):
    eager = enc_weight_grads(False, cudnn=cudnn)
    ca = enc_weight_grads(True, cudnn=cudnn)
    if same(eager, ca, fp64_ref=fp64):
        print(f"cudnn={cudnn}: OK")
    else:
        g = "weight_hh_l0"  # the only wrong gradient
        e_err = (eager[g].double() - fp64[g]).abs().max()
        ca_err = (ca[g].double() - fp64[g]).abs().max()
        print(f"cudnn={cudnn}: WRONG GRADS in {g}: |grad-fp64|max = {ca_err:.1e} (eager: {e_err:.1e})")
```

Output:

```
torch 2.14.0+cu130, cuDNN 92400, NVIDIA GeForce RTX 5070 Ti
cudnn=True: WRONG GRADS in weight_hh_l0: |grad-fp64|max = 5.9e-04 (eager: 3.3e-07)
cudnn=False: OK
```

Compiled autograd's `weight_hh` error vs fp64 is ~1800x the eager fp32 error; every
other parameter matches.

### Ablation

All on the MSE wiring:

| change                                 | result                                             |
| -------------------------------------- | -------------------------------------------------- |
| none (two chained cuDNN LSTMs)         | WRONG GRADS — only the first LSTM's `weight_hh_l0` |
| `torch.backends.cudnn.enabled = False` | OK                                                 |
| single LSTM (no decoder chained)       | OK                                                 |
| backward without compiled autograd     | OK                                                 |
| the second LSTM's own gradients        | always OK                                          |

torch.compile backend/mode passed to `compiled_autograd._enable(...)`:

| backend / mode                       | result      |
| ------------------------------------ | ----------- |
| `inductor` (default)                 | WRONG GRADS |
| `aot_eager`                          | WRONG GRADS |
| `eager`                              | WRONG GRADS |
| `inductor`, `mode="reduce-overhead"` | WRONG GRADS |
| `inductor`, `mode="max-autotune"`    | WRONG GRADS |

## Expected behavior

Compiled autograd matches eager gradients (within fp32 noise), or graph-breaks/errors instead of
silently returning a wrong gradient.

## Python environment

Environment managed with [uv](https://docs.astral.sh/uv/). The exact `pyproject.toml` used
(`uv.lock` in the gist; reproduce with `uv sync && uv run python minimal_repro.py`):

```toml
[project]
name = "autogradbug"
version = "0.1.0"
description = ""
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.5.3",
    "torch>=2.14.0",
]
```

## Versions

```
PyTorch version: 2.14.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.4 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version: Could not collect
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.12.3 (main, Jul 15 2026, 23:46:41) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-7.0.0-31-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to:
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 5070 Ti
Nvidia driver version: 580.173.02
cuDNN version: 9.24.0 (from torch.backends.cudnn.version())
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU: AMD Ryzen 7 5800X3D 8-Core Processor (16 threads, x86_64)

Versions of relevant libraries:
[pip3] Could not collect
[conda] Could not collect
```
