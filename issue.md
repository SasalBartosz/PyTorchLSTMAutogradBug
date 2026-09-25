# Title

[compiled autograd] Chained cuDNN LSTMs silently produce a wrong weight_hh_l0 gradient

## 🐛 Describe the bug

When the output sequence of one cuDNN LSTM is passed to a second cuDNN LSTM,
compiled autograd silently returns a wrong `weight_hh_l0` gradient for the first
LSTM. No exception or warning is emitted. Every other parameter gradient matches
eager.

Three conditions must all hold:

1. The LSTMs run on cuDNN (`torch.backends.cudnn.enabled = False` makes every
   gradient correct).
2. A second cuDNN LSTM consumes the first LSTM's output, so gradients flow between
   the two backward nodes.
3. Backward runs under compiled autograd. Ordinary eager backward remains close to
   the fp64 reference.

### Minimal reproducer

```python
import torch
import torch.nn as nn
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import same

B, T, D = 2, 4, 8  # batch, timesteps, features


def enc_weight_grads(compiled, dtype=torch.float32, cudnn=True):
    torch.backends.cudnn.enabled = cudnn
    torch.manual_seed(0)
    enc = nn.LSTM(D, D, batch_first=True).to(device="cuda", dtype=dtype)
    dec = nn.LSTM(D, D, batch_first=True).to(device="cuda", dtype=dtype)
    x = torch.randn(B, T, D, device="cuda").to(dtype)
    y = torch.randn(B, T, D, device="cuda").to(dtype)

    loss = nn.functional.mse_loss(dec(enc(x)[0])[0], y)

    if compiled:
        with compiled_autograd._enable(torch.compile(dynamic=False)):
            loss.backward()
    else:
        loss.backward()
    return {n: p.grad.clone() for n, p in enc.named_parameters()}


print(
    f"torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, "
    f"{torch.cuda.get_device_name()}"
)

fp64 = enc_weight_grads(False, torch.float64)  # fp64 reference

for cudnn in (True, False):
    eager = enc_weight_grads(False, cudnn=cudnn)
    ca = enc_weight_grads(True, cudnn=cudnn)
    if same(eager, ca, fp64_ref=fp64, log_error=lambda *a, **k: None):
        print(f"cudnn={cudnn}: OK")
    else:
        g = "weight_hh_l0"  # the only wrong gradient
        e_err = (eager[g].double() - fp64[g]).abs().max()
        ca_err = (ca[g].double() - fp64[g]).abs().max()
        print(
            f"cudnn={cudnn}: WRONG GRADS in {g}: "
            f"|grad-fp64|max = {ca_err:.1e} (eager: {e_err:.1e})"
        )
```

### Observed output

PyTorch 2.14.0 stable:

```text
torch 2.14.0+cu130, cuDNN 92400, NVIDIA GeForce RTX 5070 Ti
cudnn=True: WRONG GRADS in weight_hh_l0: |grad-fp64|max = 5.9e-04 (eager: 3.3e-07)
cudnn=False: OK
```

The problem reproduces unchanged on the 2026-09-25 nightly, including the same
error magnitudes:

```text
torch 2.15.0.dev20260925+cu130, cuDNN 92600, NVIDIA GeForce RTX 5070 Ti
cudnn=True: WRONG GRADS in weight_hh_l0: |grad-fp64|max = 5.9e-04 (eager: 3.3e-07)
cudnn=False: OK
```

The attached full tlparse archive was generated from this nightly run. It shows
a graph break at `aten._cudnn_rnn_backward.default` because no fake implementation
is registered; execution nevertheless completes and returns the wrong gradient.

Compiled autograd's `weight_hh_l0` maximum error against fp64 is about 1800 times
the eager fp32 error. `torch._dynamo.utils.same` rejects that gradient using its
default tolerance and fp64-reference accuracy check; every other parameter passes.

### Ablations

| Change | Result |
| --- | --- |
| None (two chained cuDNN LSTMs) | WRONG GRADS — only the first LSTM's `weight_hh_l0` |
| `torch.backends.cudnn.enabled = False` | OK |
| Single LSTM (no decoder chained) | OK |
| Backward without compiled autograd | OK |
| The second LSTM's own gradients | Always OK |

On PyTorch 2.14.0, the result is independent of the compiler backend or Inductor
mode passed to `compiled_autograd._enable(...)`:

| Backend / mode | Result |
| --- | --- |
| `inductor` (default) | WRONG GRADS |
| `aot_eager` | WRONG GRADS |
| `eager` | WRONG GRADS |
| `inductor`, `mode="reduce-overhead"` | WRONG GRADS |
| `inductor`, `mode="max-autotune"` | WRONG GRADS |

### Expected behavior

Compiled autograd should match eager gradients within expected fp32 numerical
error. If it cannot safely compile the cuDNN RNN backward, it should fall back
correctly or raise an error rather than return an incorrect gradient.

### Reproducer environment

The self-contained reproducer, `pyproject.toml`, and `uv.lock` are available at:
https://github.com/SasalBartosz/PyTorchLSTMAutogradBug

Stable reproduction:

```sh
uv sync
uv run python minimal_repro.py
```

One-off nightly reproduction:

```sh
uv run --isolated --no-project --python 3.12 \
  --with torch --with numpy \
  --prerelease allow \
  --index pytorch-nightly=https://download.pytorch.org/whl/nightly/cu130 \
  python minimal_repro.py
```

### Related reports

- https://github.com/pytorch/pytorch/issues/82577 discusses how monolithic cuDNN
  RNN operations and saved buffers complicate AOTAutograd, but does not report
  incorrect gradients.
- https://github.com/pytorch/pytorch/issues/158131 reports a different
  `_cudnn_rnn.default` compilation/fake-tensor failure, not compiled-autograd
  gradient corruption.

I found no issue or pull request reporting this exact failure mode as of
2026-09-25.

### AI assistance disclosure

> - **Tool:** AI coding agent
> - **Scope:** The agent helped surface the discrepancy during LSTM benchmarking,
>   minimize the reproducer, search for related reports, and organize/edit this issue.

I personally ran and verified the stable and nightly reproductions, ablations,
environment details, and attached trace. I reviewed the report and take
responsibility for its contents.

## Error logs

No exception or warning is emitted. This is a silent correctness failure; the
observed output is included above.

## Versions

The issue reproduces on both:

- PyTorch `2.14.0+cu130`, cuDNN `9.24.0`
- PyTorch nightly `2.15.0.dev20260925+cu130`, cuDNN `9.26.0`

Stable environment (`collect_env.py`):

```text
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
