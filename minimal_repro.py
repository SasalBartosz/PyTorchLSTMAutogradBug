"""Minimal repro: compiled autograd gives a wrong encoder weight_hh gradient
when a second cuDNN LSTM consumes the first LSTM's final state."""

import torch
import torch.nn as nn
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import same


def enc_grads(compiled, dtype=torch.float32, cudnn=True):
    torch.backends.cudnn.enabled = cudnn
    torch.manual_seed(0)
    enc = nn.LSTM(8, 8, 1, batch_first=True).to(device="cuda", dtype=dtype)
    dec = nn.LSTM(8, 8, 1, batch_first=True).to(device="cuda", dtype=dtype)
    x = torch.randn(4, 6, 8, device="cuda").to(dtype)  # same input values for all dtypes
    y = torch.randn(4, 3, 8, device="cuda").to(dtype)

    out, (h, c) = enc(x)
    loss = out.square().sum() + dec(y, (h, c))[0].square().sum()  # decoder consumes final state

    ctx = compiled_autograd._enable(torch.compile(dynamic=False)) if compiled else torch.enable_grad()
    with ctx:
        loss.backward()
    return [p.grad.clone() for p in enc.parameters()]


print(f"torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}, {torch.cuda.get_device_name()}")

# fp64 baseline (cuDNN RNN is fp32/fp16-only, so this runs the native kernels)
fp64 = enc_grads(compiled=False, dtype=torch.float64)

for cudnn in (True, False):
    eager = enc_grads(compiled=False, cudnn=cudnn)
    ca = enc_grads(compiled=True, cudnn=cudnn)
    ok = same(eager, ca, fp64_ref=fp64)
    print(f"cudnn={cudnn}: {'OK' if ok else 'WRONG GRADS'}")
    if not ok:
        for n, e, c, r in zip(("w_ih", "w_hh", "b_ih", "b_hh"), eager, ca, fp64):
            print(f"  {n}: |eager-fp64|max={ (e.double()-r).abs().max():.2e}  "
                  f"|ca-fp64|max={(c.double()-r).abs().max():.2e}")
