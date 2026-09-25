"""Minimal repro: compiled autograd gives a wrong encoder weight_hh_l0 gradient
when its output sequence is consumed by a second cuDNN LSTM."""

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
    x = torch.randn(B, T, D, device="cuda").to(dtype)  # same input values for all dtypes
    y = torch.randn(B, T, D, device="cuda").to(dtype)  # target

    loss = nn.functional.mse_loss(dec(enc(x)[0])[0], y)  # two chained LSTMs

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
