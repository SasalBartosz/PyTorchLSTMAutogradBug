# Compiled autograd returns a wrong cuDNN LSTM gradient

Minimal repro for a pytorch/pytorch issue: under compiled autograd, an encoder LSTM's
last-layer `weight_hh` gradient is silently wrong when its final state is consumed by a
second cuDNN LSTM. See `issue.md` for the full draft.

## Run

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU:

```sh
uv sync
uv run python minimal_repro.py
```

Expected output (torch 2.14.0+cu130, cuDNN 9.24.0):

```
cudnn=True: WRONG GRADS in weight_hh_l0: |grad-fp64|max = 5.9e-04 (eager: 3.3e-07)
cudnn=False: OK
```

The comparison uses `torch._dynamo.utils.same` with an fp64 baseline.
