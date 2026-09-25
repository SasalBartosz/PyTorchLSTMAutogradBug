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
cudnn=True: WRONG GRADS
  w_ih: |eager-fp64|max=1.06e-03  |ca-fp64|max=1.06e-03
  w_hh: |eager-fp64|max=1.69e-04  |ca-fp64|max=2.39e-01
  b_ih: |eager-fp64|max=3.72e-04  |ca-fp64|max=3.72e-04
  b_hh: |eager-fp64|max=3.72e-04  |ca-fp64|max=3.72e-04
cudnn=False: OK
```

The comparison uses `torch._dynamo.utils.same` with an fp64 baseline.
