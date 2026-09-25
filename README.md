# Compiled autograd returns a wrong cuDNN LSTM gradient

Minimal repro for a pytorch/pytorch issue: under compiled autograd, an encoder LSTM's
`weight_hh_l0` gradient is silently wrong when its output sequence is consumed by a
second cuDNN LSTM. See `issue.md` for the full draft.

## Run

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU:

```sh
uv sync
uv run python minimal_repro.py
```

Expected output (confirmed on torch 2.14.0+cu130 with cuDNN 9.24.0 and
2.15.0.dev20260925+cu130 with cuDNN 9.26.0):

```
cudnn=True: WRONG GRADS in weight_hh_l0: |grad-fp64|max = 5.9e-04 (eager: 3.3e-07)
cudnn=False: OK
```

The comparison uses `torch._dynamo.utils.same` with an fp64 baseline.

## Test the latest nightly

```sh
uv run --isolated --no-project --python 3.12 \
  --with torch --with numpy \
  --prerelease allow \
  --index pytorch-nightly=https://download.pytorch.org/whl/nightly/cu130 \
  python minimal_repro.py
```
