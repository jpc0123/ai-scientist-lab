# RGB-T real detection baseline app

Fixed entry for `baseline_key=dfine_s`.

## Backends

| Backend | When | Implementation id |
|---------|------|-------------------|
| **Vendored DFINE-S** | `third_party/DFINE` present and `dfine_backend=auto\|dfine` | `dfine_s_vendored_v0_8_9` |
| **Torch mini stand-in** | `dfine_backend=standin` or vendor missing | `torch_mini_standin_v0_8_1` |

Pinned vendor commit: see `third_party/VENDOR.md` (`7fe2f888…`).

## Environments

- `rgbt-detection-v2` → `scientist-rgbt-detection:v2` (CPU torch)
- `rgbt-detection-v2-cuda` → `scientist-rgbt-detection:v2-cuda` (CUDA torch + DFINE deps)

```bat
docker build -t scientist-rgbt-detection:v2-cuda -f dockerfiles/rgbt-detection-v2-cuda/Dockerfile .
scientist-lab run examples\rgbt_remote_cuda_dfine_rgb_contract.json
```

## Entry

```bat
python run_detection_experiment.py --config ... --output-dir ... --data-root ... --execution-mode fast_eval
```
