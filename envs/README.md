# Conda environment snapshot

Captured before the cluster migration, from `~/anaconda3/envs/flash-sd-kde` on soal-11.

**Python 3.13.9, torch 2.7.1+cu118, CUDA 11.8.**

Note this is a much newer stack than the `torch_cuda-11.8` environments recorded in
the safari-dev repo (Python 3.9.19 / torch 2.2.1). The two are not interchangeable.

## Files

| File | Use |
| --- | --- |
| `flash-sd-kde.environment.yml` | Full conda export. The most reliable single source. |
| `flash-sd-kde.explicit.txt` | Exact conda package URLs. Same-platform reproduction. |
| `flash-sd-kde.from-history.yml` | Only 5 packages were requested through conda; everything else came from pip. Not sufficient on its own. |
| `flash-sd-kde.pip-freeze.txt` | Pip packages. See the caveat below before using it. |
| `flash-sd-kde.versions.txt` | Python, torch, and CUDA versions. |

## Caveat on the pip freeze

24 of its 55 entries point at `file:///home/task_*/conda-bld/...` paths. Those are
conda build directories on the cluster, not real packages, so
`pip install -r flash-sd-kde.pip-freeze.txt` will fail on them. They correspond to
conda-installed packages (numpy, pandas, scipy, matplotlib and their dependencies)
that `environment.yml` already covers.

## Recreating

Install the conda side first, then the pip-only packages:

```bash
conda env create -n flash-sd-kde -f flash-sd-kde.environment.yml
```

The CUDA stack is pip-installed from the cu118 index and is the part most likely to
need attention on new hardware:

```bash
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

That pulls the matching `nvidia-*-cu11` runtime wheels and `triton` 3.3.1. On a
different CUDA version, choose the corresponding PyTorch index instead of cu118.
