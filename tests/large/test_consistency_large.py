from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import numpy as np
import pytest
import torch

from flash_sd_kde.kde import kde_eval
from globals import PRECISION_FP32_IEEE
from kernels.emp_score_16d_ordered_splitk import emp_score_16d_ordered_splitk
from kernels.emp_score_16d_symmetric_atomic import emp_score_16d_symmetric_atomic


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    pytest.importorskip("triton")


@pytest.mark.large
def test_emp_score_backends_consistency_large():
    _require_cuda()
    rng = np.random.default_rng(10)
    train = rng.normal(size=(1024, 16)).astype(np.float32)
    h = 1.0

    pdf_ordered, weighted_ordered = emp_score_16d_ordered_splitk(
        train,
        h,
        device=torch.device("cuda"),
        precision_mode=PRECISION_FP32_IEEE,
        use_precomputed_norms=True,
        autotune=False,
    )
    pdf_sym, weighted_sym = emp_score_16d_symmetric_atomic(
        train,
        h,
        device=torch.device("cuda"),
        precision_mode=PRECISION_FP32_IEEE,
        use_precomputed_norms=True,
        block_size=64,
    )

    np.testing.assert_allclose(
        pdf_sym.detach().cpu().numpy(),
        pdf_ordered.detach().cpu().numpy(),
        rtol=5e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        weighted_sym.detach().cpu().numpy(),
        weighted_ordered.detach().cpu().numpy(),
        rtol=5e-3,
        atol=5e-3,
    )


@pytest.mark.large
def test_kde_eval_manual_chunking_equivalence():
    _require_cuda()
    rng = np.random.default_rng(11)
    train = rng.normal(size=(512, 16)).astype(np.float32)
    queries = rng.normal(size=(1024, 16)).astype(np.float32)
    h = 1.1

    full = kde_eval(train, queries, h, device="cuda", precision_mode=PRECISION_FP32_IEEE)
    half = queries.shape[0] // 2
    part_a = kde_eval(train, queries[:half], h, device="cuda", precision_mode=PRECISION_FP32_IEEE)
    part_b = kde_eval(train, queries[half:], h, device="cuda", precision_mode=PRECISION_FP32_IEEE)
    concat = torch.cat([part_a, part_b], dim=0)

    np.testing.assert_allclose(full.detach().cpu().numpy(), concat.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.large
def test_kde_eval_generalized_matches_specialized_large():
    _require_cuda()
    rng = np.random.default_rng(12)
    train = rng.normal(size=(512, 16)).astype(np.float32)
    queries = rng.normal(size=(768, 16)).astype(np.float32)
    h = 1.0

    specialized = kde_eval(
        train,
        queries,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        prefer_specialized_dims=True,
    )
    generalized = kde_eval(
        train,
        queries,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        prefer_specialized_dims=False,
    )

    np.testing.assert_allclose(
        generalized.detach().cpu().numpy(),
        specialized.detach().cpu().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
