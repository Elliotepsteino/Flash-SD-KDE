from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import numpy as np
import pytest
import torch

from flash_sd_kde.kde import (
    emp_sd_kde_fit_transform,
    kde_eval,
    kde_eval_linearized,
    kde_eval_linearized_nonfused,
)
from flash_sd_kde.estimator import FlashSDKDE
from flash_sd_kde.reference import (
    empirical_score_nd_numpy,
    empirical_sd_kde_transform_nd_numpy,
    kde_eval_1d_linearized_numpy,
    kde_eval_1d_numpy,
    kde_eval_nd_numpy,
    kde_eval_nd_linearized_numpy,
)
from globals import PRECISION_FP32_IEEE
from kernels.emp_score_16d_ordered_splitk import emp_score_16d_ordered_splitk
from kernels.emp_score_16d_symmetric_atomic import emp_score_16d_symmetric_atomic


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    pytest.importorskip("triton")


@pytest.mark.small
def test_kde_eval_1d_matches_numpy():
    _require_cuda()
    rng = np.random.default_rng(0)
    train = rng.normal(size=256).astype(np.float32)
    queries = rng.normal(size=128).astype(np.float32)
    h = 0.7

    kde_gpu = kde_eval(train, queries, h, device="cuda")
    kde_cpu = kde_eval_1d_numpy(queries, train, h)

    np.testing.assert_allclose(kde_gpu.detach().cpu().numpy(), kde_cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.small
def test_kde_eval_16d_matches_numpy():
    _require_cuda()
    rng = np.random.default_rng(1)
    train = rng.normal(size=(256, 16)).astype(np.float32)
    queries = rng.normal(size=(64, 16)).astype(np.float32)
    h = 1.2

    kde_gpu = kde_eval(train, queries, h, device="cuda", precision_mode=PRECISION_FP32_IEEE)
    kde_cpu = kde_eval_nd_numpy(queries, train, h)

    np.testing.assert_allclose(kde_gpu.detach().cpu().numpy(), kde_cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.small
def test_kde_eval_7d_matches_numpy_generalized():
    _require_cuda()
    rng = np.random.default_rng(12)
    train = rng.normal(size=(192, 7)).astype(np.float32)
    queries = rng.normal(size=(48, 7)).astype(np.float32)
    h = 0.95

    kde_gpu = kde_eval(
        train,
        queries,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
    )
    kde_cpu = kde_eval_nd_numpy(queries, train, h)

    np.testing.assert_allclose(kde_gpu.detach().cpu().numpy(), kde_cpu, rtol=2e-3, atol=2e-3)


@pytest.mark.small
def test_kde_eval_1d_generalized_matches_specialized():
    _require_cuda()
    rng = np.random.default_rng(13)
    train = rng.normal(size=256).astype(np.float32)
    queries = rng.normal(size=128).astype(np.float32)
    h = 0.9

    specialized = kde_eval(train, queries, h, device="cuda", precision_mode=PRECISION_FP32_IEEE)
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


@pytest.mark.small
def test_kde_eval_16d_generalized_matches_specialized():
    _require_cuda()
    rng = np.random.default_rng(14)
    train = rng.normal(size=(160, 16)).astype(np.float32)
    queries = rng.normal(size=(40, 16)).astype(np.float32)
    h = 1.05

    specialized = kde_eval(train, queries, h, device="cuda", precision_mode=PRECISION_FP32_IEEE)
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


@pytest.mark.small
def test_kde_eval_16d_linearized_matches_numpy():
    _require_cuda()
    rng = np.random.default_rng(5)
    train = rng.normal(size=(192, 16)).astype(np.float32)
    queries = rng.normal(size=(48, 16)).astype(np.float32)
    h = 1.05

    kde_gpu = kde_eval_linearized(train, queries, h, device="cuda", precision_mode=PRECISION_FP32_IEEE)
    kde_cpu = kde_eval_nd_linearized_numpy(queries, train, h)

    np.testing.assert_allclose(kde_gpu.detach().cpu().numpy(), kde_cpu, rtol=2e-3, atol=2e-3)


@pytest.mark.small
def test_kde_eval_1d_linearized_nonfused_matches_numpy():
    _require_cuda()
    rng = np.random.default_rng(6)
    train = rng.normal(size=128).astype(np.float32)
    queries = rng.normal(size=64).astype(np.float32)
    h = 0.85

    kde_gpu = kde_eval_linearized_nonfused(
        train,
        queries,
        h,
        device="cuda",
        chunk_size=64,
    )
    kde_cpu = kde_eval_1d_linearized_numpy(queries, train, h)

    np.testing.assert_allclose(kde_gpu.detach().cpu().numpy(), kde_cpu, rtol=2e-3, atol=2e-3)


@pytest.mark.small
def test_kde_eval_16d_linearized_nonfused_matches_numpy():
    _require_cuda()
    rng = np.random.default_rng(7)
    train = rng.normal(size=(128, 16)).astype(np.float32)
    queries = rng.normal(size=(32, 16)).astype(np.float32)
    h = 1.15

    kde_gpu = kde_eval_linearized_nonfused(
        train,
        queries,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        chunk_size=64,
    )
    kde_cpu = kde_eval_nd_linearized_numpy(queries, train, h)

    np.testing.assert_allclose(kde_gpu.detach().cpu().numpy(), kde_cpu, rtol=2e-3, atol=2e-3)


@pytest.mark.small
def test_emp_score_ordered_matches_numpy():
    _require_cuda()
    rng = np.random.default_rng(2)
    train = rng.normal(size=(128, 16)).astype(np.float32)
    h = 0.9

    pdf_gpu, weighted_gpu = emp_score_16d_ordered_splitk(
        train,
        h,
        device=torch.device("cuda"),
        precision_mode=PRECISION_FP32_IEEE,
        use_precomputed_norms=True,
        autotune=False,
    )
    pdf_cpu, weighted_cpu, _ = empirical_score_nd_numpy(train, h)

    np.testing.assert_allclose(pdf_gpu.detach().cpu().numpy(), pdf_cpu, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(weighted_gpu.detach().cpu().numpy(), weighted_cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.small
def test_emp_score_symmetric_matches_ordered():
    _require_cuda()
    rng = np.random.default_rng(3)
    train = rng.normal(size=(256, 16)).astype(np.float32)
    h = 1.1

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
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        weighted_sym.detach().cpu().numpy(),
        weighted_ordered.detach().cpu().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.small
def test_emp_sd_kde_fit_transform_runs():
    _require_cuda()
    rng = np.random.default_rng(4)
    train = rng.normal(size=(128, 16)).astype(np.float32)
    h = 1.0

    debiased = emp_sd_kde_fit_transform(
        train,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        emp_score_backend=None,
        use_precomputed_norms=True,
        autotune=False,
    )
    assert debiased.shape == train.shape
    assert torch.isfinite(debiased).all()


@pytest.mark.small
def test_emp_sd_kde_fit_transform_7d_matches_numpy():
    _require_cuda()
    rng = np.random.default_rng(15)
    train = rng.normal(size=(96, 7)).astype(np.float32)
    h = 0.92

    debiased_gpu = emp_sd_kde_fit_transform(
        train,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        emp_score_backend=None,
        use_precomputed_norms=True,
        autotune=False,
    )
    debiased_cpu = empirical_sd_kde_transform_nd_numpy(train, h)

    np.testing.assert_allclose(
        debiased_gpu.detach().cpu().numpy(),
        debiased_cpu,
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.small
def test_emp_sd_kde_fit_transform_16d_generalized_matches_specialized():
    _require_cuda()
    rng = np.random.default_rng(16)
    train = rng.normal(size=(96, 16)).astype(np.float32)
    h = 1.03

    specialized = emp_sd_kde_fit_transform(
        train,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        emp_score_backend=None,
        use_precomputed_norms=True,
        autotune=False,
        prefer_specialized_dims=True,
    )
    generalized = emp_sd_kde_fit_transform(
        train,
        h,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        emp_score_backend=None,
        use_precomputed_norms=True,
        autotune=False,
        prefer_specialized_dims=False,
    )

    np.testing.assert_allclose(
        generalized.detach().cpu().numpy(),
        specialized.detach().cpu().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.small
def test_estimator_sd_kde_runs_for_non16d():
    _require_cuda()
    rng = np.random.default_rng(17)
    train = rng.normal(size=(96, 7)).astype(np.float32)
    queries = rng.normal(size=(24, 7)).astype(np.float32)

    est = FlashSDKDE(
        mode="sd_kde",
        bandwidth=1.0,
        device="cuda",
        precision_mode=PRECISION_FP32_IEEE,
        autotune=False,
    )
    est.fit(train)
    scores = est.score_samples(queries)

    assert scores.shape == (queries.shape[0],)
    assert np.isfinite(scores).all()
