from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from flash_sd_kde.kde import emp_sd_kde_fit_transform, kde_eval
from flash_sd_kde.reference import silverman_bandwidth_1d, silverman_bandwidth_nd
from globals import (
    DEFAULT_EPS,
    DEFAULT_KDE_BACKEND,
    DEFAULT_PRECISION_MODE,
    ND_FEATURES,
)

FlashMode = Literal["sd_kde", "kde"]
BandwidthSpec = float | Literal["silverman"]


@dataclass
class FlashSDKDE:
    """sklearn-style KDE estimator for Flash-SD-KDE and standard KDE.

    The API mirrors sklearn's `KernelDensity` at a high level:
    - `fit(X)` stores train data (or SD-debiased train data for `mode="sd_kde"`).
    - `score_samples(X)` returns log density values for each query sample.
    - `score(X)` returns average log density.

    Notes:
    - CUDA is required by the underlying kernels in this repository.
    - `mode="sd_kde"` currently requires 16-D inputs.
    """

    bandwidth: BandwidthSpec = "silverman"
    mode: FlashMode = "sd_kde"
    device: str | torch.device = "cuda"
    precision_mode: str = DEFAULT_PRECISION_MODE
    kde_backend: str = DEFAULT_KDE_BACKEND
    emp_score_backend: str | None = None
    use_precomputed_norms: bool = True
    autotune: bool = True
    eps: float = DEFAULT_EPS

    # Set during fit
    bandwidth_: float | None = field(default=None, init=False)
    n_features_in_: int | None = field(default=None, init=False)
    X_fit_: np.ndarray | None = field(default=None, init=False, repr=False)
    _fit_eval_data: torch.Tensor | np.ndarray | None = field(default=None, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, X: np.ndarray, y=None) -> "FlashSDKDE":
        del y
        x_fit = self._validate_input(X)
        self.n_features_in_ = x_fit.shape[1]
        self.bandwidth_ = self._resolve_bandwidth(x_fit)
        self.X_fit_ = x_fit
        self._validate_mode()

        if self.mode == "sd_kde":
            if self.n_features_in_ != ND_FEATURES:
                raise ValueError(
                    f'mode="sd_kde" currently requires {ND_FEATURES} features, '
                    f"but got {self.n_features_in_}."
                )
            self._fit_eval_data = emp_sd_kde_fit_transform(
                x_fit,
                self.bandwidth_,
                device=self.device,
                precision_mode=self.precision_mode,
                emp_score_backend=self.emp_score_backend,
                use_precomputed_norms=self.use_precomputed_norms,
                autotune=self.autotune,
                eps=self.eps,
            )
        else:
            if self.n_features_in_ == 1:
                self._fit_eval_data = x_fit[:, 0].astype(np.float32, copy=False)
            else:
                self._fit_eval_data = x_fit

        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        assert self.n_features_in_ is not None
        assert self.bandwidth_ is not None
        assert self._fit_eval_data is not None

        queries = self._validate_input(X)
        if queries.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {queries.shape[1]} features, but estimator was fitted with "
                f"{self.n_features_in_} features."
            )

        if self.n_features_in_ == 1:
            query_eval = queries[:, 0].astype(np.float32, copy=False)
        else:
            query_eval = queries

        density = kde_eval(
            self._fit_eval_data,
            query_eval,
            self.bandwidth_,
            device=self.device,
            precision_mode=self.precision_mode,
            kde_backend=self.kde_backend,
            use_precomputed_norms=self.use_precomputed_norms,
            autotune=self.autotune,
        )
        log_density = torch.log(density + self.eps)
        return log_density.detach().cpu().numpy()

    def score(self, X: np.ndarray, y=None) -> float:
        del y
        log_density = self.score_samples(X)
        return float(np.mean(log_density))

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                "Expected X with shape (n_samples, n_features), consistent with sklearn."
            )
        if arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        return arr

    def _resolve_bandwidth(self, X: np.ndarray) -> float:
        if isinstance(self.bandwidth, str):
            if self.bandwidth != "silverman":
                raise ValueError(
                    f"Unsupported bandwidth='{self.bandwidth}'. Use float or 'silverman'."
                )
            if X.shape[1] == 1:
                return float(silverman_bandwidth_1d(X[:, 0]))
            return float(silverman_bandwidth_nd(X))

        bw = float(self.bandwidth)
        if bw <= 0:
            raise ValueError("bandwidth must be positive.")
        return bw

    def _validate_mode(self) -> None:
        if self.mode not in {"sd_kde", "kde"}:
            raise ValueError(
                f"Unsupported mode='{self.mode}'. Expected one of ['sd_kde', 'kde']."
            )

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("Estimator is not fitted. Call fit(X) before score_samples(X).")
