from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets

from benchmarks.mnist_fashion_pca16_ood_config import BackendVariant, MnistFashionOodConfig
from flash_sd_kde.kde import kde_eval, kde_eval_linearized
from flash_sd_kde.reference import silverman_bandwidth_nd
from flash_sd_kde.utils import get_repo_state, make_run_dir, write_json
from globals import (
    DEFAULT_EPS,
    EMP_SD_KDE_VARIANT_EXACT,
    EMP_SD_KDE_VARIANT_LINEARIZED,
    EMP_SCORE_BACKEND_ORDERED_SPLITK,
    EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC,
    ND_FEATURES,
)
from kernels.emp_score_16d_ordered_splitk import emp_score_16d_ordered_splitk
from kernels.emp_score_16d_symmetric_atomic import emp_score_16d_symmetric_atomic


def _load_dataset(root: Path, *, train: bool, fashion: bool) -> torch.Tensor:
    if fashion:
        dataset = datasets.FashionMNIST(root=str(root), train=train, download=True)
    else:
        dataset = datasets.MNIST(root=str(root), train=train, download=True)
    data = dataset.data.float() / 255.0
    return data


def _flatten(x: torch.Tensor) -> np.ndarray:
    return x.reshape(x.shape[0], -1).numpy().astype(np.float32, copy=False)


def _time_call(fn, *, device: torch.device) -> Tuple[object, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize(device)
        return result, start.elapsed_time(end) / 1000.0
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _select_bandwidth(
    train: np.ndarray,
    val: np.ndarray,
    base_h: float,
    *,
    config: MnistFashionOodConfig,
    backend: BackendVariant,
    device: torch.device,
) -> Tuple[float, float]:
    best_h = base_h
    best_ll = -np.inf
    for mult in config.bandwidth_multipliers:
        h = base_h * mult
        dens = kde_eval(
            train,
            val,
            h,
            device=device,
            precision_mode=backend.precision_mode,
            kde_backend=backend.kde_backend,
            use_precomputed_norms=backend.use_precomputed_norms,
            autotune=backend.autotune,
        )
        dens_np = dens.detach().cpu().numpy()
        ll = float(np.mean(np.log(dens_np + DEFAULT_EPS)))
        if ll > best_ll:
            best_ll = ll
            best_h = h
    return best_h, best_ll


def _compute_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    safe_id_scores = np.maximum(id_scores, DEFAULT_EPS)
    mean_ll = float(np.mean(np.log(safe_id_scores)))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "mean_log_likelihood": mean_ll}


def _collect_backend_variants(config: MnistFashionOodConfig) -> Dict[str, BackendVariant]:
    variants = list(config.backend_variants)
    if not variants:
        raise ValueError("backend_variants must be non-empty.")
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError("backend_variants names must be unique.")
    valid_variants = {EMP_SD_KDE_VARIANT_EXACT, EMP_SD_KDE_VARIANT_LINEARIZED}
    for variant in variants:
        if variant.emp_sd_kde_variant not in valid_variants:
            raise ValueError(
                f"emp_sd_kde_variant must be one of {sorted(valid_variants)}, got {variant.emp_sd_kde_variant}."
            )
    return {variant.name: variant for variant in variants}


def _emp_score_backend(
    backend: BackendVariant,
    train: np.ndarray,
    bandwidth: float,
    *,
    device: torch.device,
):
    if backend.emp_score_backend == EMP_SCORE_BACKEND_ORDERED_SPLITK:
        return emp_score_16d_ordered_splitk(
            train,
            bandwidth,
            device=device,
            precision_mode=backend.precision_mode,
            use_precomputed_norms=backend.use_precomputed_norms,
            autotune=backend.autotune,
        )
    if backend.emp_score_backend == EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC:
        return emp_score_16d_symmetric_atomic(
            train,
            bandwidth,
            device=device,
            precision_mode=backend.precision_mode,
            use_precomputed_norms=backend.use_precomputed_norms,
            autotune=backend.autotune,
        )
    raise ValueError(f"unsupported emp_score_backend: {backend.emp_score_backend}")


def _kde_eval_emp_linearized(
    backend: BackendVariant,
    train: np.ndarray,
    queries: np.ndarray,
    bandwidth: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    return kde_eval_linearized(
        train,
        queries,
        bandwidth,
        device=device,
        precision_mode=backend.precision_mode,
        kde_backend=backend.kde_backend,
        use_precomputed_norms=backend.use_precomputed_norms,
        autotune=backend.autotune,
    )


def run_benchmark(config: MnistFashionOodConfig) -> Path:
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("benchmark expects CUDA device for v2 kernels.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but was requested.")
    if config.pca_components != ND_FEATURES:
        raise ValueError(f"pca_components must be {ND_FEATURES} for v2 kernels.")

    backend_variants = _collect_backend_variants(config)
    if config.bandwidth_backend_name not in backend_variants:
        raise ValueError("bandwidth_backend_name must match a backend_variants entry.")
    if config.primary_backend_name not in backend_variants:
        raise ValueError("primary_backend_name must match a backend_variants entry.")
    bandwidth_backend = backend_variants[config.bandwidth_backend_name]

    run_dir = make_run_dir(tag=config.output_tag)
    data_root = run_dir / "datasets"

    mnist_train = _load_dataset(data_root, train=True, fashion=False)
    mnist_test = _load_dataset(data_root, train=False, fashion=False)
    fashion_test = _load_dataset(data_root, train=False, fashion=True)

    x_train = _flatten(mnist_train)
    x_id = _flatten(mnist_test)
    x_ood = _flatten(fashion_test)

    pca = PCA(n_components=config.pca_components, random_state=config.seed)
    x_train_16 = pca.fit_transform(x_train).astype(np.float32, copy=False)
    x_id_16 = pca.transform(x_id).astype(np.float32, copy=False)
    x_ood_16 = pca.transform(x_ood).astype(np.float32, copy=False)

    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(x_train_16.shape[0])
    val_idx = perm[: config.n_val]
    train_pool_idx = perm[config.n_val :]
    train_pool = x_train_16[train_pool_idx]

    pool_perm = rng.permutation(train_pool.shape[0])
    train_pool = train_pool[pool_perm]
    train_pool_idx = train_pool_idx[pool_perm]

    curves_n_train = max(config.n_train_list)

    results = {
        "config": asdict(config),
        "meta": get_repo_state(),
        "n_train_list": list(config.n_train_list),
        "backend_variants": {name: asdict(variant) for name, variant in backend_variants.items()},
        "bandwidth_backend_name": config.bandwidth_backend_name,
        "primary_backend_name": config.primary_backend_name,
        "val_indices": val_idx.tolist(),
        "train_indices": {},
        "bandwidth": {},
        "bandwidth_loglik": {},
        "metrics": {name: {} for name in backend_variants},
        "runtime_sec": {name: {} for name in backend_variants},
        "curves_n_train": curves_n_train,
    }

    densities_for_curves: Dict[str, Dict[str, np.ndarray]] = {}

    for n_train in config.n_train_list:
        train_subset = train_pool[:n_train]
        train_idx = train_pool_idx[:n_train]
        results["train_indices"][str(n_train)] = train_idx.tolist()

        base_h = silverman_bandwidth_nd(train_subset)
        if config.use_val_bandwidth:
            val_set = x_train_16[val_idx]
            h, val_ll = _select_bandwidth(
                train_subset,
                val_set,
                base_h,
                config=config,
                backend=bandwidth_backend,
                device=device,
            )
        else:
            h = base_h
            val_ll = float("nan")

        results["bandwidth"][str(n_train)] = float(h)
        results["bandwidth_loglik"][str(n_train)] = val_ll

        for backend_name, backend in backend_variants.items():
            _ = kde_eval(
                train_subset,
                x_id_16,
                h,
                device=device,
                precision_mode=backend.precision_mode,
                kde_backend=backend.kde_backend,
                use_precomputed_norms=backend.use_precomputed_norms,
                autotune=backend.autotune,
            )
            if backend.emp_sd_kde_variant == EMP_SD_KDE_VARIANT_LINEARIZED:
                _ = _kde_eval_emp_linearized(backend, train_subset, x_id_16, h, device=device)
            else:
                _ = _emp_score_backend(backend, train_subset, h, device=device)
            torch.cuda.synchronize(device)

            def kde_eval_id():
                return kde_eval(
                    train_subset,
                    x_id_16,
                    h,
                    device=device,
                    precision_mode=backend.precision_mode,
                    kde_backend=backend.kde_backend,
                    use_precomputed_norms=backend.use_precomputed_norms,
                    autotune=backend.autotune,
                )

            def kde_eval_ood():
                return kde_eval(
                    train_subset,
                    x_ood_16,
                    h,
                    device=device,
                    precision_mode=backend.precision_mode,
                    kde_backend=backend.kde_backend,
                    use_precomputed_norms=backend.use_precomputed_norms,
                    autotune=backend.autotune,
                )

            dens_id, t_eval_id = _time_call(kde_eval_id, device=device)
            dens_ood, t_eval_ood = _time_call(kde_eval_ood, device=device)
            dens_id_np = dens_id.detach().cpu().numpy()
            dens_ood_np = dens_ood.detach().cpu().numpy()

            metrics_kde = _compute_metrics(dens_id_np, dens_ood_np)
            runtime_kde = {"eval_id_sec": t_eval_id, "eval_ood_sec": t_eval_ood}

            if backend.emp_sd_kde_variant == EMP_SD_KDE_VARIANT_LINEARIZED:
                def emp_eval_id():
                    return _kde_eval_emp_linearized(backend, train_subset, x_id_16, h, device=device)

                def emp_eval_ood():
                    return _kde_eval_emp_linearized(backend, train_subset, x_ood_16, h, device=device)

                dens_id_emp, t_eval_id_emp = _time_call(emp_eval_id, device=device)
                dens_ood_emp, t_eval_ood_emp = _time_call(emp_eval_ood, device=device)
                t_score = 0.0
                t_shift = 0.0
            else:
                def score_kernel():
                    return _emp_score_backend(backend, train_subset, h, device=device)

                (pdf_sum, weighted_sum), t_score = _time_call(score_kernel, device=device)

                train_tensor = torch.as_tensor(train_subset, device=device, dtype=torch.float32)

                def shift_data():
                    inv_h2 = 1.0 / (h * h)
                    score = (weighted_sum / (pdf_sum[:, None] + DEFAULT_EPS) - train_tensor) * inv_h2
                    delta = 0.5 * (h * h)
                    return train_tensor + delta * score

                debiased, t_shift = _time_call(shift_data, device=device)

                def emp_eval_id():
                    return kde_eval(
                        debiased,
                        x_id_16,
                        h,
                        device=device,
                        precision_mode=backend.precision_mode,
                        kde_backend=backend.kde_backend,
                        use_precomputed_norms=backend.use_precomputed_norms,
                        autotune=backend.autotune,
                    )

                def emp_eval_ood():
                    return kde_eval(
                        debiased,
                        x_ood_16,
                        h,
                        device=device,
                        precision_mode=backend.precision_mode,
                        kde_backend=backend.kde_backend,
                        use_precomputed_norms=backend.use_precomputed_norms,
                        autotune=backend.autotune,
                    )

                dens_id_emp, t_eval_id_emp = _time_call(emp_eval_id, device=device)
                dens_ood_emp, t_eval_ood_emp = _time_call(emp_eval_ood, device=device)
            dens_id_emp_np = dens_id_emp.detach().cpu().numpy()
            dens_ood_emp_np = dens_ood_emp.detach().cpu().numpy()

            metrics_emp = _compute_metrics(dens_id_emp_np, dens_ood_emp_np)
            runtime_emp = {
                "score_sec": t_score,
                "shift_sec": t_shift,
                "eval_id_sec": t_eval_id_emp,
                "eval_ood_sec": t_eval_ood_emp,
            }

            results["metrics"][backend_name][str(n_train)] = {
                "kde": metrics_kde,
                "emp_sd_kde": metrics_emp,
            }
            results["runtime_sec"][backend_name][str(n_train)] = {
                "kde": runtime_kde,
                "emp_sd_kde": runtime_emp,
            }

            if n_train == curves_n_train and config.save_density_arrays:
                densities_for_curves[backend_name] = {
                    "kde_id": dens_id_np,
                    "kde_ood": dens_ood_np,
                    "emp_id": dens_id_emp_np,
                    "emp_ood": dens_ood_emp_np,
                }

    write_json(run_dir / "results.json", results)

    if config.save_density_arrays and densities_for_curves:
        flat = {}
        for backend_name, density_dict in densities_for_curves.items():
            for key, values in density_dict.items():
                flat[f"{backend_name}_{key}"] = values
        np.savez_compressed(run_dir / "densities_curves.npz", **flat)

    return run_dir


def main() -> None:
    config = MnistFashionOodConfig()
    run_dir = run_benchmark(config)
    print(f"Benchmark complete. Results in {run_dir}")


if __name__ == "__main__":
    main()
