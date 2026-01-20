from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

# Global string identifiers
PRECISION_FAST_TF32 = "fast_tf32"
PRECISION_FP32_IEEE = "fp32_ieee"

KDE_BACKEND_SPLITK_STREAM = "splitk_stream"
KDE_BACKEND_ATOMIC = "atomic"

EMP_SCORE_BACKEND_ORDERED_SPLITK = "ordered_splitk"
EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC = "symmetric_atomic"

EMP_SD_KDE_VARIANT_EXACT = "emp_sd_kde_exact"
EMP_SD_KDE_VARIANT_LINEARIZED = "emp_sd_kde_linearized"

DATASET_MNIST = "mnist"
DATASET_FASHION_MNIST = "fashion_mnist"

BACKEND_FLASH_SPLITK_SYM = "flash_splitk_sym"
BACKEND_FLASH_SPLITK_ORDERED = "flash_splitk_ordered"
BACKEND_FLASH_SPLITK_SYM_LINEARIZED = "flash_splitk_sym_linearized"
BACKEND_NON_FLASH_ATOMIC_SYM = "non_flash_atomic_sym"
BACKEND_NON_FLASH_ATOMIC_ORDERED = "non_flash_atomic_ordered"

# Global numeric constants
ND_FEATURES = 16
DEFAULT_EPS = 1e-12
FILE_STORAGE_ROOT = "file_storage"

# Default knobs
DEFAULT_PRECISION_MODE = PRECISION_FAST_TF32
DEFAULT_KDE_BACKEND = KDE_BACKEND_SPLITK_STREAM
DEFAULT_EMP_SCORE_BACKEND = EMP_SCORE_BACKEND_SYMMETRIC_ATOMIC
