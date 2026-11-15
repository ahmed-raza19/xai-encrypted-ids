"""Paths, seeds and global experiment constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../Iteration_3
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "results" / "models"

for p in (DATA_PROC, RESULTS, FIGURES, MODELS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
SEEDS = [42, 43, 44, 45, 46]

# Time-ordered split fractions (Phase 2, Step 3.6)
SPLIT = (0.60, 0.20, 0.20)

# Per-dataset row caps after cleaning (keeps every attack row where possible;
# benign rows are uniformly subsampled). Needed for a 16 GB CPU-only laptop.
MAX_ROWS = {
    "dohbrw": 300_000,
    "hikari": 300_000,
    "cic2017": 400_000,
    "cic2018": 400_000,
}

# Explanation-cost protocol (Phase 2, Step 7 / E1)
N_EXPLAIN = 200          # flows explained per configuration (100 attack + 100 benign)
N_WARMUP = 10
BACKGROUND_K = 50        # k-means summarised background for KernelSHAP / DeepSHAP
BUDGETS = {
    "kernelshap": [100, 500, 2048],   # coalition samples
    "lime": [500, 1000, 5000],        # perturbations
    "ig": [16, 32, 64],               # integration steps
}

# Selective trigger defaults (Phase 2, Step 5)
TRIGGER = {
    "fpr_target": 0.01,      # alert threshold tau chosen for FPR <= 1% on validation
    "band_frac": 0.05,       # uncertainty band contains 5% of validation flows
    "novelty_q": 0.99,       # Mahalanobis novelty threshold quantile
    "audit_rate": 0.005,     # benign audit sampling rate rho
    "beta": 0.10,            # synchronous overhead budget fraction
    "bucket_s": 0.05,        # token-bucket capacity (s); Phase 2 proposed 2 s, which never binds at
                             # sub-millisecond detection cost (see Phase 3 methodology changes)
    "cache_ttl_s": 60.0,
    "cache_top_m": 10,       # features used in the cache key
    "deadline_fast_s": 1.0,
    "deadline_deep_s": 30.0,
}
