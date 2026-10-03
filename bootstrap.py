"""
bootstrap.py
------------
Moving block bootstrap (following Schmeling, 2009) for the extended model,
resampling full rows (VIX_t, DIV_t, R_{t+1}) jointly to preserve time-series
dependence.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def moving_block_bootstrap_sample(data, block_length, rng):
    """Draw one moving-block-bootstrap resample of `data` (a DataFrame)."""
    n = len(data)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])
    idx = idx[:n]
    return data.iloc[idx].reset_index(drop=True)


def run_block_bootstrap(df, y_col="next_month_return", x_cols=("vix_level", "div_yield"),
                         n_boot=5000, seed=42, block_length=None):
    """Run the moving block bootstrap for the extended model.

    Returns
    -------
    boot_beta1, boot_beta2 : np.ndarray
        Bootstrap replications of beta1 (first x_col) and beta2 (second x_col).
    block_length : int
        The block length actually used.
    """
    x_cols = list(x_cols)
    n = len(df)
    if block_length is None:
        block_length = int(round(n ** (1 / 3)))  # rule of thumb

    rng = np.random.default_rng(seed)
    boot_beta1 = np.empty(n_boot)
    boot_beta2 = np.empty(n_boot)

    for b in range(n_boot):
        sample = moving_block_bootstrap_sample(df, block_length, rng)
        Xb = sm.add_constant(sample[x_cols])
        yb = sample[y_col]
        res_b = sm.OLS(yb, Xb).fit()
        boot_beta1[b] = res_b.params[x_cols[0]]
        boot_beta2[b] = res_b.params[x_cols[1]]

    return boot_beta1, boot_beta2, block_length


def bootstrap_ci(boot_values, alpha_level=0.05):
    """Percentile confidence interval from bootstrap replications."""
    lower = 100 * (alpha_level / 2)
    upper = 100 * (1 - alpha_level / 2)
    return np.percentile(boot_values, [lower, upper])


def save_ci_comparison_table(results_ols, results_hac, ci_beta1, ci_beta2,
                              table_dir, filename="bootstrap_ci_comparison.csv"):
    """Compare OLS, HAC, and bootstrap 95% CIs for beta1 (VIX) and beta2 (DIV)."""
    os.makedirs(table_dir, exist_ok=True)

    ols_ci1 = results_ols.conf_int().loc["vix_level"].values
    ols_ci2 = results_ols.conf_int().loc["div_yield"].values
    hac_ci1 = results_hac.conf_int().loc["vix_level"].values
    hac_ci2 = results_hac.conf_int().loc["div_yield"].values

    table = pd.DataFrame({
        "Coefficient": ["beta1 (VIX)"] * 3 + ["beta2 (DIV)"] * 3,
        "Method": ["OLS", "HAC", "Block Bootstrap"] * 2,
        "CI Lower": [ols_ci1[0], hac_ci1[0], ci_beta1[0], ols_ci2[0], hac_ci2[0], ci_beta2[0]],
        "CI Upper": [ols_ci1[1], hac_ci1[1], ci_beta1[1], ols_ci2[1], hac_ci2[1], ci_beta2[1]],
    })
    table["Contains Zero"] = (table["CI Lower"] <= 0) & (table["CI Upper"] >= 0)
    table.to_csv(os.path.join(table_dir, filename), index=False)
    return table


def plot_bootstrap_distributions(boot_beta1, boot_beta2, ci_beta1, ci_beta2,
                                  point_beta1, point_beta2, fig_dir,
                                  filename="bootstrap_distribution.png"):
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_specs = [
        (axes[0], boot_beta1, ci_beta1, point_beta1, "beta1 (VIX)"),
        (axes[1], boot_beta2, ci_beta2, point_beta2, "beta2 (DIV)"),
    ]
    for ax, boot_vals, ci, point_est, name in plot_specs:
        ax.hist(boot_vals, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(ci[0], color="red", linestyle="--", label="95% Bootstrap CI")
        ax.axvline(ci[1], color="red", linestyle="--")
        ax.axvline(point_est, color="black", linestyle="-", label="OLS point estimate")
        ax.axvline(0, color="gray", linestyle=":", label="0")
        ax.set_title(f"Block-Bootstrap Distribution: {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150)
    plt.show()
    return path
