"""
nonparametric.py
-----------------
Permutation test for the regression coefficients (baseline and extended
models). Under H0 (no association between predictors and next-month
returns), the pairing between (VIX_t, DIV_t) and R_{t+1} is arbitrary, so
next_month_return is repeatedly shuffled and the model refit to build an
empirical null distribution for each beta, with no distributional
assumptions required.

Note: this permutes the response (y), not the predictors, so predictor-side
time dependence (e.g. VIX persistence) does not bias the test. The main
assumption is exchangeability of y under H0.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def _perm_pvalue(observed_stat, perm_dist):
    """Two-sided empirical p-value with a +1 continuity correction, so the
    p-value can never be exactly zero regardless of n_perm."""
    return (np.sum(np.abs(perm_dist) >= np.abs(observed_stat)) + 1) / (len(perm_dist) + 1)


def run_permutation_test(df, y_col="next_month_return",
                          x_cols_baseline=("vix_level",),
                          x_cols_extended=("vix_level", "div_yield"),
                          n_perm=5000, seed=42):
    """Run the permutation test for both the baseline and extended models.

    Returns
    -------
    result : dict with keys
        'observed' : dict of observed beta estimates
        'perm_dist' : dict of permutation null distributions (np.ndarray)
        'p_values' : dict of permutation p-values
        'ols_p_values' : dict of OLS p-values, for comparison
    """
    x_cols_baseline = list(x_cols_baseline)
    x_cols_extended = list(x_cols_extended)

    y = df[y_col]
    X_base = sm.add_constant(df[x_cols_baseline])
    X_ext = sm.add_constant(df[x_cols_extended])

    observed_base = sm.OLS(y, X_base).fit()
    observed_ext = sm.OLS(y, X_ext).fit()

    obs_beta_base = observed_base.params[x_cols_baseline[0]]
    obs_beta1 = observed_ext.params[x_cols_extended[0]]
    obs_beta2 = observed_ext.params[x_cols_extended[1]]

    rng = np.random.default_rng(seed)
    y_values = y.values.copy()

    perm_beta_base = np.empty(n_perm)
    perm_beta1 = np.empty(n_perm)
    perm_beta2 = np.empty(n_perm)

    for i in range(n_perm):
        y_shuffled = rng.permutation(y_values)
        perm_beta_base[i] = sm.OLS(y_shuffled, X_base).fit().params[x_cols_baseline[0]]
        res_ext = sm.OLS(y_shuffled, X_ext).fit()
        perm_beta1[i] = res_ext.params[x_cols_extended[0]]
        perm_beta2[i] = res_ext.params[x_cols_extended[1]]

    p_base = _perm_pvalue(obs_beta_base, perm_beta_base)
    p1 = _perm_pvalue(obs_beta1, perm_beta1)
    p2 = _perm_pvalue(obs_beta2, perm_beta2)

    return {
        "observed": {"beta_base": obs_beta_base, "beta1": obs_beta1, "beta2": obs_beta2},
        "perm_dist": {"beta_base": perm_beta_base, "beta1": perm_beta1, "beta2": perm_beta2},
        "p_values": {"beta_base": p_base, "beta1": p1, "beta2": p2},
        "ols_p_values": {
            "beta_base": observed_base.pvalues[x_cols_baseline[0]],
            "beta1": observed_ext.pvalues[x_cols_extended[0]],
            "beta2": observed_ext.pvalues[x_cols_extended[1]],
        },
    }


def save_permutation_table(result, table_dir, filename="permutation_test.csv"):
    os.makedirs(table_dir, exist_ok=True)
    table = pd.DataFrame({
        "Coefficient": ["beta (baseline, VIX)", "beta1 (extended, VIX)", "beta2 (extended, DIV)"],
        "Observed Estimate": [
            result["observed"]["beta_base"], result["observed"]["beta1"], result["observed"]["beta2"]
        ],
        "Permutation p-value": [
            result["p_values"]["beta_base"], result["p_values"]["beta1"], result["p_values"]["beta2"]
        ],
        "OLS p-value": [
            result["ols_p_values"]["beta_base"], result["ols_p_values"]["beta1"], result["ols_p_values"]["beta2"]
        ],
    })
    table.to_csv(os.path.join(table_dir, filename), index=False)
    return table


def plot_permutation_distributions(result, fig_dir, filename="permutation_distribution.png"):
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    plot_specs = [
        (axes[0], "beta_base", "beta (baseline, VIX)"),
        (axes[1], "beta1", "beta1 (extended, VIX)"),
        (axes[2], "beta2", "beta2 (extended, DIV)"),
    ]
    for ax, key, name in plot_specs:
        perm_dist = result["perm_dist"][key]
        obs_val = result["observed"][key]
        p_val = result["p_values"][key]

        ax.hist(perm_dist, bins=50, color="lightgray", edgecolor="white")
        ax.axvline(obs_val, color="red", linewidth=1.5, label=f"Observed = {obs_val:.5f}")
        ax.set_title(f"{name}\n(p = {p_val:.4f})")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150)
    plt.show()
    return path
