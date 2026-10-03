"""
models.py
---------
Baseline (VIX-only) and extended (VIX + dividend yield) OLS regressions,
including HAC (Newey-West) standard errors and the baseline-vs-extended
comparison table.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps


def fit_baseline(df):
    """Fit R_{t+1} = alpha + beta * VIX_t + eps_t.

    Returns the fitted statsmodels results object.
    """
    y = df["next_month_return"]
    X = sm.add_constant(df["vix_level"])
    return sm.OLS(y, X).fit()


def baseline_hypothesis_test(results, alpha_level=0.05):
    """Two-sided t-test of H0: beta = 0 for the baseline model.

    Returns a dict with the key statistics and a plain-language conclusion.
    """
    beta = results.params["vix_level"]
    se_beta = results.bse["vix_level"]
    t_beta = results.tvalues["vix_level"]
    p_beta = results.pvalues["vix_level"]
    dof = int(results.df_resid)  # n - 2

    crit_t = sps.t.ppf(1 - alpha_level / 2, dof)
    reject = abs(t_beta) > crit_t

    return {
        "beta": beta, "se": se_beta, "t_stat": t_beta, "p_value": p_beta,
        "df": dof, "crit_t": crit_t, "reject_h0": reject,
    }


def save_baseline_table(results, table_dir, filename="baseline_regression.csv"):
    os.makedirs(table_dir, exist_ok=True)
    table = pd.DataFrame({
        "Coefficient": ["alpha (Intercept)", "beta (VIX)"],
        "Estimate": [results.params["const"], results.params["vix_level"]],
        "Std. Error": [results.bse["const"], results.bse["vix_level"]],
        "t-stat": [results.tvalues["const"], results.tvalues["vix_level"]],
        "p-value": [results.pvalues["const"], results.pvalues["vix_level"]],
    })
    table.to_csv(os.path.join(table_dir, filename), index=False)
    return table


def fit_extended(df):
    """Fit R_{t+1} = alpha + beta1*VIX_t + beta2*DIV_t + eps_t.

    Returns (results_ols, results_hac) — the same model fit with ordinary
    and HAC (Newey-West) standard errors respectively.
    """
    y = df["next_month_return"]
    X = sm.add_constant(df[["vix_level", "div_yield"]])
    model = sm.OLS(y, X)

    results_ols = model.fit()

    n = len(df)
    maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))  # Newey-West rule of thumb
    results_hac = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    return results_ols, results_hac


def extended_hypothesis_tests(results_ols, results_hac, alpha_level=0.05):
    """Individual t-tests for beta1, beta2 (OLS and HAC) plus the joint F-test.

    Returns a dict of results.
    """
    dof = int(results_ols.df_resid)  # n - 3
    crit_t = sps.t.ppf(1 - alpha_level / 2, dof)

    out = {"df": dof, "crit_t": crit_t}
    for name, col in [("beta1", "vix_level"), ("beta2", "div_yield")]:
        out[name] = {
            "estimate": results_ols.params[col],
            "se_ols": results_ols.bse[col],
            "se_hac": results_hac.bse[col],
            "t_ols": results_ols.tvalues[col],
            "t_hac": results_hac.tvalues[col],
            "p_ols": results_ols.pvalues[col],
            "p_hac": results_hac.pvalues[col],
            "reject_ols": abs(results_ols.tvalues[col]) > crit_t,
            "reject_hac": abs(results_hac.tvalues[col]) > crit_t,
        }

    out["f_stat"] = results_ols.fvalue
    out["f_pvalue"] = results_ols.f_pvalue
    out["reject_joint"] = results_ols.f_pvalue < alpha_level
    out["adj_r2"] = results_ols.rsquared_adj
    return out


def save_extended_comparison_table(baseline_table, results_ols, results_hac,
                                    table_dir, filename="baseline_vs_extended.csv"):
    """Build and save the baseline-vs-extended coefficient comparison table."""
    os.makedirs(table_dir, exist_ok=True)

    comparison = pd.DataFrame({
        "Coefficient": ["alpha (Intercept)", "beta_VIX", "beta_DIV"],
        "Baseline Estimate": [
            baseline_table.loc[baseline_table["Coefficient"] == "alpha (Intercept)", "Estimate"].values[0],
            baseline_table.loc[baseline_table["Coefficient"] == "beta (VIX)", "Estimate"].values[0],
            np.nan,
        ],
        "Baseline SE": [
            baseline_table.loc[baseline_table["Coefficient"] == "alpha (Intercept)", "Std. Error"].values[0],
            baseline_table.loc[baseline_table["Coefficient"] == "beta (VIX)", "Std. Error"].values[0],
            np.nan,
        ],
        "Extended Estimate": [
            results_ols.params["const"], results_ols.params["vix_level"], results_ols.params["div_yield"]
        ],
        "Extended SE (OLS)": [
            results_ols.bse["const"], results_ols.bse["vix_level"], results_ols.bse["div_yield"]
        ],
        "Extended SE (HAC)": [
            results_hac.bse["const"], results_hac.bse["vix_level"], results_hac.bse["div_yield"]
        ],
    })
    comparison.to_csv(os.path.join(table_dir, filename), index=False)
    return comparison
