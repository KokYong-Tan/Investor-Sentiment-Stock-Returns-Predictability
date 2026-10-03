"""
regime.py
---------
High- vs. low-VIX regime comparison: two-sample (Welch) t-test on next-month
returns, subgroup size check, and boxplot visualization.
"""

import os
import pandas as pd
from scipy import stats as sps
import matplotlib.pyplot as plt


def regime_ttest(data, regime_col, label, min_recommended_n=30):
    """Welch's two-sample t-test comparing next-month returns across a
    High/Low VIX regime split.

    Returns a dict of summary statistics and test results.
    """
    high = data.loc[data[regime_col] == "high", "next_month_return"]
    low = data.loc[data[regime_col] == "low", "next_month_return"]

    n_high, n_low = len(high), len(low)
    mean_high, mean_low = high.mean(), low.mean()
    sd_high, sd_low = high.std(ddof=1), low.std(ddof=1)

    t_stat, p_val = sps.ttest_ind(high, low, equal_var=False)

    se_high2 = sd_high**2 / n_high
    se_low2 = sd_low**2 / n_low
    welch_df = (se_high2 + se_low2)**2 / (se_high2**2 / (n_high - 1) + se_low2**2 / (n_low - 1))

    crit_t = sps.t.ppf(0.975, welch_df)
    reject = abs(t_stat) > crit_t

    min_n = min(n_high, n_low)
    adequate = min_n >= min_recommended_n

    return {
        "Split": label,
        "N_high": n_high, "N_low": n_low,
        "Mean_high": mean_high, "Mean_low": mean_low,
        "SD_high": sd_high, "SD_low": sd_low,
        "Mean_diff": mean_high - mean_low,
        "t_stat": t_stat, "df": welch_df, "p_value": p_val,
        "reject_h0": reject, "crit_t": crit_t,
        "Min_subgroup_n": min_n, "Subgroup_size_adequate": adequate,
    }


def save_regime_table(results_list, table_dir, filename="regime_comparison.csv"):
    os.makedirs(table_dir, exist_ok=True)
    table = pd.DataFrame(results_list)
    table.to_csv(os.path.join(table_dir, filename), index=False)
    return table


def plot_regime_boxplots(df, fig_dir, filename="regime_comparison_boxplot.png"):
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    plot_specs = [
        (axes[0], "regime_median", "Median VIX Split"),
        (axes[1], "regime_threshold20", "VIX = 20 Threshold Split"),
    ]
    for ax, regime_col, title in plot_specs:
        data_to_plot = [
            df.loc[df[regime_col] == "low", "next_month_return"],
            df.loc[df[regime_col] == "high", "next_month_return"],
        ]
        bp = ax.boxplot(data_to_plot, tick_labels=["Low-VIX", "High-VIX"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["lightsteelblue", "lightsalmon"]):
            patch.set_facecolor(color)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Next-month return")

    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150)
    plt.show()
    return path
