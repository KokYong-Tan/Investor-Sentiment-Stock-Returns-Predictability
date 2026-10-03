"""
descriptive.py
--------------
Descriptive analysis: time series plot, histogram/Q-Q plot, scatter plot,
summary statistics table, and VIX-dividend-yield multicollinearity check.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def plot_timeseries(df, fig_dir, filename="timeseries_returns_vix.png"):
    """Dual-axis time series plot of monthly returns and VIX."""
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["month_end_date"], df["price_return"], color="tab:blue",
             linewidth=0.9, label="S&P 500 monthly return")
    ax1.set_ylabel("Monthly return", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(df["month_end_date"], df["vix_level"], color="tab:red",
             linewidth=1.1, label="VIX")
    ax2.set_ylabel("VIX level", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("S&P 500 Monthly Returns and VIX")
    fig.tight_layout()
    path = os.path.join(fig_dir, filename)
    plt.savefig(path, dpi=150)
    plt.show()
    return path


def plot_histogram_qq(df, fig_dir, filename="returns_histogram_qqplot.png"):
    """Return histogram + Q-Q plot, plus a Shapiro-Wilk normality test.

    Returns
    -------
    sw_stat, sw_p : float
        Shapiro-Wilk test statistic and p-value.
    """
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(df["price_return"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Histogram of Monthly Returns")
    axes[0].set_xlabel("Monthly return")
    axes[0].set_ylabel("Frequency")

    stats.probplot(df["price_return"], dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot vs Normal Distribution")

    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=150)
    plt.show()

    sw_stat, sw_p = stats.shapiro(df["price_return"])
    return sw_stat, sw_p


def plot_scatter(df, fig_dir, filename="vix_vs_next_return_scatter.png"):
    """Scatter of VIX_t vs. next-month return, with a simple linear fit line."""
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["vix_level"], df["next_month_return"], alpha=0.6,
               color="darkorange", edgecolor="k", linewidth=0.3)

    z = np.polyfit(df["vix_level"], df["next_month_return"], 1)
    x_line = np.linspace(df["vix_level"].min(), df["vix_level"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="black", linestyle="--",
            linewidth=1.2, label=f"Fit: y = {z[1]:.4f} + {z[0]:.5f}x")

    ax.set_xlabel("VIX (month t)")
    ax.set_ylabel("S&P 500 return (month t+1)")
    ax.set_title("VIX vs. Next-Month S&P 500 Return")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=150)
    plt.show()


def summary_statistics_table(df, table_dir, filename="summary_statistics.csv"):
    """Mean/SD/min/max table for next-month return, VIX, and dividend yield."""
    os.makedirs(table_dir, exist_ok=True)
    summary_vars = df[["next_month_return", "vix_level", "div_yield"]].rename(columns={
        "next_month_return": "Next-Month Return",
        "vix_level": "VIX",
        "div_yield": "Dividend Yield",
    })
    summary_table = summary_vars.agg(["count", "mean", "std", "min", "max"]).T
    summary_table.columns = ["N", "Mean", "SD", "Min", "Max"]
    summary_table.to_csv(os.path.join(table_dir, filename))
    return summary_table


def vif_check(df, table_dir, filename="vif_table.csv"):
    """Correlation and VIF between VIX and dividend yield.

    Returns
    -------
    corr : float
    vif_data : pd.DataFrame
    """
    os.makedirs(table_dir, exist_ok=True)
    corr = df[["vix_level", "div_yield"]].corr().iloc[0, 1]

    X = df[["vix_level", "div_yield"]].copy()
    X_const = add_constant(X)
    vif_data = pd.DataFrame({
        "Variable": X_const.columns,
        "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
    })
    vif_data = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)
    vif_data.to_csv(os.path.join(table_dir, filename), index=False)
    return corr, vif_data
