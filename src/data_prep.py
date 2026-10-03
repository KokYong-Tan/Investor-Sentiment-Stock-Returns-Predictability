"""
data_prep.py
------------
Functions for loading, cleaning, merging, and lagging the S&P 500 / VIX data
into the analysis-ready monthly dataset used throughout the project.
"""

import os
import numpy as np
import pandas as pd


def load_raw_data(raw_dir):
    """Load the raw S&P 500 and VIX CSV files from `raw_dir`.

    Returns
    -------
    sp500 : pd.DataFrame
    vix : pd.DataFrame
    """
    sp500 = pd.read_csv(os.path.join(raw_dir, "SP_500_Index-2001-2025.csv"))
    vix = pd.read_csv(os.path.join(raw_dir, "VIX-2001-2025.csv"))
    return sp500, vix


def clean_sp500(sp500):
    """Clean the raw CRSP S&P 500 file and construct dividend yield.

    Returns a DataFrame with columns:
        year_month, month_end_date, price_return, total_return, div_yield
    """
    sp500 = sp500.copy()
    sp500["MthCalDt"] = pd.to_datetime(sp500["MthCalDt"])
    sp500 = sp500.sort_values("MthCalDt").reset_index(drop=True)
    sp500["year_month"] = sp500["MthCalDt"].dt.to_period("M")

    sp500_clean = sp500[["year_month", "MthCalDt", "MthPrcRet", "MthTotRet"]].copy()
    sp500_clean = sp500_clean.rename(columns={
        "MthCalDt": "month_end_date",
        "MthPrcRet": "price_return",
        "MthTotRet": "total_return",
    })

    # Dividend yield control: DIV_t = TotalReturn_t - PriceReturn_t
    sp500_clean["div_yield"] = sp500_clean["total_return"] - sp500_clean["price_return"]

    assert sp500_clean["year_month"].duplicated().sum() == 0, "Duplicate months found!"
    assert sp500_clean.isna().sum().sum() == 0, "Unexpected missing values in S&P 500 data!"

    return sp500_clean


def clean_and_resample_vix(vix):
    """Forward-fill missing daily VIX values and resample to month-end.

    Returns a DataFrame with columns: year_month, vix_date, vix_level
    """
    vix = vix.copy()
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.sort_values("date").reset_index(drop=True)

    vix["vix"] = vix["vix"].ffill()

    vix["year_month"] = vix["date"].dt.to_period("M")
    vix_month_end = (
        vix.sort_values("date")
        .groupby("year_month")
        .last()
        .reset_index()
        .rename(columns={"date": "vix_date", "vix": "vix_level"})
        [["year_month", "vix_date", "vix_level"]]
    )
    return vix_month_end


def merge_and_lag(sp500_clean, vix_month_end):
    """Merge the cleaned S&P 500 and VIX series and construct next_month_return.

    VIX_t and DIV_t (this month's values) are aligned to predict R_{t+1}
    (next month's price return), so the last row (no next-month return
    available) is dropped.
    """
    merged = pd.merge(sp500_clean, vix_month_end, on="year_month", how="inner")
    merged = merged.sort_values("year_month").reset_index(drop=True)

    merged["next_month_return"] = merged["price_return"].shift(-1)
    merged_final = merged.dropna(subset=["next_month_return"]).reset_index(drop=True)
    return merged_final


def add_regime_indicators(df, threshold=20):
    """Add median-split and fixed-threshold High/Low VIX regime columns."""
    df = df.copy()
    median_vix = df["vix_level"].median()
    df["regime_median"] = np.where(df["vix_level"] >= median_vix, "high", "low")
    df["regime_threshold20"] = np.where(df["vix_level"] >= threshold, "high", "low")
    return df


def build_dataset(raw_dir, interim_dir=None, processed_dir=None, threshold=20):
    """Run the full data preparation pipeline end to end.

    Parameters
    ----------
    raw_dir : str
        Folder containing the raw CSV files.
    interim_dir : str, optional
        If provided, saves the resampled month-end VIX series here.
    processed_dir : str, optional
        If provided, saves the final merged_monthly.csv here.
    threshold : float
        VIX threshold for the fixed-threshold regime split.

    Returns
    -------
    merged_final : pd.DataFrame
        The analysis-ready monthly dataset.
    """
    sp500, vix = load_raw_data(raw_dir)
    sp500_clean = clean_sp500(sp500)
    vix_month_end = clean_and_resample_vix(vix)

    if interim_dir is not None:
        os.makedirs(interim_dir, exist_ok=True)
        vix_month_end.to_csv(os.path.join(interim_dir, "vix_month_end.csv"), index=False)

    merged_final = merge_and_lag(sp500_clean, vix_month_end)
    merged_final = add_regime_indicators(merged_final, threshold=threshold)

    if processed_dir is not None:
        os.makedirs(processed_dir, exist_ok=True)
        merged_final.to_csv(os.path.join(processed_dir, "merged_monthly.csv"), index=False)

    return merged_final
