# eda.py
import io
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV file uploaded via Streamlit into a pandas DataFrame.
    """
    # uploaded_file is a SpooledTemporaryFile-like object
    return pd.read_csv(uploaded_file)


def get_basic_info(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return textual info about shape, columns and dtypes.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    shape_str = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    return shape_str, info_str


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing values per column.
    """
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values.round(2),
    })
    missing_df = missing_df.sort_values("missing_count", ascending=False)
    return missing_df.reset_index(drop=True)


def get_duplicate_count(df: pd.DataFrame) -> int:
    """
    Count duplicate rows in the dataset.
    """
    return df.duplicated().sum()


def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for numeric columns.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T


def plot_correlation_heatmap(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Create a correlation heatmap for numeric columns.
    Returns a matplotlib Figure or None if correlation can't be computed.
    """
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    ax.set_title("Correlation Heatmap", pad=20)

    plt.tight_layout()
    return fig
