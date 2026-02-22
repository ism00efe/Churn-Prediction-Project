import pandas as pd


def load_data(path: str):
    """
    Load raw churn dataset.
    """
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame):
    """
    Basic cleaning steps.
    """
    df = df.copy()

    # TotalCharges sometimes stored as string
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing rows
    df = df.dropna()

    return df
