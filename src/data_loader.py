import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, TEST_SIZE, RANDOM_STATE

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def load_processed_data(filepath):
    """Temizlenmiş veriyi yükler."""
    return pd.read_csv(filepath)

def get_train_test_split(df):
    """Veriyi özellikler ve hedef değişken olarak ayırır, train/test split yapar."""
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test