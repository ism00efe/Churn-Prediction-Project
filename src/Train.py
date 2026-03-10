import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import pandas as pd

from src.data_loader import load_data
from src.Data import clean_data
from src.features import build_features
import src.config as config 

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, fbeta_score

from src import config
from src.data_loader import load_processed_data, get_train_test_split
from src.features import build_preprocessor

# 1. Veriyi Yükle
raw_df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Yapısal Temizlik (Data.py)
cleaned_df = clean_data(raw_df, config.COLUMNS_TO_DROP)

# 3. Model İçin Özellik Mühendisliği (features.py)
final_df = build_features(cleaned_df, config)

# Tüm işlemler (clean_data ve build_features) bittikten sonra:
final_df.to_csv("data/processed/cleaned_data.csv", index=False)

def main():
    print("Veri yükleniyor...")
    df = load_processed_data(config.PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print("Pipeline oluşturuluyor...")
    preprocessor = build_preprocessor(X_train)
    
    pipe_log = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=config.RANDOM_STATE))
    ])

    # F2 Scorer (Recall ağırlıklı)
    f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10], 
        'clf__class_weight': [None, 'balanced', {0: 1, 1: 1.5}, {0: 1, 1: 2}] 
    }

    print("Hiperparametre optimizasyonu (GridSearchCV) başlatılıyor...")
    grid_search = GridSearchCV(pipe_log, param_grid, cv=5, scoring=f2_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"En iyi parametreler: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    print("Model kalibre ediliyor...")
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)

    print(f"Model diske kaydediliyor -> {config.MODEL_SAVE_PATH}")
    # Klasör yoksa oluştur
    config.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, config.MODEL_SAVE_PATH)
    
    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    main()
