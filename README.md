# Churn Prediction Project

This repository contains an end-to-end customer churn prediction workflow built on the Telco Customer Churn dataset.  
The project covers data cleaning, feature engineering, model training with hyperparameter optimization, probability calibration, and two serving interfaces:

- A REST API with FastAPI
- An interactive UI with Streamlit

## Project Scope

Primary objective: estimate customer churn probability and support retention decision-making with threshold-based risk classification.

Implemented capabilities:

- Data preprocessing and cleaning pipeline
- Structured feature engineering for mixed categorical/numeric data
- Logistic Regression training with `GridSearchCV` and `F2`-focused scoring
- Probability calibration with isotonic calibration
- Model persistence to disk (`joblib`)
- Online inference via FastAPI and Streamlit

## Repository Structure

```text
.
├── App/
│   ├── Main.py              # FastAPI service
│   └── app.py               # Streamlit interface
├── Models/
│   └── Model.pkl            # Trained calibrated model artifact
├── Nootbooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03.ipynb
├── data/
│   ├── Raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── cleaned_data.csv
├── src/
│   ├── config.py            # Constants, paths, business scenario parameters
│   ├── Data.py              # Data cleaning
│   ├── features.py          # Feature engineering + preprocessing
│   ├── data_loader.py       # IO and train/test split helpers
│   ├── Train.py             # Training entry point
│   └── evaluate.py          # Confusion-matrix and net-profit utilities
├── requirements.txt
└── pyproject.toml
```

## Dataset

- **Dataset**: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Task**: Binary classification (`Churn`)
- **Target column**: `Churn`
- **Current local location**: `data/Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Cleaning and preparation steps

Data processing currently applies the following logic:

1. Convert `TotalCharges` to numeric (`TotalCharges_num`) with coercion
2. Drop rows where `TotalCharges_num` is null
3. Drop rows where `tenure == 0`
4. Remove configured columns (`customerID`, `TotalCharges`)
5. Encode binary fields (`Yes/No`, `gender`) to numeric
6. Encode ordinal-like categorical fields (`Contract`, `MultipleLines`)
7. One-hot encode selected columns (`InternetService`, `PaymentMethod`)
8. Remove low-correlation columns (threshold from config)
9. Drop configured multicollinearity columns
10. Optimize dtypes and export processed dataset

## Modeling Pipeline

Training pipeline is implemented in `src/Train.py`.

- **Base estimator**: `LogisticRegression`
- **Preprocessing**: `ColumnTransformer` + `StandardScaler` for non-binary numeric columns
- **Hyperparameter search**: `GridSearchCV` (5-fold CV)
- **Optimization metric**: `F-beta` with `beta=2` (`F2`), favoring recall
- **Calibration**: `CalibratedClassifierCV(method="isotonic", cv=5)`
- **Saved artifact**: `Models/Model.pkl`

## Model Performance

Final test metrics (from `Nootbooks/02_modeling.ipynb`, tuning sonrası):

| Metric | Value |
| :--- | ---: |
| Accuracy | 0.7235 |
| Precision (Churn=1) | 0.4876 |
| Recall (Churn=1) | 0.7914 |
| F1 Score (Churn=1) | 0.6035 |
| F2 Score (Churn=1) | 0.7038 |

### Hyperparameters searched

- `clf__C`: `[0.01, 0.1, 1, 10]`
- `clf__class_weight`:
  - `None`
  - `"balanced"`
  - `{0: 1, 1: 1.5}`
  - `{0: 1, 1: 2}`

## Business Evaluation Utilities

`src/evaluate.py` includes helpers for decision-threshold and business impact analysis:

- `calculate_confusion_matrix_metrics(...)`
- `calculate_net_profit(...)`

Scenario parameters (`v_cost`, `c_cost`, `r_rate`, `negative_impact_rate`) are defined in `src/config.py`.

## Installation

### 1) Clone repository

```bash
git clone <your-repository-url>
cd Churn-Prediction-Project-1
```

### 2) Create virtual environment

```bash
python -m venv .venv
```

Activate environment:

- **Windows (PowerShell)**:

```powershell
.venv\Scripts\Activate.ps1
```

- **Linux / macOS**:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Training Workflow

Run model training:

```bash
python -m src.Train
```

What this run does:

- Reads raw data
- Applies cleaning + feature engineering
- Writes processed file to `data/processed/cleaned_data.csv`
- Trains and calibrates model
- Saves model to `Models/Model.pkl`

## Run FastAPI Service

Start API server:

```bash
uvicorn App.Main:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:

- `GET /`
- `GET /health`
- `POST /predict`
- Interactive docs: `http://127.0.0.1:8000/docs`

### Example prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 24,
    "OnlineBackup": 1,
    "DeviceProtection": 1,
    "OnlineSecurity": 1,
    "TechSupport": 0,
    "Contract": 1,
    "PaperlessBilling": 1,
    "MonthlyCharges": 74.5,
    "InternetService_DSL": 0,
    "InternetService_Fiber optic": 1,
    "InternetService_No": 0,
    "PaymentMethod_Bank transfer (automatic)": 0,
    "PaymentMethod_Credit card (automatic)": 1,
    "PaymentMethod_Electronic check": 0
  }'
```

Expected response format:

```json
{
  "churn_probability": 0.4123,
  "will_churn": true,
  "applied_threshold": 0.4
}
```

## Run Streamlit App

Start local UI:

```bash
streamlit run App/app.py
```

The Streamlit app loads `Models/Model.pkl`, collects feature inputs from the form, and displays churn risk according to the current threshold (`0.4`).

## Reproducibility Notes

- `RANDOM_STATE` is set to `42`
- Test split uses stratified split with `test_size=0.2`
- Paths for processed data and model output are centralized in `src/config.py`
- Training should be run before API/UI inference if `Models/Model.pkl` is missing

## Known Constraints

- Folder naming is currently `Nootbooks/` (intentional as in repository state)
- Raw dataset is referenced in code as `data/raw/...` but repository folder is `data/Raw/...`; use consistent casing in your environment to avoid path issues
- `requirements.txt` includes broad notebook/UI dependencies in addition to training/runtime packages

## Future Improvements

- Add `pytest` coverage for cleaning, feature engineering, and API schema validation
- Add model/version metadata and experiment tracking
- Add Dockerfile and containerized run flow for API and Streamlit
- Add CI pipeline (lint, test, train smoke test)
- Add threshold tuning report and calibrated probability diagnostics
