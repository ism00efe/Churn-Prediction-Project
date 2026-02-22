import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data import load_data, clean_data


def main():
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    df = clean_data(df)

    # Target
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID"])

    # Categorical / Numerical split
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
