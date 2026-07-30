import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_credit_data(data_path: str = "data/german_credit.csv") -> pd.DataFrame:
    """
    Downloads and loads the German Credit Dataset into a Pandas DataFrame.
    """
    if not os.path.exists(data_path):
        print("[INFO] Dataset not found locally. Downloading from remote repository...")
        url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
        df = pd.read_csv(url)

        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"[SUCCESS] Dataset saved locally at: {data_path}")
    else:
        print(f"[INFO] Loading dataset from local path: {data_path}")
        df = pd.read_csv(data_path)

    return df


def preprocess_data(df: pd.DataFrame):
    """
    Cleans target labels, encodes categorical features, and splits into train/test sets.
    """
    # Create a copy to avoid modifying original dataframe
    df = df.copy()

    # Identify and map target column dynamically
    # Target standard: 1 = Credit Default (High Risk), 0 = Good Standing
    if "Creditability" in df.columns:
        # In GermanCredit.csv, 1 = Good, 0 = Bad. We flip so 1 = Default Risk.
        df["target"] = df["Creditability"].apply(lambda x: 1 if x == 0 else 0)
        df = df.drop(columns=["Creditability"])
    elif "default" in df.columns:
        df["target"] = df["default"]
        df = df.drop(columns=["default"])
    elif "default.payment.next.month" in df.columns:
        df["target"] = df["default.payment.next.month"]
        df = df.drop(columns=["default.payment.next.month"])
    else:
        # Fallback: Assume the last column in dataset is the target variable
        last_col = df.columns[-1]
        print(f"[WARNING] Standard target column not found. Defaulting to last column: '{last_col}'")
        df["target"] = df[last_col]
        df = df.drop(columns=[last_col])

    # Separate feature matrix (X) and target vector (y)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Encode non-numeric / categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Data Split Completed successfully.")
    print(f"       - Training Samples : {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"       - Testing Samples  : {X_test.shape[0]} rows")
    print(f"       - Default Rate     : {np.mean(y_train):.2%}")

    return X_train, X_test, y_train, y_test, label_encoders


if __name__ == "__main__":
    data = load_credit_data()
    X_train, X_test, y_train, y_test, encoders = preprocess_data(data)