# train.py
# ---------------------------------------------------------------
# Trains a Random Forest model to predict if a student will PASS
# or FAIL based on study habits and background.
# Logs everything to MLflow so you can compare experiments.
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# ── 1. LOAD DATA ────────────────────────────────────────────────
# The UCI Student Performance dataset (Math course)
# Source: https://archive.ics.uci.edu/ml/datasets/Student+Performance
# We download it directly so the pipeline works on any machine.
DATA_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Student-Performance-Dataset/master/student-mat.csv"
)
# Replace the DATA_URL lines with this:
print("📥 Loading dataset...")
df = pd.read_csv("Data/student-mat.csv", sep=";", encoding="latin1")


# ── 2. CREATE TARGET COLUMN ─────────────────────────────────────
# G3 is the final grade (0-20).
# We convert it to binary: PASS (>=10) or FAIL (<10)
df["pass"] = (df["G3"] >= 10).astype(int)

# Drop the three grade columns — we don't want the model
# to "cheat" by seeing intermediate grades
df = df.drop(columns=["G1", "G2", "G3"])

# ── 3. ENCODE CATEGORICAL COLUMNS ───────────────────────────────
# Computers can't understand text like "yes"/"no" or "GP"/"MS"
# LabelEncoder converts them to numbers (0 and 1, etc.)
categorical_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ── 4. SPLIT DATA ───────────────────────────────────────────────
X = df.drop(columns=["pass"])
y = df["pass"]

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Data ready — Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ── 5. TRAIN WITH MLFLOW TRACKING ───────────────────────────────
# MLflow records every experiment run so you can compare them
# Run `mlflow ui` in your terminal to see the dashboard

mlflow.set_experiment("student-grade-predictor")

# Hyperparameters — try changing these and re-running!
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42

with mlflow.start_run():

    # --- Train model ---
    print("🚀 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n📊 Results:")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Fail','Pass'])}")

    # --- Log parameters to MLflow ---
    # "Parameters" = settings you chose before training
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # --- Log metrics to MLflow ---
    # "Metrics" = results after training
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # --- Save confusion matrix as image ---
    os.makedirs("Results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"]
    )
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2%})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("Results/model_results.png")
    plt.close()

    # Log the image to MLflow too
    mlflow.log_artifact("Results/model_results.png")

    # --- Save the trained model ---
    os.makedirs("Model", exist_ok=True)
    joblib.dump(model, "Model/model.pkl")

    # Also log model to MLflow registry
    mlflow.sklearn.log_model(model, "random-forest-model")

    print("\n✅ Model saved to Model/model.pkl")
    print("✅ Results saved to Results/model_results.png")
    print("✅ All metrics logged to MLflow")
    print("\n💡 Run `mlflow ui` in your terminal to view the dashboard!")