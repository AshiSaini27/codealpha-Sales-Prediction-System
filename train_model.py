# =====================================================
# CAR PRICE PREDICTION
# Part 1 : Load Dataset
# =====================================================
import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("dataset/car data.csv")

# First 5 rows
print(df.head())

# Dataset information
print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nUnique Fuel Types:")
print(df["Fuel_Type"].unique())

print("\nUnique Selling Types:")
print(df["Selling_type"].unique())

print("\nUnique Transmission Types:")
print(df["Transmission"].unique())

# =====================================================
# CAR PRICE PREDICTION
# Part 2 : Data Preprocessing
# =====================================================

import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/car data.csv")

print("Original Dataset Shape:", df.shape)

# -----------------------------------------------------
# Create Car Age Feature
# -----------------------------------------------------

current_year = 2025
df["Car_Age"] = current_year - df["Year"]

# Drop unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

print("\nDataset after Feature Engineering:")
print(df.head())

# -----------------------------------------------------
# One-Hot Encoding
# -----------------------------------------------------

df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

print("\nDataset after Encoding:")
print(df.head())

print("\nFinal Dataset Shape:", df.shape)

# -----------------------------------------------------
# Features and Target
# -----------------------------------------------------

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Names:")
print(X.columns)

# =====================================================
# CAR PRICE PREDICTION
# Part 3 : Model Training
# =====================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("dataset/car data.csv")

# =====================================================
# FEATURE ENGINEERING
# =====================================================

current_year = 2025
df["Car_Age"] = current_year - df["Year"]

df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# =====================================================
# ENCODING
# =====================================================

df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

# Convert True/False to 1/0
df = df.astype(int, errors="ignore")

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# MODEL TRAINING
# =====================================================

model = LinearRegression()

model.fit(X_train_scaled, y_train)

# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(X_test_scaled)

# =====================================================
# EVALUATION
# =====================================================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n========== MODEL PERFORMANCE ==========")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved successfully!")

# =====================================================
# CREATE IMAGES FOLDER
# =====================================================

import os

os.makedirs("images", exist_ok=True)

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Price")

plt.savefig("images/prediction.png")

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.coef_

feature_names = X.columns

plt.figure(figsize=(10,6))

plt.barh(feature_names, importance)

plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("images/feature_importance.png")

plt.show()

print("\nGraphs saved successfully!")