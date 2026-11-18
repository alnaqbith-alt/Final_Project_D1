"""
Dubai Real Estate Investment Prediction Project
"""

from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent

# Find the folder containing the CSV dataset.
possible_data_folders = [
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
    SCRIPT_DIR / "CDS 4003",
    SCRIPT_DIR.parent / "CDS 4003"
]

DATA_DIR = None
for folder in possible_data_folders:
    if (folder / "Training Dataset.csv").exists():
        DATA_DIR = folder
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find the dataset file. Place this Python script in the "
        "same folder as 'Training Dataset.csv', "
        "or keep the dataset in the parent folder."
    )

os.chdir(DATA_DIR)
print("Dataset/output folder:", DATA_DIR)


# Import Libraries


# Part 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_validate
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)


# Load the Dataset


# Part 2: Load the Dataset

train = pd.read_csv(DATA_DIR / "Training Dataset.csv", low_memory=False)

# Standardize column names if original DLD names are present
rename_columns = {
    "instance_date": "Transaction_Date",
    "area_name_en": "Area_Name",
    "procedure_name_en": "Procedure_Name"
}
train = train.rename(columns=rename_columns)

print("Dataset shape:", train.shape)

print("\nFirst five rows:")
print(train.head())

print("\nDataset information:")
train.info()


# Data Understanding

print("Statistical Summary:")
print(train.describe(include="all").to_string())

print("\nColumn Names:")
print(train.columns.tolist())

print("\nTarget Distribution:")
print(train["Result"].value_counts())

print("\nTarget Percentages:")
print(train["Result"].value_counts(normalize=True) * 100)


# Check Data Quality

print("Missing Values in Dataset:")
print(train.isnull().sum())

print("\nDuplicate Rows:")
print("Duplicates:", train.duplicated().sum())

print("\nData Types:")
print(train.dtypes)

quality = pd.DataFrame({
    "Column": train.columns,
    "Missing": train.isnull().sum().values,
    "Missing_Percent": (
        train.isnull().mean().values * 100
    ).round(2),
    "Unique_Values": train.nunique().values
})

print("\nData Quality Summary:")
print(quality)

quality.to_csv("data_quality.csv", index=False)


# Data Cleaning


# Part 5: Data Cleaning

train_clean = train.copy()

# Remove duplicate records
train_clean = train_clean.drop_duplicates()

# Remove rows with missing target values
train_clean = train_clean.dropna(subset=["Result"])

# Clean text values
text_columns = train_clean.select_dtypes(
    include=["object"]
).columns

for col in text_columns:
    train_clean[col] = train_clean[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    train_clean[col] = train_clean[col].replace("", np.nan)

# Convert date columns
for col in ["Transaction_Date"]:

    if col in train_clean.columns:
        train_clean[col] = pd.to_datetime(
            train_clean[col], errors="coerce"
        )

        train_clean.loc[
            train_clean[col].dt.year <= 1990, col
        ] = pd.NaT

# Check invalid numeric values
for col in ["Property_Area_Sqm", "Price_AED"]:

    if col in train_clean.columns:
        train_clean.loc[train_clean[col] <= 0, col] = np.nan

# Check outliers using IQR
for col in ["Property_Area_Sqm", "Price_AED"]:

    if col in train_clean.columns:

        Q1 = train_clean[col].quantile(0.25)
        Q3 = train_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = (
            (train_clean[col] < lower) |
            (train_clean[col] > upper)
        ).sum()

        print(col, "Outliers:", outliers)

# Keep valid outliers because large luxury properties are real transactions.

print("\nRows Before Cleaning:", len(train))
print("Rows After Cleaning:", len(train_clean))


# Exploratory Data Analysis


# Part 6: Exploratory Data Analysis

# Chart 1: Investment Class Distribution

plt.figure(figsize=(6, 4))
train_clean["Result"].value_counts().sort_index().plot(
    kind="bar", figsize=(6, 4)
)
plt.title("Dubai Property Investment Distribution")
plt.xlabel("Result (0 = Standard Value, 1 = Premium Value)")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("purchase_distribution.png")
plt.close()


# Chart 2: Property Area Distribution

plt.figure(figsize=(7, 4))

area_limit = train_clean["Property_Area_Sqm"].quantile(0.95)

plt.hist(
    train_clean["Property_Area_Sqm"].dropna().clip(upper=area_limit),
    bins=40
)

plt.title("Property Area Distribution (Sq.m)")
plt.xlabel("Property Area (Sq.m)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()


# Chart 3: Premium Rate by Dubai Area

plt.figure(figsize=(9, 5))

area_rate = train_clean.groupby(
    "Area_Name"
)["Result"].agg(["mean", "count"])

area_rate = area_rate[
    area_rate["count"] >= 200
]

area_rate = area_rate.sort_values(
    "mean", ascending=False
).head(10)

area_rate["mean"].plot(
    kind="bar", figsize=(9, 5)
)

plt.title("Premium Investment Rate by Dubai Area")
plt.xlabel("Dubai Area")
plt.ylabel("Premium Rate")
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("country_purchase_rate.png")
plt.close()


# Chart 4: Median Property Area by Investment Result

plt.figure(figsize=(6, 4))

train_clean.groupby("Result")["Property_Area_Sqm"].median().plot(
    kind="bar", figsize=(6, 4)
)

plt.title("Median Property Area by Investment Result")
plt.xlabel("Result")
plt.ylabel("Median Property Area (Sq.m)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("average_price.png")
plt.close()


# Chart 5: Monthly Premium Investment Rate

plt.figure(figsize=(8, 4))

monthly_rate = train_clean.groupby(
    train_clean["Transaction_Date"].dt.month
)["Result"].mean()

monthly_rate.plot(
    kind="line", marker="o", figsize=(8, 4)
)

plt.title("Premium Investment Rate by Month")
plt.xlabel("Month")
plt.ylabel("Premium Rate")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_purchase_rate.png")
plt.close()


# Feature Engineering


# Feature Engineering

# Create new features from Transaction_Date
train_clean["Trans_Year"] = train_clean["Transaction_Date"].dt.year
train_clean["Trans_Month"] = train_clean["Transaction_Date"].dt.month
train_clean["Trans_Quarter"] = train_clean["Transaction_Date"].dt.quarter

# Remove columns not suitable for prediction (including Price columns to prevent leakage)
drop_columns = [
    "Result",
    "Transaction_ID",
    "Transaction_Date",
    "Price_AED",
    "Price_PSF",
    "Area_ID",
    "Procedure_ID",
    "Quality_Flags",
    "Project_Name",
    "Building_Name"
]

X = train_clean.drop(
    columns=drop_columns, errors="ignore"
)

y = train_clean["Result"].astype(int)

# Remove columns with one unique value
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_columns)

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = [
    col for col in X.columns
    if col not in numeric_columns
]

# Convert categories to text
for col in categorical_columns:
    X[col] = X[col].apply(
        lambda x: str(x) if pd.notna(x) else np.nan
    )

print("Numeric Columns:", numeric_columns)
print("\nCategorical Columns:", categorical_columns)
print("\nFinal Features:", X.shape[1])


# Split Training and Validation Data


# Split the Data

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Rows:", len(X_train))
print("Validation Rows:", len(X_valid))


# Data Preprocessing


# Data Preprocessing

# Replace missing numeric values with the median
numeric_process = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Replace missing categories and encode them
categorical_process = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_process, numeric_columns),
    ("cat", categorical_process, categorical_columns)
])


# Build Machine Learning Models


# Build Machine Learning Models

models = {

    "Decision Tree": Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("preprocess", preprocessor),
        ("model", BernoulliNB())
    ])
}

print("Models created:")
for name in models:
    print(name)


# Hyperparameter Tuning


# Hyperparameter Tuning

parameters = {

    "Decision Tree": {
        "model__max_depth": [5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 5, 10]
    },

    "Naive Bayes": {
        "model__alpha": [0.1, 0.5, 1, 2]
    }
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

best_models = {}
tuning_results = []

for name in parameters:

    print("\nTuning:", name)

    search = RandomizedSearchCV(
        estimator=models[name],
        param_distributions=parameters[name],
        n_iter=5,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_

    tuning_results.append({
        "Model": name,
        "Best F1": search.best_score_,
        "Best Parameters": str(search.best_params_)
    })

tuning_df = pd.DataFrame(tuning_results)

print("\nTuning Results:")
print(tuning_df.to_string(index=False))

tuning_df.to_csv(
    "hyperparameter_tuning.csv", index=False
)


# Cross-Validation


# Cross-Validation

# Check model performance on the training data

cv_results = []

for name, model in best_models.items():

    print("Checking:", name)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring={
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1": "f1",
            "ROC_AUC": "roc_auc"
        },
        n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_Accuracy"].mean(),
        "Precision": scores["test_Precision"].mean(),
        "Recall": scores["test_Recall"].mean(),
        "F1": scores["test_F1"].mean(),
        "ROC_AUC": scores["test_ROC_AUC"].mean()
    })

cv_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results:")
print(cv_df.to_string(index=False))

cv_df.to_csv("cross_validation.csv", index=False)


# Final Model Evaluation


# Final Model Evaluation

results = []

for name, model in best_models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(y_valid, predictions)
    precision = precision_score(
        y_valid, predictions, zero_division=0
    )
    recall = recall_score(
        y_valid, predictions, zero_division=0
    )
    f1 = f1_score(
        y_valid, predictions, zero_division=0
    )
    auc = roc_auc_score(y_valid, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_valid, predictions)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Standard Value", "Premium Value"]
    ).plot()

    plt.title(name + " Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        name.replace(" ", "_") + "_confusion.png"
    )
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_valid, probabilities
    )

    plt.title(name + " ROC Curve")
    plt.tight_layout()
    plt.savefig(
        name.replace(" ", "_") + "_roc.png"
    )
    plt.close()


# Compare Models


# Compare Models

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "F1", ascending=False
)

print("Model Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison.csv", index=False
)

plt.figure(figsize=(11, 6))
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
].plot(kind="bar", figsize=(11, 6))

plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()


# Select the Best Model


# Select the Best Model

# Select the best model using cross-validation F1.
# Only Decision Tree and Naive Bayes are included in this comparison.

best_name = cv_df.sort_values(
    "F1", ascending=False
).iloc[0]["Model"]

print("Selected Model:", best_name)

best_model = best_models[best_name]

# Train the final model using all labelled data
best_model.fit(X, y)

# Save the model
joblib.dump(
    best_model,
    DATA_DIR / "dubai_property_model.pkl"
)

# Save feature names
joblib.dump(
    X.columns.tolist(),
    DATA_DIR / "model_columns.pkl"
)


# Final Project Summary


# Final Project Summary

print("DUBAI REAL ESTATE PREDICTION PROJECT")
print("------------------------------------")

print("Total Records:", len(train))
print("Training Split (80%):", len(X_train))
print("Validation Split (20%):", len(X_valid))
print("Predictor Columns:", len(X.columns))

print("\nSelected Model:", best_name)

print("\nModel Performance:")
print(results_df.to_string(index=False))

print("\nProject completed successfully.")


print("\nSaved files are located in:", DATA_DIR)
print("Model file:", DATA_DIR / "dubai_property_model.pkl")
print("Feature columns:", DATA_DIR / "model_columns.pkl")