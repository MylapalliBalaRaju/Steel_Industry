from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

TARGET_COLUMN = "Usage_kWh"
ENERGY_COLUMN = "Energy"
QUALITY_COLUMN = "Quality"
YIELD_COLUMN = "Yield"


@dataclass
class ModelBundle:
    random_forest: Pipeline
    linear_regression: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions_rf: np.ndarray
    predictions_lr: np.ndarray


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the steel industry dataset from CSV."""
    df = pd.read_csv(file_path)
    return df


def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["hour"] = df["date"].dt.hour
    return df


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning and basic feature engineering.
    - removes duplicates
    - fixes numeric types
    - fills missing values
    - adds useful features
    """
    df = df.copy()

    # Standardize column names for easier use.
    df.columns = [col.strip() for col in df.columns]

    # Drop duplicates.
    df = df.drop_duplicates()

    # Create date-based features.
    df = _create_time_features(df)

    # Convert known numeric columns safely.
    numeric_hint_columns = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
    ]
    for col in numeric_hint_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add Energy alias from Usage_kWh.
    if TARGET_COLUMN in df.columns:
        df[ENERGY_COLUMN] = df[TARGET_COLUMN]

    # If quality/yield are not present, build proxies from available columns.
    if QUALITY_COLUMN not in df.columns:
        if "Lagging_Current_Power_Factor" in df.columns and "Leading_Current_Power_Factor" in df.columns:
            df[QUALITY_COLUMN] = (
                df["Lagging_Current_Power_Factor"].fillna(df["Lagging_Current_Power_Factor"].median())
                + df["Leading_Current_Power_Factor"].fillna(df["Leading_Current_Power_Factor"].median())
            ) / 2
        else:
            df[QUALITY_COLUMN] = 1.0

    if YIELD_COLUMN not in df.columns:
        # Proxy yield from normalized inverse reactive power.
        reactive_cols = [
            c
            for c in [
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
            ]
            if c in df.columns
        ]
        if reactive_cols:
            reactive_total = df[reactive_cols].sum(axis=1)
            scaled = (reactive_total - reactive_total.min()) / (reactive_total.max() - reactive_total.min() + 1e-9)
            df[YIELD_COLUMN] = 1 - scaled
        else:
            df[YIELD_COLUMN] = 1.0

    # Fill missing values:
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    # Carbon formula requested.
    if ENERGY_COLUMN in df.columns:
        df["Carbon_Emission"] = df[ENERGY_COLUMN] * 0.82

    # Golden signature efficiency.
    df["Efficiency"] = (df[QUALITY_COLUMN] + df[YIELD_COLUMN]) / (df[ENERGY_COLUMN] + 1e-9)

    return df


def build_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found.")

    feature_df = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    categorical_features = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in feature_df.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return feature_df, target, preprocessor


def train_models(df: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    X, y, preprocessor = build_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
            ),
        ]
    )

    lr_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    preds_rf = rf_model.predict(X_test)
    preds_lr = lr_model.predict(X_test)

    return ModelBundle(
        random_forest=rf_model,
        linear_regression=lr_model,
        X_test=X_test,
        y_test=y_test,
        predictions_rf=preds_rf,
        predictions_lr=preds_lr,
    )


def evaluate_models(bundle: ModelBundle) -> Dict[str, Dict[str, float]]:
    def metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(rmse),
            "R2": float(r2_score(y_true, y_pred)),
        }

    return {
        "RandomForest": metrics(bundle.y_test, bundle.predictions_rf),
        "LinearRegression": metrics(bundle.y_test, bundle.predictions_lr),
    }


def find_golden_batch(df: pd.DataFrame) -> pd.Series:
    # Requested formula:
    # df["Efficiency"] = (df["Quality"] + df["Yield"]) / df["Energy"]
    # golden_batch = df.loc[df["Efficiency"].idxmax()]
    return df.loc[df["Efficiency"].idxmax()]


def optimization_suggestions(row: pd.Series) -> list[str]:
    messages = []
    temp_columns = [c for c in row.index if "temp" in c.lower()]
    speed_columns = [c for c in row.index if "speed" in c.lower()]

    for col in temp_columns:
        val = pd.to_numeric(row[col], errors="coerce")
        if pd.notna(val):
            messages.append(f"Caution: {col} is {val:.2f}. If temperature increases further, reduce it.")

    for col in speed_columns:
        val = pd.to_numeric(row[col], errors="coerce")
        if pd.notna(val):
            messages.append(f"Caution: {col} is {val:.2f}. If speed increases further, decrease it.")

    if not messages:
        messages.append(
            "No explicit temperature/speed columns found; monitor process parameters and reduce temp/speed if they trend upward."
        )

    return messages
