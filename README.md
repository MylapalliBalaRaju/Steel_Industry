# Steel Industry Energy & Carbon Optimization (Step-by-Step)

This project provides a complete pipeline for:
- Cleaning steel industry energy data
- Predicting energy (`Usage_kWh`) with **Random Forest Regressor** (main) and **Linear Regression** (comparison)
- Calculating carbon emissions using:
  - `Carbon_Emission = Energy * 0.82`
- Finding the **Golden Signature Batch** using:
  - `Efficiency = (Quality + Yield) / Energy`
- Giving optimization caution messages for temperature and speed
- Displaying all results in a **Streamlit dashboard**

---

## 1) Dataset
Use your `steel_industry` CSV with columns such as:

- `date`
- `Usage_kWh`
- `Lagging_Current_Reactive.Power_kVarh`
- `Leading_Current_Reactive_Power_kVarh`
- `CO2(tCO2)`
- `Lagging_Current_Power_Factor`
- `Leading_Current_Power_Factor`
- `NSM`
- `WeekStatus`
- `Day_of_week`
- `Load_Type`

> If your dataset does not contain `Quality` and `Yield`, this code creates useful proxy features automatically.

---

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3) Train models from command line

```bash
python train_and_save.py --data steel_industry.csv
```

This script:
1. Loads data
2. Cleans data (duplicates, types, missing values)
3. Encodes categories and normalizes numeric data in a preprocessing pipeline
4. Trains Random Forest and Linear Regression
5. Prints metrics (MAE, RMSE, R²)
6. Saves models as `random_forest_model.joblib` and `linear_regression_model.joblib`

---

## 4) Run Streamlit dashboard (recommended)

```bash
streamlit run app.py
```

Dashboard includes:
- Cleaned data preview
- Model comparison (RF vs LR)
- Actual vs predicted chart
- Carbon calculation table
- Golden batch detection
- Optimization caution messages

---

## 5) Project structure

```text
.
├── app.py
├── requirements.txt
├── train_and_save.py
└── src
    └── pipeline.py
```

---

## 6) Technology mapping

| Part | Technology |
|---|---|
| AI Model | Scikit-learn |
| Backend | Flask (optional) |
| Dashboard | Streamlit ✅ |
| Database | MySQL (supported by dependency, integrate as needed) |
| Visualization | Plotly / Streamlit charts |

---

## 7) Notes
- Main prediction model: **RandomForestRegressor**
- Baseline comparison: **LinearRegression**
- Carbon formula is enforced in code: `df["Carbon_Emission"] = df["Energy"] * 0.82`
- Golden signature formula is enforced in code:
  - `df["Efficiency"] = (df["Quality"] + df["Yield"]) / df["Energy"]`
  - `golden_batch = df.loc[df["Efficiency"].idxmax()]`
