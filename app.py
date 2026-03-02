import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import (
    clean_and_prepare,
    evaluate_models,
    find_golden_batch,
    optimization_suggestions,
    train_models,
)

st.set_page_config(page_title="Steel Industry Energy Optimizer", layout="wide")
st.title("⚙️ Steel Industry Energy, Carbon, and Golden Signature Dashboard")

st.markdown(
    """
This dashboard performs:
1. Data cleaning and feature engineering
2. Energy prediction (Random Forest main model, Linear Regression baseline)
3. Carbon emission calculation (`Carbon = Energy × 0.82`)
4. Golden signature (best efficiency batch) detection
5. Optimization suggestions
"""
)

uploaded_file = st.file_uploader("Upload `steel_industry.csv`", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    df = clean_and_prepare(raw_df)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    bundle = train_models(df)
    report = evaluate_models(bundle)

    st.subheader("Model Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest R²", f"{report['RandomForest']['R2']:.4f}")
        st.metric("Random Forest RMSE", f"{report['RandomForest']['RMSE']:.4f}")
    with col2:
        st.metric("Linear Regression R²", f"{report['LinearRegression']['R2']:.4f}")
        st.metric("Linear Regression RMSE", f"{report['LinearRegression']['RMSE']:.4f}")

    st.subheader("Predicted vs Actual Energy (Random Forest)")
    comparison_df = pd.DataFrame(
        {
            "Actual_Usage_kWh": bundle.y_test.values,
            "Predicted_RF": bundle.predictions_rf,
            "Predicted_LR": bundle.predictions_lr,
        }
    )
    fig = px.scatter(
        comparison_df,
        x="Actual_Usage_kWh",
        y="Predicted_RF",
        trendline="ols",
        title="Actual vs Predicted Usage_kWh (Random Forest)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Carbon Emission")
    st.write("Formula used: `Carbon_Emission = Energy * 0.82`")
    st.dataframe(df[["Energy", "Carbon_Emission"]].head(10), use_container_width=True)

    st.subheader("Golden Signature Batch")
    golden_batch = find_golden_batch(df)
    st.write(golden_batch.to_frame(name="value"))

    st.subheader("Optimization Suggestions")
    for message in optimization_suggestions(golden_batch):
        st.warning(message)

else:
    st.info("Please upload the steel industry dataset CSV to begin.")
