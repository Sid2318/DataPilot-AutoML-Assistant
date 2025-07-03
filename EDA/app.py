import streamlit as st
import pandas as pd
from eda_helper import (
    generate_overview,
    show_data_summary,
    plot_distributions,
    plot_missing_values,
    plot_correlations,
    correlation_with_target
)


st.set_page_config(page_title="DataInsight - EDA Companion", layout="wide")
st.title("📊 DataInsight - EDA Companion")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select Target Column (Optional)", options=[None] + list(df.columns))

    st.markdown("---")
    generate_overview(df)
    show_data_summary(df)
    plot_distributions(df)
    plot_missing_values(df)
    plot_correlations(df)

    if target:
        correlation_with_target(df, target)


#for website of flask direct use ydata-profiling