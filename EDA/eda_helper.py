import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

def generate_overview(df):
    st.subheader("📦 Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {df.columns.tolist()}")
    st.write(f"Data Types:")
    st.dataframe(df.dtypes.astype(str))

    unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if unique_cols:
        st.warning(f"⚠️ Unique Columns (likely identifiers): {unique_cols}")
    else:
        st.success("✅ No fully unique columns detected.")

def show_data_summary(df):
    st.subheader("🧮 Summary Statistics")
    st.dataframe(df.describe(include='all'))

def plot_distributions(df):
    st.subheader("📈 Feature Distributions")
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

def plot_missing_values(df):
    st.subheader("🚨 Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write(missing)
        st.write("🔍 Missing Value Heatmap")
        fig, ax = plt.subplots(figsize=(10, 4))
        msno.heatmap(df, ax=ax)
        st.pyplot(fig)
    else:
        st.success("✅ No missing values found!")

def plot_correlations(df):
    st.subheader("📊 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include='number')
    if len(numeric_cols.columns) > 1:
        corr = numeric_cols.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation.")

def correlation_with_target(df, target_col):
    st.subheader(f"🎯 Correlation with Target: `{target_col}`")
    if target_col not in df.columns:
        st.warning("Target column not found.")
        return

    # Only consider numeric features
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Handle case where target is non-numeric (e.g., categorical like 'Iris-setosa')
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        st.info(f"🔄 Target column `{target_col}` is categorical. Applying Label Encoding for correlation.")
        le = LabelEncoder()
        df[target_col + '_encoded'] = le.fit_transform(df[target_col])
        target_col = target_col + '_encoded'

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if not numeric_cols:
        st.info("ℹ️ No numeric features found (excluding target) to compute correlation.")
        return

    correlations = df[numeric_cols].corrwith(df[target_col])
    corr_df = correlations.reset_index()
    corr_df.columns = ['Feature', 'Correlation']
    corr_df = corr_df.sort_values(by='Correlation', key=abs, ascending=False)

    st.dataframe(corr_df)

    fig = px.bar(corr_df, x='Feature', y='Correlation', title="Feature Correlation with Target")
    st.plotly_chart(fig, use_container_width=True)
