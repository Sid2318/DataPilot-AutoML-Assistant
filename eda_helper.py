import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

def generate(df):
    log = []
    log.append("Shape of the dataset: " + str(df.shape))
    log.append("Columns in the dataset: " + str(df.columns.tolist()))
    log.append("Unique Columns:" + str([col for col in df.columns if df[col].nunique() == len(df)]))
    log.append("Data Types:\n" + str(df.dtypes.astype(str)))
    log.append("Summary Statistics:\n" + str(df.describe(include='all')))
    numeric_cols = df.select_dtypes(include='number').columns  
    log.append("Numeric Columns :" + str(numeric_cols.tolist()))
    log.append("Categorical Columns :" + str(df.select_dtypes(include='object').columns.tolist()))  
    log.append("Missing Values:\n" + str(df.isnull().sum()))
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        log.append("Correlation Heatmap:\n" + str(numeric_df.corr().to_string()))
    else:
        log.append("Correlation Heatmap:\nNo numeric columns to compute correlation.")

    return log
    
    

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

# Add correlation matrix visualization
# st.subheader("🔍 Correlation Matrix")
# if len(df.select_dtypes(include='number').columns) > 1:
#     corr_matrix = df.corr()
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

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
