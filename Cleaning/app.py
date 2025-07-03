import streamlit as st
import pandas as pd
from cleaner import DataCleaner

st.set_page_config(page_title="DataCleaner Assistant", page_icon="🧹", layout="wide")
st.title("🧹 DataCleaner - Smart Cleaning Assistant")
st.markdown("Clean your messy data in one click! 🚀")

file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("### 🔍 Raw Dataset Preview", df.head())

    if st.button("🚀 Clean My Data"):
        cleaner = DataCleaner(df)
        cleaned_df, logs = cleaner.clean()

        st.success("✅ Data Cleaning Complete!")
        st.write("### 🧾 Cleaning Summary")
        for log in logs:
            st.markdown(f"- {log}")

        st.write("### 🧼 Cleaned Data Preview")
        st.dataframe(cleaned_df.head())

        # Download cleaned file
        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")
else:
    st.info("📁 Please upload a CSV file to begin.")
