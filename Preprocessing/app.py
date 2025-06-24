import streamlit as st
import pandas as pd
from preprocessing import PreprocessingBot

# 🎯 App Configuration
st.set_page_config(page_title="DataPilot-AutoML-Assistant", page_icon="🧠", layout="centered")
st.title("🧼 DataPilot - AutoML Assistant")
st.markdown("#### Your smart helper for automatic preprocessing & sampling 🚀")

# 📁 File Upload
st.markdown("### 📤 Upload Your Dataset (CSV)")
file = st.file_uploader("", type=["csv"])

if file:
    # 🧾 Preview Dataset
    df = pd.read_csv(file)
    st.markdown("### 📝 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # 🎯 Select Target
    st.markdown("### 🎯 Select the Target Column")
    target = st.selectbox("", df.columns)

    # ⚖️ Sampling Options
    st.markdown("### ⚖️ Choose Sampling Strategy *(Optional)*")
    sampling = st.selectbox("", [None, "smote", "undersample", "smoteenn"])

    # 🧠 Run Preprocessing
    if st.button("🚀 Run Preprocessing"):
        with st.spinner("⏳ Processing your dataset..."):
            bot = PreprocessingBot(df)
            df_cleaned, logs = bot.preprocess(target, sampling_strategy=sampling)

        st.success("🎉 Preprocessing Completed!")

        # 🔍 Show Logs
        st.markdown("### 🧾 Preprocessing Summary")
        for log in logs:
            st.markdown(f"- {log}")

        # 🧪 Show Preview
        st.markdown("### 🔎 Preview of Cleaned Data")
        st.dataframe(df_cleaned.head(), use_container_width=True)

        # 📥 Download Option
        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.markdown("### 📦 Download Your Preprocessed Dataset")
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name='preprocessed_dataset.csv',
            mime='text/csv'
        )
else:
    st.info("📂 Please upload a CSV file to get started.")
