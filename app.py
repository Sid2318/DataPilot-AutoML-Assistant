import streamlit as st
import pandas as pd
from preprocessing import PreprocessingBot

st.title("🧼 Auto Preprocessing Bot")
st.set_page_config(page_title="Auto Preprocessing Bot", page_icon="🧼")

file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])


if file:
    df = pd.read_csv(file)
    st.write("📝 **Dataset Preview**", df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)
    sampling = st.selectbox("⚖️ Choose Sampling (Optional if imbalance found)", [None, "smote", "undersample", "smoteenn"])

    if st.button("🚀 Run Preprocessing"):
        bot = PreprocessingBot(df)
        df_cleaned, logs = bot.preprocess(target, sampling_strategy=sampling)

        st.success("✅ Preprocessing Complete!")

        # Show Logs
        st.subheader("🔍 Transformations Summary")
        for log in logs:
            st.markdown(f"- {log}")

        # Show Preview
        st.subheader("🔎 Preprocessed Dataset (First 5 Rows)")
        st.dataframe(df_cleaned.head())

        # Allow Download
        csv = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name='preprocessed_dataset.csv',
            mime='text/csv'
        )
