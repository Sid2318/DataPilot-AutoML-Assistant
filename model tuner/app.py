import streamlit as st
import pandas as pd
from tuner import ModelTuner
from sklearn.preprocessing import LabelEncoder
from optuna.visualization import plot_contour,plot_optimization_history,plot_parallel_coordinate,plot_slice,plot_param_importances


# Configure Streamlit page
st.set_page_config(page_title="Model Tuner - Optuna", layout="wide")
st.title("🚀 Model Tuner with Optuna")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select the Target Column", df.columns)

    if st.button("🧪 Start Model Tuning"):
        # Drop rows with NaNs
        df_clean = df.dropna()
        y = df_clean[target_col]
        X = df_clean.drop(columns=[target_col])

        # Detect problem type
        problem_type = "classification" if y.nunique() <= 10 else "regression"
        st.info(f"🔍 Detected Problem Type: **{problem_type.capitalize()}**")

        # Encode target if needed
        if y.dtype == 'object' or problem_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Initialize and store tuner in session
        if "tuner" not in st.session_state:
            st.session_state.tuner = ModelTuner()
        tuner = st.session_state.tuner

        # Choose models
        if problem_type == 'classification':
            model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'svm', 'knn']
            scoring = 'accuracy'
        else:
            model_names = ['random_forest', 'xgboost', 'lightgbm', 'svm', 'knn']
            scoring = 'neg_mean_squared_error'

        # Store model names in session for later use
        st.session_state.model_names = model_names

        # Tuning models
        with st.spinner("⏳ Tuning models using Optuna..."):
            results = []
            for model_name in model_names:
                st.write(f"🔧 Tuning: **{model_name}**")
                result = tuner.tune_model(model_name, X.values, y, problem_type=problem_type, scoring=scoring, n_trials=30)
                if problem_type == 'classification':
                    st.success(f"✅ {model_name} tuned! \nAccuracy: {result['best_score']:.4f}")
                else:
                    st.success(f"✅ {model_name} tuned!  \nMean Squared Error : {result['best_score']:.4f}")
                
                results.append((model_name, result['best_score']))

        # Generate report and store it
        report = tuner.generate_report()
        st.session_state.report = report

        # Find best model from results
        best_model = None
        best_score = float('-inf') if problem_type == 'classification' else float('inf')

        for model_name, score in results:
            if (problem_type == 'classification' and score > best_score) or \
               (problem_type == 'regression' and score < best_score):
                best_score = score
                best_model = model_name

        st.session_state.best_model = best_model

# Show results if available
if "report" in st.session_state and "best_model" in st.session_state:
    st.subheader("📊 Tuning Summary")
    st.text_area("🔎 Report", st.session_state.report, height=300)

    selected_model = st.selectbox(
        "📉 Select model to view optimization history",
        st.session_state.model_names,
        index=st.session_state.model_names.index(st.session_state.best_model)
    )

    tuner = st.session_state.tuner

    if st.button("📈 Show Optimization History"):
        study = tuner.plot_history(selected_model)
        fig1 = plot_optimization_history(study)
        st.plotly_chart(fig1, use_container_width=True)
        # fig2 = plot_contour(study)
        fig2 =plot_parallel_coordinate(study)
        st.plotly_chart(fig2,use_container_width=True)
        fig3 =plot_param_importances(study)
        st.plotly_chart(fig3,use_container_width=True)
