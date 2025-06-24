# app.py
import io
from datetime import datetime
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tuner import ModelTuner
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

st.title("🧠 AutoML Tuner + Ensemble Builder")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    target_column = st.selectbox("Select target column", df.columns)

    if "tuner" not in st.session_state:
        st.session_state.tuner = None
        st.session_state.sorted_results = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None

    n_trials = st.number_input("Number of trials for tuning", min_value=10, max_value=200, value=30, step=10)
    st.session_state["n_trials"] = n_trials

    if st.button("Run AutoML Tuning"):
        df = df.dropna()
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Detect problem type
        problem_type = "classification" if y.nunique() <= 10 else "regression"
        st.info(f"🔍 Detected Problem Type: **{problem_type.capitalize()}**")

        # Encode target if needed
        if y.dtype == 'object' or problem_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tuner = ModelTuner()
        if problem_type == 'classification':
            model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'svm', 'knn']
            scoring = 'accuracy'
        else:
            model_names = ['random_forest', 'xgboost', 'lightgbm', 'svm', 'knn']
            scoring = 'neg_mean_squared_error'

        with st.spinner("Tuning models..."):
            results = []
            for model_name in model_names:
                st.write(f"🔧 Tuning: **{model_name}**")
                result = tuner.tune_model(model_name, X_train, y_train, problem_type=problem_type, scoring=scoring, n_trials=n_trials)
                if problem_type == 'classification':
                    st.success(f"✅ {model_name} tuned! Accuracy: {result['best_score']:.4f}")
                else:
                    st.success(f"✅ {model_name} tuned! MSE: {abs(result['best_score']):.4f}")  # abs() since it's neg MSE

                results.append((model_name, result['best_score']))

            st.session_state.tuner = tuner
            st.session_state.sorted_results = tuner.get_sorted_results()
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.model_names = model_names

        # Generate report and store it
        report = tuner.generate_report()
        st.session_state.report = report

    # Show sorted results if available
    if st.session_state.sorted_results is not None:
        st.subheader("🔍 Sorted Results")
        st.dataframe(st.session_state.sorted_results)

        ensemble_type = st.selectbox("Choose Ensemble Type", ["voting", "stacking"], key="ensemble_type")
        top_n = st.slider("Top N models for ensemble", 1, len(st.session_state.sorted_results), 3)

        final_estimator_name = None
        if ensemble_type == "stacking":
            st.subheader("Choose Final Estimator for Stacking")
            available_models = st.session_state.sorted_results["Model"].tolist()
            final_estimator_name = st.selectbox("Final Estimator", available_models, key="final_estimator_name")

        # Build only when button is clicked
        if st.button("Build Ensemble and Evaluate"):
            tuner = st.session_state.tuner
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            ensemble = tuner.build_ensemble(
                X_train, y_train, 
                top_n=top_n, 
                ensemble_type=ensemble_type,
                final_estimator_name=final_estimator_name if ensemble_type == "stacking" else None
            )

            # Fit the ensemble
            with st.spinner("Fitting ensemble..."):
                ensemble.fit(X_train, y_train)

            # Predict
            y_pred = ensemble.predict(X_test)

            # Evaluate
            problem_type = tuner.infer_problem_type(y_train)
            if problem_type == "classification":
                st.success(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            else:
                st.success(f"✅ MSE: {mean_squared_error(y_test, y_pred):.4f}")

            st.subheader("🧩 Models & Parameters in Ensemble")
            sorted_models = tuner.get_sorted_results().head(top_n)
            for _, row in sorted_models.iterrows():
                st.markdown(f"**Model**: `{row['Model']}`")
                st.markdown(f"**Best Score**: `{row['Best Score']:.4f}`")
                params_lines = []
                for k, v in row['Best Parameters'].items():
                    if isinstance(v, float):
                        params_lines.append(f"{k:<20}: {v:.4f}")
                    else:
                        params_lines.append(f"{k:<20}: {v}")
                params_block = "\n".join(params_lines)

                st.markdown(f"**Parameters:**")
                st.code(params_block, language='yaml')

# 📝 Generate and Download Text Report
    if "report" in st.session_state and "sorted_results" in st.session_state:
        st.subheader("📄 Download Full AutoML Report")

        report_text = f"# AutoML Tuning Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report_text += f"## Target Column: {target_column}\n"
        report_text += f"## Problem Type: {problem_type.capitalize()}\n"
        report_text += f"## Number of Trials: {st.session_state['n_trials']}\n\n"

        report_text += "## Sorted Results:\n"
        df = st.session_state.sorted_results.copy()
        for idx, row in df.iterrows():
            report_text += f"\nModel {idx+1}: {row['Model']}\n"
            report_text += f"  - Problem Type : {row['Problem Type']}\n"
            report_text += f"  - Best Score   : {row['Best Score']:.4f}\n"
            report_text += f"  - Best Params  :\n"
            for k, v in row['Best Parameters'].items():
                report_text += f"      • {k}: {v}\n"

        report_text +=  f"\n## Ensumble:\n"
        report_text +=  f"Models = {tuner.get_sorted_results().head(top_n) }\n"
        if ensemble_type == "stacking":
            report_text +=  f" Final Model = {final_estimator_name}\n"
        

        report_text += "\n\n## Summary:\n"
        report_text += st.session_state.report
        
    

        # Save as downloadable file
        report_file = io.BytesIO()
        report_file.write(report_text.encode('utf-8'))
        report_file.seek(0)

        st.download_button(
            label="⬇️ Download Report as TXT",
            data=report_file,
            file_name="automl_report.txt",
            mime="text/plain"
        )