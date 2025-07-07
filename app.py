import streamlit as st
import io
from datetime import datetime
from cleaner import DataCleaner
from eda_helper import generate
from preprocessing import PreprocessingBot
from tuner import ModelTuner
from explain_report import explain_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from ydata_profiling import ProfileReport
from optuna.visualization import plot_contour,plot_optimization_history,plot_parallel_coordinate,plot_slice,plot_param_importances
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title="AutoML Assistant", page_icon="🚀", layout="wide")
st.title("🚀 AutoML Assistant")

# Step 1: Upload Dataset
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type=["csv"])

if uploaded_file:
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    
    if "cleaned_data" not in st.session_state:
        st.session_state["cleaned_data"] = None

    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Raw Dataset Preview", df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Step 2: Data Cleaning
    st.header("Step 2: Data Cleaning")
    if st.button("🚀 Clean My Data"):
        # if 'cleaned_data' not in st.session_state:
        #     st.session_state['cleaned_data'] = None

        cleaner = DataCleaner(df)
        cleaner_df ,logs = cleaner.clean()

        st.session_state['cleaned_data'] = cleaner_df
        st.write("### ✅ Cleaned Data Preview")
        st.write(cleaner.df)

        st.session_state["logs"] += logs

    # Step 3: Exploratory Data Analysis
    st.header("Step 3: Exploratory Data Analysis")
    if st.button("🚀 Exploratory Data Analysis"):
        if 'cleaned_data' in st.session_state:
            st.session_state["logs"] += generate(st.session_state['cleaned_data'])

            with st.spinner("Generating profiling report..."):
                profile = ProfileReport(st.session_state['cleaned_data'], explorative=True)
                profile_html = profile.to_html()
                components.html(profile_html, height=1000, scrolling=True)
        else:
            st.warning("Please clean the data first.")

    # Step 4: Data Preprocessing
    st.header("Step 4: Data Preprocessing")
    if st.session_state["cleaned_data"] is not None:
        target_col = st.selectbox(
            "Select the target column:",
            st.session_state["cleaned_data"].columns,
            index=0,
            help="Choose the column you want to predict."
        )
    else:
        st.warning("Please clean the data first to select the target column.")
        st.stop()

    if 'cleaned_data' in st.session_state:
        st.markdown("### ⚖️ Choose Sampling Strategy *(Optional)*")
        sampling = st.selectbox("Sampling Strategy", [None, "smote", "undersample", "smoteenn"], help="Select a strategy to handle class imbalance.",key="sampling_strategy_2")

        if st.button("🚀 Data Preprocessing", help="Click to preprocess the data."):
            bot = PreprocessingBot(st.session_state['cleaned_data'])
            preprocessed_df, logs = bot.preprocess(target_col, sampling_strategy=sampling)
            st.session_state['preprocessed_data'] = preprocessed_df

            st.write("### 🧼 Preprocessed Data Preview")
            st.dataframe(preprocessed_df)

            st.session_state["logs"] += logs
            
    
    print(st.session_state["logs"])

    # Step 5: Model Tuning
    st.header("Step 5: Model Tuning")
    if 'preprocessed_data' in st.session_state:
        if "tuner" not in st.session_state:
            st.session_state.tuner = None
            if "sorted_results" not in st.session_state:
                st.session_state["sorted_results"] = None
            if "tuner" not in st.session_state:
                st.session_state["tuner"] = None
            if "X_train" not in st.session_state:
                st.session_state["X_train"] = None
            if "X_test" not in st.session_state:
                st.session_state["X_test"] = None
            if "y_train" not in st.session_state:
                st.session_state["y_train"] = None
            if "y_test" not in st.session_state:
                st.session_state["y_test"] = None
            
        n_trials = st.number_input("Number of trials for tuning", min_value=10, max_value=200, value=30, step=10)
        st.session_state["n_trials"] = n_trials

        if st.button("Run AutoML Tuning", help="Click to tune models automatically."):
            df = df.dropna()
            X = df.drop(columns=[target_col])
            y = df[target_col]

            problem_type = "classification" if y.nunique() <= 10 else "regression"
            st.session_state["problem_type"] = problem_type
            st.info(f"🔍 Detected Problem Type: **{problem_type.capitalize()}**")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            tuner = ModelTuner()
            if problem_type == 'classification':
                model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'svm', 'knn']
                scoring = 'accuracy'
            else:
                model_names = ['random_forest', 'xgboost', 'lightgbm', 'svm', 'knn']
                scoring = 'neg_mean_squared_error'

            with st.spinner("Tuning models..."):
                for model_name in model_names:
                    st.write(f"🔧 Tuning: **{model_name}**")
                    result = tuner.tune_model(model_name, X_train, y_train, problem_type=problem_type, scoring=scoring, n_trials=n_trials)
                    if problem_type == 'classification':
                        st.success(f"✅ {model_name} tuned! Accuracy: {result['best_score']:.4f}")
                    else:
                        st.success(f"✅ {model_name} tuned! MSE: {abs(result['best_score']):.4f}")

                st.session_state.tuner = tuner
                st.session_state.sorted_results = tuner.get_sorted_results()
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.model_names = model_names

                report = tuner.generate_report()
                st.session_state["logs"] += report.split('\n')

        if st.session_state.get("sorted_results") is not None:
            st.subheader("🔍 Sorted Results")
            st.dataframe(st.session_state.sorted_results)

            ensemble_type = st.selectbox("Choose Ensemble Type", ["voting", "stacking"], key="ensemble_type", help="Select the type of ensemble model.")
            top_n = st.slider("Top N models for ensemble", 1, len(st.session_state.sorted_results), 3, help="Choose the number of top models to include in the ensemble.")

            final_estimator_name = None
            if ensemble_type == "stacking":
                st.subheader("Choose Final Estimator for Stacking")
                available_models = st.session_state.sorted_results["Model"].tolist()
                final_estimator_name = st.selectbox("Final Estimator", available_models, key="final_estimator_name")

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

                with st.spinner("Fitting ensemble..."):
                    ensemble.fit(X_train, y_train)

                y_pred = ensemble.predict(X_test)

                problem_type = st.session_state["problem_type"]
                if problem_type == "classification":
                    performance = f"Accuracy: {accuracy_score(y_test, y_pred):.4f}"
                    st.success(f"✅ {performance}")
                else:
                    performance = f"MSE: {mean_squared_error(y_test, y_pred):.4f}"
                    st.success(f"✅ {performance}")

                st.session_state["ensemble_built"] = True
                st.session_state["ensemble_type_selected"] = ensemble_type
                st.session_state["top_n_selected"] = top_n
                st.session_state["final_estimator_used"] = final_estimator_name
                st.session_state["ensemble_performance"] = performance

                st.subheader("🧩 Models & Parameters in Ensemble")
                sorted_models = st.session_state.sorted_results.head(top_n)
                for _, row in sorted_models.iterrows():
                    st.markdown(f"**Model**: `{row['Model']}`")
                    st.markdown(f"**Best Score**: `{row['Best Score']:.4f}`")
                    params_block = "\n".join([
                        f"{k:<20}: {v:.4f}" if isinstance(v, float) else f"{k:<20}: {v}"
                        for k, v in row['Best Parameters'].items()
                    ])
                    st.markdown("**Parameters:**")
                    st.code(params_block, language='yaml')

                # st.subheader("📄 Download Full AutoML Report")
    
                report_text = f"# AutoML Tuning Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                report_text += "## Logs:\n"
                for log in st.session_state["logs"]:
                    report_text += f"- {log}\n"
                report_text += f"## Target Column: {target_col}\n"
                report_text += f"## Problem Type: {st.session_state['problem_type'].capitalize()}\n"
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

                report_text += "\n\n## Summary:\n"
                # report_text += st.session_state.report

                if st.session_state.get("ensemble_built", False):
                    report_text += "\n\n## Ensemble Summary:\n"
                    report_text += f"Ensemble Type         : {st.session_state['ensemble_type_selected'].capitalize()}\n"
                    report_text += f"Top N Models Used     : {st.session_state['top_n_selected']}\n"
                    if st.session_state['ensemble_type_selected'] == "stacking":
                        report_text += f"Final Estimator       : {st.session_state['final_estimator_used']}\n"
                    report_text += f"Ensemble Performance  : {st.session_state['ensemble_performance']}\n"
                    report_text += "\n### Models in Ensemble:\n"
                    sorted_models = st.session_state.sorted_results.head(st.session_state['top_n_selected'])
                    for _, row in sorted_models.iterrows():
                        report_text += f"- {row['Model']} (Score: {row['Best Score']:.4f})\n"

                report_file = io.BytesIO()
                report_file.write(report_text.encode('utf-8'))
                report_file.seek(0)
                st.session_state["report_text"] = report_text

                st.session_state["report_ready"] = True
                best_model_name = st.session_state.sorted_results.iloc[0]['Model']
                problem_type = st.session_state["problem_type"]

                with st.spinner("🧠 Your AI Data Scientist is thinking..."):
                    try:
                        explanation = explain_report(
                            report_text=st.session_state["report_text"],
                            target_column=target_col, 
                            problem_type=problem_type,
                            best_model_name=best_model_name
                        )
                        st.subheader("🧠 AI-Powered Explanation")
                        st.markdown(explanation)
                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")

