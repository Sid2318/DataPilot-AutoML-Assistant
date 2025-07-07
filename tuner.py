# model_tuner.py

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


class ModelTuner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = {}
        self.best_scores = {}
        self.study_results = {}

    def infer_problem_type(self, y):
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        return "regression" if y_series.nunique() > 10 else "classification"

    def _get_model_params_space(self, model_name, trial, problem_type):
        if model_name == "logistic_regression" and problem_type == "classification":
            return {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
                "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
                "max_iter": 1000,
                "random_state": self.random_state,
            }
        elif model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": self.random_state,
            }
        elif model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.random_state,
            }
        elif model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.random_state,
                "verbosity": -1,
            }
        elif model_name == "svm":
            if problem_type == 'classification':
                return {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'kernel': trial.suggest_categorical('kernel', [ 'rbf']),
                }
            else:
                return {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'kernel': trial.suggest_categorical('kernel', [ 'rbf']),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
                }
        elif model_name == "knn":
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"]),
            }
        return {}

    def _get_model_instance(self, model_name, params, problem_type):
        if model_name == "logistic_regression" and problem_type == "classification":
            return LogisticRegression(**params)
        elif model_name == "random_forest":
            return RandomForestClassifier(**params) if problem_type == "classification" else RandomForestRegressor(**params)
        elif model_name == "xgboost":
            return XGBClassifier(**params, eval_metric="logloss") if problem_type == "classification" else XGBRegressor(**params, eval_metric="rmse")
        elif model_name == "lightgbm":
            return LGBMClassifier(**params) if problem_type == "classification" else LGBMRegressor(**params)
        elif model_name == "svm":
            return SVC(**params, probability=True) if problem_type == "classification" else SVR(**params)
        elif model_name == "knn":
            return KNeighborsClassifier(**params) if problem_type == "classification" else KNeighborsRegressor(**params)
        return None

    def _objective(self, trial, model_name, X, y, problem_type, scoring):
        params = self._get_model_params_space(model_name, trial, problem_type)
        model = self._get_model_instance(model_name, params, problem_type)
        if model is None:
            return float("-inf")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state) if problem_type == "classification" else KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        try:
            score = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
            return score
        except Exception as e:
            print(f"Trial failed for {model_name}: {e}")
            return float("-inf")

    def tune_model(self, model_name, X, y, problem_type=None, scoring=None, n_trials=None):
        problem_type = self.infer_problem_type(y)
        if scoring is None:
            scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"
        if n_trials is None:
            n_trials = 30  

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(lambda trial: self._objective(trial, model_name, X, y, problem_type, scoring), n_trials=n_trials, timeout=600,n_jobs=-1)

        self.best_params[model_name] = study.best_params
        self.best_scores[model_name] = study.best_value
        self.study_results[model_name] = study

        # Add logging for tuning steps
        print(f"Tuning model: {model_name}")
        print(f"Best parameters for {model_name}: {self.best_params[model_name]}")
        print(f"Best score for {model_name}: {self.best_scores[model_name]}")

        return {
            "model": model_name,
            "problem_type": problem_type,
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
        }

    def generate_report(self):
        report = []
        for model_name in self.best_params:
            score = self.best_scores[model_name]
            problem_type = (
                "classification" if score > 0 else "regression"
            )  # Based on score sign

            if problem_type == "classification":
                report.append(
                    f"Model: {model_name}\nAccuracy: {score:.4f}\nBest Params: {self.best_params[model_name]}\n"
                )
            else:
                report.append(
                    f"Model: {model_name}\nMean Squared Error: {score:.4f}\nBest Params: {self.best_params[model_name]}\n"
                )
        return '\n'.join(report)
    
    def get_sorted_results(self):
        results = []
        for model_name in self.best_params:
            score = self.best_scores[model_name]
            params = self.best_params[model_name]
            problem_type = "classification" if score > 0 else "regression"
            results.append({
                "Model": model_name,
                "Problem Type": problem_type,
                "Best Score": score,
                "Best Parameters": params,
            })
        return pd.DataFrame(sorted(results, key=lambda x: x["Best Score"], reverse=True))

    def build_ensemble(self, X, y, top_n=3, ensemble_type=None, final_estimator_name=None):
        problem_type = self.infer_problem_type(y)
        sorted_df = self.get_sorted_results()
        top_models_info = sorted_df.head(top_n)

        estimators = []
        for _, row in top_models_info.iterrows():
            model_name = row["Model"]
            params = row["Best Parameters"]
            model = self._get_model_instance(model_name, params, problem_type)
            estimators.append((model_name, model))

        if final_estimator_name:
            final_estimator = self._get_model_instance(final_estimator_name, self.best_params.get(final_estimator_name, {}), problem_type)
        else:
            final_estimator = LogisticRegression() if problem_type == "classification" else LinearRegression()

        if problem_type == "classification":
            if ensemble_type == "voting":
                return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
            else:
                return StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
        else:
            if ensemble_type == "voting":
                return VotingRegressor(estimators=estimators, n_jobs=-1)
            else:
                return StackingRegressor(estimators=estimators, final_estimator=final_estimator, n_jobs=-1)
