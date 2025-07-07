import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

class PreprocessingBot:
    def __init__(self, df):
        self.df = df
        self.pipeline = None
        self.logs = []
        self.imbalance_detected = False
        self.sampling_needed = False
        self.task_type = "unknown"

    def check_task_type(self, y):
        unique_vals = y.nunique()
        if unique_vals > 10 :
            self.task_type = "regression"
            self.logs.append("📈 Detected Task: **Regression** (Target has more than 10 unique numeric values)")
        else:
            self.task_type = "classification"
            self.logs.append("🔢 Detected Task: **Classification** (Target has 10 or fewer unique values)")

    def check_imbalance(self, y, threshold=0.3):
        class_counts = y.value_counts(normalize=True)
        self.logs.append("📊 Target Class Distribution:")
        for label, perc in class_counts.items():
            self.logs.append(f"   - {label}: {round(perc*100, 2)}%")

        if len(class_counts) > 1 and class_counts.min() < threshold:
            self.logs.append("⚠️ Class imbalance detected (a class has < 30%).")
            self.imbalance_detected = True
        else:
            self.logs.append("✅ No significant class imbalance detected.")

    def preprocess(self, target_col,sampling_strategy=None):
        # Step 1: Split features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        # Step 2: Detect column types
        numeric = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if(len(numeric)):
            self.logs.append(f"Numeric columns: {numeric}")
            self.logs.append(f"Scaling: StandardScaler for numeric")

        if(len(categorical)):
            self.logs.append(f"Categorical columns: {categorical}")
            self.logs.append(f"Encoding: OneHotEncoder with fit beforehand")

        self.logs.append(f"Imputation: Mean for numeric, Most Frequent for categorical")

        # Add logging for preprocessing steps
        self.logs.append("Starting data preprocessing...")
        self.logs.append(f"Detected numeric columns: {numeric}")
        self.logs.append(f"Detected categorical columns: {categorical}")
        
        # Step 3: Define preprocessing steps
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Manual OneHotEncoder fit (important step)
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(X[categorical])  # Fit separately before pipeline
        ohe_columns = encoder.get_feature_names_out(categorical)

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', encoder)  # Use the fitted encoder here
        ])

        # Step 4: Combine transformers
        self.pipeline = ColumnTransformer([
            ('num', num_pipeline, numeric),
            ('cat', cat_pipeline, categorical)
        ])

        # Step 5: Fit-transform the data
        X_processed = self.pipeline.fit_transform(X)

        # Step 6: Create final processed DataFrame
        all_columns = self.pipeline.get_feature_names_out()

        X_sampled, y_sampled = X_processed, y.values
        self.check_task_type(y)

        if self.task_type == "classification":
            self.check_imbalance(y)

            if self.imbalance_detected and sampling_strategy:
                self.logs.append(f"🎯 Sampling strategy selected: {sampling_strategy}")
                try:
                    if sampling_strategy == 'smote':
                        sampler = SMOTE(random_state=42)
                    elif sampling_strategy == 'undersample':
                        sampler = RandomUnderSampler(random_state=42)
                    elif sampling_strategy == 'smoteenn':
                        sampler = SMOTEENN(random_state=42)
                    else:
                        raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

                    X_sampled, y_sampled = sampler.fit_resample(X_processed, y)
                    self.logs.append(f"✅ Sampling applied. Shape: {X_processed.shape} ➡️ {X_sampled.shape}")
                except Exception as e:
                    self.logs.append(f"❌ Sampling failed: {str(e)}")

            elif not self.imbalance_detected:
                self.logs.append("ℹ️ Sampling skipped. Classes are already balanced.")
        else:
            self.logs.append("ℹ️ Skipping sampling — task is regression.")

        df_processed = pd.DataFrame(X_processed, columns=all_columns)
        df_processed[target_col] = y.values  # Append target

        self.logs.append("Data preprocessing completed.")

        return df_processed, self.logs
