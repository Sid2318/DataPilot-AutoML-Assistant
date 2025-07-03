import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.logs = []

    def drop_constant_columns(self):
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_cols:
            self.df.drop(columns=constant_cols, inplace=True)
            self.logs.append(f"🗑️ Dropped constant columns: {constant_cols}")
    
    def drop_monotonic_id_like_columns(self):
        id_like_cols = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                diffs = self.df[col].diff().dropna()
                if (diffs == 1).all():
                    id_like_cols.append(col)
        if id_like_cols:
            self.df.drop(columns=id_like_cols, inplace=True)
            self.logs.append(f"🧾 Dropped ID-like columns (monotonic with step 1): {id_like_cols}")

    def fix_data_types(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.logs.append(f"🕒 Converted '{col}' to datetime.")
                except:
                    try:
                        self.df[col] = self.df[col].astype(float)
                        self.logs.append(f"🔢 Converted '{col}' to numeric.")
                    except:
                        continue

    def handle_missing_values(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in [np.float64, np.int64]:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    self.logs.append(f"📉 Filled missing numeric '{col}' with mean.")
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    self.logs.append(f"🅰️ Filled missing categorical '{col}' with mode.")

    def standardize_text(self):
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].str.strip().str.lower()
            self.logs.append(f"🔤 Standardized text format in '{col}'.")

    def clean(self):
        self.drop_constant_columns()
        self.drop_monotonic_id_like_columns()
        self.fix_data_types()
        self.handle_missing_values()
        self.standardize_text()
        return self.df, self.logs
