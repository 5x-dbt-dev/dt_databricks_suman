# models/databricks/outlier_detection.py

import dbt
import pandas as pd
from sklearn.ensemble import IsolationForest

@dbt.model()
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame
