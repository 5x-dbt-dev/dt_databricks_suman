# models/databricks/linear_regression_model.py

import dbt
import pandas as pd
from sklearn.linear_model import LinearRegression

@dbt.model()
def train_linear_regression(df: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(df[['feature1', 'feature2']], df['target'])
    return model
