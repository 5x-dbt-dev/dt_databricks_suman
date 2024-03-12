# models/databricks/linear_regression_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression

def model(dbt, session) -> pd.DataFrame:
    dbt.config(
        packages=["pandas", "scikit-learn"]
    )
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [1.1, 2.3, 3.1, 4.2, 5.0]
    }
    df = pd.DataFrame(data)
    model = LinearRegression()
    model.fit(df[['feature1', 'feature2']], df['target'])
    df['predictions'] = model.predict(df[['feature1', 'feature2']])
    return df
