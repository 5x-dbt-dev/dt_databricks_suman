# models/databricks/outlier_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest

def model(dbt, session) -> pd.DataFrame:
    dbt.config(
        packages=["pandas", "scikit-learn"]
    )
    data = {
        'feature1': [20, 30, 35, 22, 85, 24, 23, 22],
        'feature2': [30, 35, 37, 31, 88, 33, 30, 31]
    }
    df = pd.DataFrame(data)
    isol_forest = IsolationForest(contamination=0.1)
    preds = isol_forest.fit_predict(df)
    df['outliers'] = preds
    return df
