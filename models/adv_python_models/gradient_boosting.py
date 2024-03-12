# models/databricks/gradient_boosting.py

import pandas as pd
import xgboost

def model(dbt, session) -> pd.DataFrame:
    dbt.config(
        packages=["pandas", "xgboost"]
    )
    data = {
        'feature1': [2, 3, 5, 7, 11, 13, 17, 19],
        'feature2': [29, 23, 31, 37, 41, 43, 47, 53],
        'target': [0, 1, 0, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    X, y = df[['feature1', 'feature2']], df['target']
    model1 = xgboost.XGBClassifier()
    model1.fit(X, y)
    df['predictions'] = model1.predict(X)
    return df
    # this is a comment 
