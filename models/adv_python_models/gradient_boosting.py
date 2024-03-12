# models/databricks/gradient_boosting.py

import dbt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

@dbt.model()
def gradient_boosting(df: pd.DataFrame) -> pd.DataFrame:
    X = df[['feature1', 'feature2', 'feature3']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                            max_depth = 5, alpha = 10, n_estimators = 10)
    model.fit(X_train, y_train)
    df['predictions'] = model.predict(X)
    return df
