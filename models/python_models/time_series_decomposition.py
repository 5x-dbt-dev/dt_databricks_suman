# models/databricks/time_series_decomposition.py

import dbt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

@dbt.model()
def decompose_time_series(df: pd.DataFrame) -> pd.DataFrame:
    decomposition = seasonal_decompose(df['time_series_column'], model='additive', period=12)
    return pd.concat([df, decomposition.trend, decomposition.seasonal, decomposition.resid], axis=1)
