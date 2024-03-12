# models/databricks/time_series_decomposition.py

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def model(dbt, session) -> pd.DataFrame:
    dbt.config(
        packages=["pandas", "statsmodels"]
    )
    data = {
        'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'value': [1.2, 3.4, 2.5, 4.8, 3.6] * 20
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    decomposition = seasonal_decompose(df['value'], model='additive')
    df = df.join(decomposition.trend).join(decomposition.seasonal).join(decomposition.resid)
    return df.reset_index()
