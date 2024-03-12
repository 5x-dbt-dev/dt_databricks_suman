# models/databricks/clustering_with_kmeans.py

import pandas as pd
from sklearn.cluster import KMeans

def model(dbt, session) -> pd.DataFrame:
    dbt.config(
        packages=["pandas", "scikit-learn"]
    )
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
        'feature2': [8, 7, 6, 5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)
    kmeans = KMeans(n_clusters=2, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['feature1', 'feature2']])
    return df
