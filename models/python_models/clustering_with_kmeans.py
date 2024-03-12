# models/databricks/clustering_with_kmeans.py

import dbt
import pandas as pd
from sklearn.cluster import KMeans

@dbt.model()
def kmeans_clustering(df: pd.DataFrame) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df[['feature1', 'feature2']])
    return df
