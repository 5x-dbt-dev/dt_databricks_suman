# models/databricks/pca_analysis.py

import dbt
import pandas as pd
from sklearn.decomposition import PCA

@dbt.model()
def pca_analysis(df: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df[['feature1', 'feature2', 'feature3']])
    return pd.DataFrame(data = principalComponents, columns = ['principal_component_1', 'principal_component_2'])
