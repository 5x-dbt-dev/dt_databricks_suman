import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def model(dbt, session) -> pd.DataFrame:
    # Configuring Databricks Environment
    dbt.config(
        packages=["pandas", "scikit-learn", "matplotlib", "seaborn"]
    )
    
    # Generating Sample Data for PCA
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Creating noisy features
    noise = np.random.normal(scale=0.1, size=(n_samples, 2))
    features = np.random.rand(n_samples, n_features-2)
    data = np.hstack([features, noise])
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(1, n_features+1)])
    
    # Performing PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df)
    
    # Visualizing Results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1])
    plt.title('PCA Analysis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()
    
    return pd.DataFrame(data=principalComponents, columns=['principal_component_1', 'principal_component_2'])
