import pandas as pd


def model(dbt, session):
    dbt.config(
        packages=["pandas"]
    )
    data = {
        'numbers': [1, 2, 3, 4, 5],
        'letters': ['a', 'b', 'c', 'd', 'e']
    }
    df = pd.DataFrame(data)
    return df
