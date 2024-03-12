# models/my_python_model.py

import pandas as pd
from dbt.adapters.factory import get_adapter

def model(dbt, session):
    data = {
        'numbers': [1, 2, 3, 4, 5],
        'letters': ['a', 'b', 'c', 'd', 'e']
    }
    df = pd.DataFrame(data)

    # Log output (can also interact with your Databricks data warehouse)
    dbt.log(df)

    # Optionally, you can write this DataFrame back to your warehouse
    # using the adapter available from the session
    adapter = get_adapter(session.config)
    # adapter code to interact with the data warehouse

    # Return a DataFrame as the result of the model
    return df
