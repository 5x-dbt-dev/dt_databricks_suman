# models/databricks/tensorflow_neural_network.py

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def model(dbt, session) -> pd.DataFrame:
    dbt.config(
        materialized="table",
        packages=["pandas", "scikit-learn", "tensorflow"]
    )

    # Sample data
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 1, 3, 1, 2],
        'target': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['feature1', 'feature2']])
    y = df['target'].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),  # 10 neurons in the hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    # Make predictions
    predictions = model.predict(X_test)

    # Create a DataFrame to return
    results_df = pd.DataFrame(X_test, columns=['feature1', 'feature2'])
    results_df['predicted'] = predictions.flatten()

    return results_df
