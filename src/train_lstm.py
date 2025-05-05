import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import load_data, create_time_series, preprocess_data
import os
import pandas as pd
import json
def build_lstm_model(seq_length):
    """Build an LSTM model for time series prediction."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train, seq_length):
    """Train the LSTM model with early stopping."""
    model = build_lstm_model(seq_length)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                        validation_split=0.2, callbacks=[early_stopping], verbose=1)
    return model, history

def predict_future(model, last_sequence, scaler, steps=180):
    """Predict future values for the next 'steps' days."""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Reshape for model input
        current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], 1))
        # Predict the next value
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        # Append the prediction
        future_predictions.append(next_pred[0, 0])
        # Update the sequence by removing the oldest value and adding the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    
    # Inverse transform the predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions.flatten()

def plot_loss(history, country):
    """Plot and save training and validation loss."""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Training Loss for {country}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_{country.replace(" ", "_")}.png')
    plt.close()

if __name__ == "__main__":
    # Load data
    data_path = r"D:\DEPI DataAnalytics\Projects\cyberattack_analysis\Data\formatted_cyber_with_region.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        exit(1)

    df = load_data(data_path)
    if df is None or df.empty:
        print("Loaded data is empty or invalid.")
        exit(1)

    # Select top 5 countries
    top_countries = df.groupby('Country')['Total_Attack_Percentage'].mean().nlargest(5).index
    seq_length = 10

    # Dictionary to store predictions for the report
    future_predictions = {}

    for country in top_countries:
        # Prepare data
        series = create_time_series(df, country)
        if series.empty:
            print(f"No data for {country}")
            continue

        X, y, scaler = preprocess_data(series, seq_length)
        if len(X) < 20:  # Skip if too few samples
            print(f"Insufficient data for {country}")
            continue

        # Correctly split X and y into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train model
        print(f"Training model for {country}")
        model, history = train_model(X_train, y_train, seq_length)
        model_path = f'model_{country.replace(" ", "_")}.h5'
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Plot loss
        plot_loss(history, country)

        # Evaluate model
        predictions = model.predict(X_test, verbose=0)
        try:
            predictions = scaler.inverse_transform(predictions)
            y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        except ValueError as e:
            print(f"Scaler inverse transform failed for {country}: {e}")
            continue

        rmse = np.sqrt(np.mean((predictions - y_test_unscaled) ** 2))
        print(f'RMSE for {country}: {rmse}')

        # Generate future predictions (next 6 months = 180 days)
        last_sequence = X[-1]  # Use the last sequence from the dataset
        future_vals = predict_future(model, last_sequence, scaler, steps=180)
        # Calculate average future attack percentage
        avg_future = np.mean(future_vals)
        # Compare with the last known value to calculate percentage change
        last_value = scaler.inverse_transform([y[-1]])[0]
        percentage_change = ((avg_future - last_value) / last_value) * 100
        # Ensure percentage_change is a scalar for formatting
        percentage_change = percentage_change.item()
        future_predictions[country] = percentage_change
        print(f"Future prediction for {country}: {percentage_change:.2f}% change over the next 6 months")

    # Save future predictions to a file for the report
    with open('web/future_predictions.json', 'w') as f:
        json.dump(future_predictions, f, indent=2)
    print("Future predictions saved to web/future_predictions.json")