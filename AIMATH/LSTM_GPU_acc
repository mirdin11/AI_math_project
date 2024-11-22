import cupy as cp  # Import CuPy
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Data Loading and Preprocessing
def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df[['Close']]
    return data

def preprocess_data(data):
    dataset = data.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # Convert scaled_data to CuPy array
    scaled_data = cp.asarray(scaled_data)
    return scaled_data, scaler

# 2. Create Dataset for Single-Step Forecasting
def create_dataset(dataset, sequence_length):
    x = []
    y = []
    dataset_length = len(dataset)
    for i in range(sequence_length, dataset_length):
        x.append(dataset[i - sequence_length:i])
        y.append(dataset[i])
    x = cp.array(x)
    y = cp.array(y)
    return x, y

# 3. Build the LSTM Model
def build_model(sequence_length):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(units=64))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 4. Recursive Forecasting Function
def recursive_forecast(model, last_sequence, forecast_horizon, scaler, sequence_length):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(forecast_horizon):
        input_seq = current_sequence.reshape((1, sequence_length, 1))
        # Convert input_seq to NumPy array for TensorFlow
        input_seq_np = cp.asnumpy(input_seq)
        next_pred_scaled = model.predict(input_seq_np)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
        predictions.append(next_pred)

        # Update the current sequence
        next_pred_scaled_value = next_pred_scaled[0, 0]
        # Reshape next_pred_scaled_value_cp to (1, 1)
        next_pred_scaled_value_cp = cp.array([[next_pred_scaled_value]])  # Shape (1, 1)
        current_sequence = cp.concatenate((current_sequence[1:], next_pred_scaled_value_cp), axis=0)
    return predictions


# 5. Main Function
def main():
    # Check if GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Parameters
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2021-12-31'
    sequence_length = 60
    forecast_horizon = 10

    # Load and preprocess data
    data = load_data(ticker, start_date, end_date)
    scaled_data, scaler = preprocess_data(data)

    # Split data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]

    # Create datasets
    x_train, y_train = create_dataset(train_data, sequence_length)
    x_test, y_test = create_dataset(test_data, sequence_length)

    # Reshape input data
    x_train = x_train.reshape((x_train.shape[0], sequence_length, 1))
    x_test = x_test.reshape((x_test.shape[0], sequence_length, 1))

    # Convert CuPy arrays to NumPy arrays for TensorFlow
    x_train = cp.asnumpy(x_train)
    y_train = cp.asnumpy(y_train)
    x_test = cp.asnumpy(x_test)
    y_test = cp.asnumpy(y_test)

    # Build and train the model
    model = build_model(sequence_length)

    batch_size = 64
    buffer_size = 1000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset
    )

    # Make recursive predictions for the next 10 days
    last_sequence = scaled_data[-sequence_length:]
    predictions = recursive_forecast(model, last_sequence, forecast_horizon, scaler, sequence_length)

    # Print the predictions
    print("Recursive Predictions for the Next 10 Days:")
    for i, pred in enumerate(predictions, 1):
        print(f"Day {i}: {pred:.2f}")

    # Generate future dates for plotting
    last_date = data.index[-1]
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(dates, predictions, marker='o', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction for Next {forecast_horizon} Days')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
