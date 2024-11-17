import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# 1. Data Loading and Preprocessing
def load_data(ticker, start_date, end_date):
    """
    Load historical stock data for the given ticker and date range.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    data = df[['Close']]
    return data

def preprocess_data(data):
    """
    Scale the data using MinMaxScaler.
    """
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler

# 2. Create Dataset
def create_dataset(dataset, sequence_length):
    """
    Create input-output pairs for training/testing the LSTM model.
    """
    x = []
    y = []
    for i in range(sequence_length, len(dataset)):
        x.append(dataset[i - sequence_length:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

# 3. Build the LSTM Model
def build_model(sequence_length):
    """
    Build the LSTM model architecture.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    return model

# 4. Train the Model
def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=32):
    """
    Compile and train the LSTM model.
    """
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val)
    )
    return model, history

# 5. Make Predictions
def make_predictions(model, x_test, scaler):
    """
    Use the trained model to make predictions on test data.
    """
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# 6. Evaluate the Model
def evaluate_model(predictions, y_test_scaled):
    """
    Calculate RMSE between predicted and actual stock prices.
    """
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    return rmse

# 7. Plot the Results
def plot_results(y_test_scaled, predictions, ticker, sequence_length):
    """
    Plot actual vs. predicted stock prices.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_scaled, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction with {sequence_length}-Day Sequence')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# 8. Up/Down Prediction Accuracy
def up_down_accuracy(predictions, y_actual, time_interval=1):
    """
    Calculate the accuracy of predicting the direction of stock price movement.
    """
    correct = 0
    n = len(predictions) - 1
    for i in range(n):
        if(i+time_interval>=len(predictions)):
            break
        if (predictions[i + time_interval] - predictions[i]) * (y_actual[i + time_interval] - y_actual[i]) > 0:
            correct += 1
            i = i + time_interval
    return correct / n

# 8.5. Recursive Forecasting Function
def recursive_forecast(model, last_sequence, forecast_horizon, scaler, sequence_length):
    prediction = 0
    current_sequence = last_sequence.copy()

    for _ in range(forecast_horizon):
        input_seq = current_sequence.reshape((1, sequence_length, 1))
        next_pred_scaled = model.predict(input_seq)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
        prediction = next_pred

        # Update the current sequence
        next_pred_scaled = next_pred_scaled[0, 0]
        current_sequence = np.append(current_sequence[1:], [next_pred_scaled], axis=0)


    return prediction


def make_recursive_predictions(model, x_test, scaler, forecast_horizon, sequence_length):
    """
    Use the trained model to make recursive predictions on test data.
    """
    predictions = []
    for i in range(len(x_test)):
        last_sequence = x_test[i, :, 0]
        prediction = recursive_forecast(model, last_sequence, forecast_horizon, scaler, sequence_length)
        predictions.append(prediction)
    return np.array(predictions)

# 9. Plot Training and Validation Loss
def plot_training_history(history):
    """
    Plot the training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 10. Plot Error Distribution
def plot_error_distribution(y_test_scaled, predictions):
    """
    Plot the distribution of prediction errors.
    """
    errors = y_test_scaled - predictions
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, color='purple', edgecolor='black')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.show()

# 11. Plot Predicted vs. Actual Prices Scatter Plot
def plot_scatter_actual_vs_predicted(y_test_scaled, predictions):
    """
    Plot a scatter plot of actual vs. predicted stock prices.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_scaled, predictions, alpha=0.5)
    plt.title('Actual vs. Predicted Stock Prices')
    plt.xlabel('Actual Stock Price')
    plt.ylabel('Predicted Stock Price')
    plt.plot([y_test_scaled.min(), y_test_scaled.max()], [y_test_scaled.min(), y_test_scaled.max()], 'k--', lw=2)
    plt.show()

# 12. Plot Up/Down Movement Visualization
def plot_up_down_movement(y_test_scaled, predictions):
    """
    Plot the model's ability to predict the direction of stock price movement over time.
    """
    actual_movement = np.where(np.diff(y_test_scaled.flatten()) > 0, 1, -1)
    predicted_movement = np.where(np.diff(predictions.flatten()) > 0, 1, -1)
    time_axis = np.arange(len(actual_movement))

    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, actual_movement, label='Actual Movement', color='blue', alpha=0.7)
    plt.plot(time_axis, predicted_movement, label='Predicted Movement', color='red', alpha=0.5)
    plt.title('Actual vs. Predicted Stock Price Movement Direction')
    plt.xlabel('Time')
    plt.ylabel('Movement Direction')
    plt.yticks([-1, 1], ['Down', 'Up'])
    plt.legend()
    plt.show()

# 13. Save the Model and Scaler
def save_model_and_scaler(model, scaler, model_filename, scaler_filename):
    """
    Save the trained model and scaler to disk.
    """
    # Save the model
    model.save('AIMATH/Models/'+model_filename)
    print(f"Model saved to {'AIMATH/Models/'+model_filename}")

    # Save the scaler
    joblib.dump(scaler, 'AIMATH/Scalers/'+scaler_filename)
    print(f"Scaler saved to {'AIMATH/Scalers/'+scaler_filename}")

# 14. Load the Model and Scaler
def load_model_and_scaler(model_filename, scaler_filename):
    """
    Load a trained model and scaler from disk.
    """

    model = load_model('AIMATH/Models/'+model_filename)
    print(f"Model loaded from {'AIMATH/Models/'+model_filename}")
    scaler = joblib.load('AIMATH/Scalers/'+scaler_filename)
    print(f"Scaler loaded from {'AIMATH/Scalers/'+scaler_filename}")
    return model, scaler

# 15. Main Function
def main():
    # Parameters
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2021-12-31'
    sequence_length = 1  # Time slice of 10 days

    # Filenames for saving model and scaler
    model_filename = 'lstm_stock_model.keras'
    scaler_filename = 'scaler.save'

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
    x_train = np.reshape(x_train, (x_train.shape[0], sequence_length, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], sequence_length, 1))

    # Build and train the model
    model = build_model(sequence_length)
    model, history = train_model(model, x_train, y_train, x_test, y_test)
    # Save the model and scaler
    save_model_and_scaler(model, scaler, model_filename, scaler_filename)

    # Make predictions
    predictions = make_predictions(model, x_test, scaler)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    #recursive forecast
    sequence_length = 1
    forecast_horizon = 15
    predictions_recursive = make_recursive_predictions(model, x_test, scaler, forecast_horizon, sequence_length)
    print("Recursive Forecasting updown accuracy: ", up_down_accuracy(predictions_recursive, y_test_scaled.flatten(), forecast_horizon)*100)
    plot_results(y_test_scaled, predictions_recursive, ticker, sequence_length)

    # Evaluate the model
    rmse = evaluate_model(predictions, y_test_scaled)
    print(f'RMSE with sequence length {sequence_length}: {rmse:.4f}')

    # Plot the results
    plot_results(y_test_scaled, predictions, ticker, sequence_length)

    # Plot training and validation loss
    plot_training_history(history)

    # Plot error distribution
    plot_error_distribution(y_test_scaled, predictions)

    # Plot actual vs. predicted prices scatter plot
    plot_scatter_actual_vs_predicted(y_test_scaled, predictions)

    # Up/Down Prediction Accuracy
    y_test_array = y_test_scaled.flatten()
    predictions_array = predictions.flatten()
    accuracy = up_down_accuracy(predictions_array, y_test_array, 15)
    print(f"Up/Down Prediction Accuracy: {accuracy:.2%}")

    # Plot up/down movement visualization
    plot_up_down_movement(y_test_scaled, predictions)

if __name__ == '__main__':
    main()
