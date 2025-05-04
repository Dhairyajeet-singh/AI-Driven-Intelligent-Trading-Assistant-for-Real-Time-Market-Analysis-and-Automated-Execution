import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set random seed for reproducibility
np.random.seed(42)

def get_stock_data(user_input, period="1y", interval="1d"):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    - user_input: Stock ticker symbol
    - period: Data period (e.g., "7d", "1mo", "1y", "max")
    - interval: Data interval (e.g., "1m", "1h", "1d", "1wk")
    
    Returns:
    - DataFrame with stock data
    """
    try:
        # If .NS is not in the ticker, add it for Indian stocks
        if ".NS" not in user_input:
            ticker = f"{user_input}.NS"
        else:
            ticker = user_input
            
        stock = yf.Ticker(ticker)
        
        # Get historical data
        df = stock.history(period=period, interval=interval)
        
        # Process the DataFrame
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Display data information
        print(f"\nFetched data for {ticker}")
        print(f"Shape: {df.shape}")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        
        # Save data to CSV
        df.to_csv(f"{user_input}_data.csv", index=True)
        
        return df
    
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def get_news_summary(stock_name, ticker_symbol):
    """
    Placeholder for news fetching functionality.
    
    Note: The original code used Google's Gemini API which was causing errors.
    This function now returns a message to consult financial news sources.
    
    In a production system, you could:
    1. Use a different news API (Alpha Vantage, NewsAPI, etc.)
    2. Implement web scraping with proper permissions
    3. Use a different LLM API
    """
    
    print(f"\n📈 News for {stock_name} ({ticker_symbol}):")
    message = (
        f"To get the latest news about {stock_name} ({ticker_symbol}), please consult financial news sources like:\n"
        "- Yahoo Finance\n"
        "- Economic Times\n"
        "- Moneycontrol\n"
        "- Bloomberg\n\n"
        "The original Gemini API integration was removed due to compatibility issues."
    )
    
    print(message)
    return message

def preprocess_data(df, feature='Close', test_size=0.2):
    """
    Preprocess the data for LSTM model
    """
    # Select the feature to predict (typically closing price)
    data = df[feature].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create training and testing sets
    training_data_len = int(np.ceil(len(scaled_data) * (1 - test_size)))
    
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - 60:, :]
    
    return train_data, test_data, scaler

def create_sequences(data, time_steps=60):
    """
    Create sequences for LSTM model with the specified time steps
    """
    X, y = [], []
    
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
        
    # Convert lists to numpy arrays
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be 3D [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def build_lstm_model(input_shape):
    """
    Build LSTM model for stock prediction
    """
    model = Sequential()
    
    # First LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layer
    model.add(Dense(units=25))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
    """
    Train the LSTM model
    """
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history

def make_predictions(model, test_data, scaler, time_steps=60):
    """
    Make predictions using the trained model
    """
    # Create X_test
    X_test = []
    y_test = test_data[time_steps:, 0]
    
    for i in range(time_steps, len(test_data)):
        X_test.append(test_data[i-time_steps:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Get predictions
    predictions = model.predict(X_test)
    
    # Inverse transform to get actual prices
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return predictions, y_test_actual

def evaluate_model(predictions, actual):
    """
    Evaluate model performance
    """
    # Calculate RMSE
    rmse = np.sqrt(np.mean(((predictions - actual) ** 2)))
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return rmse, mape

def plot_results(df, predictions, actual, feature='Close', ticker='Stock'):
    """
    Plot the predictions vs actual prices
    """
    plt.figure(figsize=(16, 8))
    
    # Plot training data
    train = df[:len(df)-len(predictions)]
    valid = df[len(df)-len(predictions):]
    
    # Create a DataFrame for visualization
    valid_df = valid.copy()
    valid_df['Predictions'] = predictions
    
    plt.title(f'{ticker} {feature} Price Prediction')
    plt.xlabel('Date')
    plt.ylabel(f'{feature} Price (₹)')  # Changed to ₹ for Indian stocks
    plt.plot(train[feature])
    plt.plot(valid_df[[feature, 'Predictions']])
    plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
    
    # Save the plot
    plt.savefig(f"{ticker}_prediction.png")
    return plt

def forecast_future(model, last_sequence, scaler, days_to_predict=30):
    """
    Forecast future prices beyond the available data
    """
    future_predictions = []
    current_sequence = last_sequence.reshape((1, last_sequence.shape[0], 1))
    
    for _ in range(days_to_predict):
        # Get prediction for next day
        next_pred = model.predict(current_sequence)[0]
        future_predictions.append(next_pred[0])
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_pred]], axis=1)
    
    # Convert to actual prices
    future_pred_actual = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    return future_pred_actual

def generate_future_dates(last_date, days):
    """
    Generate future dates for forecasting
    """
    # Handle different date formats
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # Generate future dates (excluding weekends for stock market)
    future_dates = []
    current_date = last_date
    days_added = 0
    
    while days_added < days:
        current_date += timedelta(days=1)
        # Skip weekends (5: Saturday, 6: Sunday)
        if current_date.weekday() < 5:  
            future_dates.append(current_date)
            days_added += 1
    
    return pd.DatetimeIndex(future_dates)

def run_stock_prediction(ticker, feature='Close', time_steps=60, test_size=0.2, epochs=50, forecast_days=30):
    """
    Run the full stock prediction pipeline
    """
    # 1. Get data
    df = get_stock_data(ticker, period="1y", interval="1d")
    if df is None:
        return None
    
    # 2. Preprocess data
    train_data, test_data, scaler = preprocess_data(df, feature, test_size)
    
    # 3. Create sequences
    X_train, y_train = create_sequences(train_data, time_steps)
    print(f"\nTraining data shape: X: {X_train.shape}, y: {y_train.shape}")
    
    # 4. Build model
    model = build_lstm_model((X_train.shape[1], 1))
    print("\nModel Summary:")
    model.summary()
    
    # 5. Train model
    model, history = train_model(model, X_train, y_train, epochs=epochs)
    
    # 6. Make predictions
    predictions, y_test_actual = make_predictions(model, test_data, scaler, time_steps)
    
    # 7. Evaluate model
    rmse, mape = evaluate_model(predictions, y_test_actual)
    
    # 8. Plot results
    plt_obj = plot_results(df, predictions, y_test_actual, feature, ticker)
    plt_obj.show()
    
    # 9. Forecast future
    last_sequence = test_data[-time_steps:]
    future_predictions = forecast_future(model, last_sequence, scaler, forecast_days)
    
    # Create future dates (excluding weekends)
    last_date = df.index[-1]
    future_dates = generate_future_dates(last_date, forecast_days)
    
    # Create DataFrame for future predictions
    future_df = pd.DataFrame(data={
        'Date': future_dates,
        f'Predicted_{feature}': future_predictions.flatten()
    })
    future_df.set_index('Date', inplace=True)
    
    print("\nFuture Predictions:")
    print(future_df.head())
    
    # Save future predictions to CSV
    future_df.to_csv(f"{ticker}_forecast.csv")
    
    # Plot future predictions
    plt.figure(figsize=(16, 8))
    plt.title(f'{ticker} {feature} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel(f'{feature} Price (₹)')
    plt.plot(df[feature][-30:])  # Show last 30 days of historical data
    plt.plot(future_df[f'Predicted_{feature}'])
    plt.legend(['Historical', 'Forecast'], loc='lower right')
    plt.savefig(f"{ticker}_forecast.png")
    plt.show()
    
    return model, scaler, df, future_df

def analyze_trend(future_df, feature='Predicted_Close'):
    """
    Analyze the forecasted trend
    """
    first_price = future_df[feature].iloc[0]
    last_price = future_df[feature].iloc[-1]
    
    change = last_price - first_price
    percent_change = (change / first_price) * 100
    
    trend = "UPWARD" if percent_change > 0 else "DOWNWARD"
    
    print(f"\n===== TREND ANALYSIS =====")
    print(f"Forecast period: {future_df.index[0].date()} to {future_df.index[-1].date()}")
    print(f"Starting price: ₹{first_price:.2f}")
    print(f"Ending price: ₹{last_price:.2f}")
    print(f"Change: ₹{change:.2f} ({percent_change:.2f}%)")
    print(f"Overall trend: {trend}")
    
    # Additional analysis
    max_price = future_df[feature].max()
    min_price = future_df[feature].min()
    max_date = future_df[feature].idxmax().date()
    min_date = future_df[feature].idxmin().date()
    
    print(f"Highest predicted price: ₹{max_price:.2f} on {max_date}")
    print(f"Lowest predicted price: ₹{min_price:.2f} on {min_date}")
    
    # Volatility (standard deviation)
    volatility = future_df[feature].std()
    print(f"Predicted volatility: ₹{volatility:.2f}")
    
    return {
        "trend": trend,
        "percent_change": percent_change,
        "volatility": volatility
    }

if __name__ == "__main__":
    print("==== INDIAN STOCK PRICE PREDICTION SYSTEM ====")
    print("This system will fetch data, build an LSTM model, and forecast stock prices")
    print("--------------------------------------------")
    
    user_input = input("Enter the stock name (e.g., TATAMOTORS): ").strip().upper()
    ticker_symbol = input("Enter stock ticker (e.g., TATAMOTORS.NS): ").strip().upper()
    
    # Get news (adapted from original pipeline)
    news_summary = get_news_summary(user_input, ticker_symbol)
    
    # Run stock prediction (for ticker)
    print("\n=== Running LSTM Stock Prediction Model ===")
    model, scaler, historical_df, future_df = run_stock_prediction(
        ticker=ticker_symbol,
        feature='Close',
        time_steps=60,
        test_size=0.2,
        epochs=50,
        forecast_days=30
    )
    
    # Analyze the forecasted trend
    trend_analysis = analyze_trend(future_df)
    
    print("\n=== PREDICTION COMPLETE ===")
    print(f"Files saved:")
    print(f"- {ticker_symbol}_data.csv: Historical data")
    print(f"- {ticker_symbol}_forecast.csv: Forecasted prices")
    print(f"- {ticker_symbol}_prediction.png: Model validation plot")
    print(f"- {ticker_symbol}_forecast.png: Future forecast plot")