import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# List of ticker symbols for the companies
ticker_symbols = ['AAPL', 'GOOGL', 'MSFT']

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Loop through each ticker symbol
for ticker_symbol in ticker_symbols:
    # Fetch the data using yfinance
    data = yf.download(ticker_symbol, start='2020-01-01', end='2023-01-01')

    # Handle missing values
    data = data.dropna()

    # Feature engineering: Calculate Moving Averages (MA)
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()

    # Calculate Exponential Moving Averages (EMA)
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Calculate RSI
    data['RSI'] = calculate_rsi(data)

    # Save processed data to a new CSV file
    data.to_csv(f'{ticker_symbol}_stock_data_processed.csv', index=False)

    # Plot the Closing Price and Moving Averages
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(data.index, data['MA_50'], label='50-Day MA')
    plt.plot(data.index, data['MA_200'], label='200-Day MA')
    plt.plot(data.index, data['EMA_50'], label='50-Day EMA')
    plt.plot(data.index, data['EMA_200'], label='200-Day EMA')
    plt.title(f'{ticker_symbol} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Plot the Relative Strength Index (RSI)
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['RSI'], label='RSI')
    plt.title(f'{ticker_symbol} Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()
