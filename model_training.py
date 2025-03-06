import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# List of ticker symbols for the companies
ticker_symbols = ['AAPL', 'GOOGL', 'MSFT']

# Function to evaluate the model for each stock
def evaluate_stock_model(ticker_symbol):
    # Load the dataset for the specific stock
    dataset = pd.read_csv(f'{ticker_symbol}_stock_data_processed.csv')
    
    # Define features and target variable
    features = dataset[['MA_50', 'MA_200', 'EMA_50', 'EMA_200', 'RSI']]
    target = dataset['Close']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = regressor.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print(f'{ticker_symbol} Mean Absolute Error: {mae}')
    print(f'{ticker_symbol} Mean Squared Error: {mse}')

# Loop through each ticker symbol and evaluate the model
for ticker_symbol in ticker_symbols:
    evaluate_stock_model(ticker_symbol)
