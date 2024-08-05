# -*- coding: utf-8 -*-
"""(London) Final Summative Group Project: Implementation and Testing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XBdQM-qJWv33rwPf71LVc48exoaT15sQ

# (LONDON MARKET) FINAL SUMMATIVE GROUP PROJECT
"""

## AUTHORS: Nana Kwaku Amoakoh, Kobina Kyereboah-Coleman
## 5th August 2024
## TODO: To develop an AI model for the prediction of product(cocoa) price for the London Market from the ICCO

"""## Downloading Necessary Modules"""

!pip install scikit-learn
!pip install scikeras
!pip uninstall -y scikit-learn
!pip install scikit-learn==1.0.2 scikeras

"""## Importing Relevant Python Libraries"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
# from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from google.colab import drive
import warnings
warnings.filterwarnings('ignore')

"""## Drive Mount"""

# Mount Google Drive
drive.mount('/content/drive')

"""## Save Dataset To Google Drive"""

# Download the dataset
!wget -O /content/ICCO_daily_prices.csv 'https://raw.githubusercontent.com/nanadotam/ITAI/main/Final-Project/ICCO_daily_prices.csv'

"""## Load Dataset"""

# Define the path to your CSV file in Google Drive
csv_path = '/content/drive/MyDrive/Colab Notebooks/ICCO Daily Prices.csv'

# Load the dataset
# data = pd.read_csv(csv_path)
data = pd.read_csv('/content/ICCO_daily_prices.csv')

"""## Data Preprocessing"""

# Select tuples with entries
data = data.iloc[0:7662]
data.head()

# Function to check for 30% data absence per column
def threshold(dataframe):

    # The data integrity threshold provided (30%)
    threshold = 0.3

    # Calculating threshold count for missing values
    threshold_count = int(threshold * len(dataframe))

    # Pulling out unsafe columns
    unsafe_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > threshold_count]
    if len(unsafe_columns) == 0:
        print("No columns have more than 30% missing values.")
    else:
        print(f"{len(unsafe_columns)} unsafe columns have been identified.")

    return unsafe_columns

threshold(data)

# Function to identify columns with missing values
def find_null_columns(x):

    # Initialize empty list to store columns with null values
    null_columns = []

    # Iterate through all columns in the dataframe
    for col in x.columns:

        # Check if column has any null values
        if x[col].isnull().any():
            # Add column name to list if it contains nulls
            null_columns.append(col)
    if len(null_columns) > 0:
        # Print the list of columns with null values
        print(f"The following columns have null values: {null_columns}")
    else:
        print("No columns have null values.")
    print(null_columns)

find_null_columns(data)

# Convert the 'Date' column to datetime without specifying a format
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# Convert price columns to numeric (remove any commas and convert to float)
price_columns = ['London futures (£ sterling/tonne)', 'New York futures (US$/tonne)',
                 'ICCO daily price (US$/tonne)', 'ICCO daily price (Euro/tonne)']

# Use 'data' consistently instead of mixing it with 'df'
for col in price_columns:
    # Check if the column is of type object (likely string) before applying string operations
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
    else:
        print(f"Column '{col}' is not of string type. Skipping string replacement.")

# Sort the data by date
data = data.sort_values('Date')

# Convert 'Date' column to datetime without specifying a format, handling errors
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# Check for any values that failed to convert (NaT)
failed_conversions = data[data['Date'].isna()]
if not failed_conversions.empty:
    print("Failed to convert the following dates:")
    print(failed_conversions)
    # Handle the failed conversions (e.g., try to fix the format, drop the rows, etc.)
    data = data.dropna(subset=['Date']) # Drop rows with invalid dates

# Continue with the rest of your code...
data['Days'] = (data['Date'] - data['Date'].min()).dt.days # Use df instead of data

# Choose which market to predict (London or New York)
market = 'London futures (£ sterling/tonne)'

# Choose which market to predict (London or New York)
market = 'London futures (£ sterling/tonne)'

# NO CHANGES NEEDED HERE.  The price data was already cleaned in a previous step.
# Clean the price data: remove commas and convert to float
# data[market] = data[market].str.replace(',', '').astype(float)

# Verify the data type of the market column
print(data[market].dtype)

# Display information about the dataframe
data.info()

# Calculate moving averages
data['MA_7'] = data[market].rolling(window=7).mean()
data['MA_30'] = data[market].rolling(window=30).mean()
data['MA_90'] = data[market].rolling(window=90).mean()

# Calculate price differences
data['Price_Diff'] = data[market].diff()

# Calculate volatility measures (standard deviation over rolling windows)
data['Volatility_7'] = data[market].rolling(window=7).std()
data['Volatility_30'] = data[market].rolling(window=30).std()
data['Volatility_90'] = data[market].rolling(window=90).std()

data.to_csv('/content/ldn_cleaned_data.csv', index=False)

# Calculate a 'Days' column if it doesn't exist (assuming you want a simple day counter)
data['Days'] = range(1, len(data) + 1)

# Prepare X and y
X = data[['Days', 'MA_7', 'MA_30', 'MA_90', 'Price_Diff', 'Volatility_7', 'Volatility_30', 'Volatility_90']]
y = data[market]

# Handle missing values in features
X = X.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Run with the adjusted parameters
for i in range(1):
    history = lstm_model.fit(X_train_lstm, y_train, epochs = 100, batch_size = 32, validation_data=(X_test_lstm, y_test), verbose=1)
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    mse_lstm = mean_squared_error(y_test, y_pred_lstm)
    r2_lstm = r2_score(y_test, y_pred_lstm)
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    print(f'LSTM Iteration {i+1}\nMean squared error: {mse_lstm}, R-squared score: {r2_lstm}, Mean absolute error: {mae_lstm}')

# Plot training history for the last iteration
plt.style.use('dark_background')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('LSTM Training and Validation Loss')
plt.legend()
plt.show()

# Plot LSTM results
plt.style.use('dark_background')
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(y_pred_lstm, label='Predicted Prices')
plt.title('LSTM: Actual vs Predicted Prices')
plt.legend()
plt.show()

# Save the LSTM model
model_path = '/content/drive/MyDrive/cocoa_price_lstm_model.keras'
lstm_model.save(model_path)

# Save the scaler separately using pickle
scaler_path = '/content/drive/MyDrive/cocoa_price_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")