import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the sales data from the CSV file
store_sales = pd.read_csv("doc/data.csv")

# Preprocess the date column
store_sales['day'] = pd.to_datetime(store_sales['day'])

# Aggregate the sales data on a monthly basis
monthly_sales = store_sales.resample('M', on='day').sum()

# Create a feature for the month
monthly_sales['month'] = monthly_sales.index.month

# Create lag features for time series forecasting
for i in range(1, 13):
    monthly_sales[f'sales_lag_{i}'] = monthly_sales['total_sales'].shift(i)

# Drop null values due to lag features
monthly_sales = monthly_sales.dropna()

print(monthly_sales)
# Split the data into features and target
X = monthly_sales.drop(['total_sales', 'month'], axis=1)
y = monthly_sales['total_sales']

# Split the data into training and testing sets
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Initialize and fit the RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions for December 2023
dec_2023_features = X.loc['2023-11-30'].values.reshape(1, -1)
dec_2023_sales = rf_model.predict(dec_2023_features)

next_3_months_features = X.iloc[-7:, :].values  # Select the last 3 months' features
next_3_months_sales = rf_model.predict(next_3_months_features)

print(next_3_months_sales)
# Visualize the Predictions
plt.figure(figsize=(15,7))
plt.plot(monthly_sales.index, monthly_sales['total_sales'], label="Original Sales")
plt.scatter(pd.date_range(start=monthly_sales.index[-1], periods=8, freq='M')[1:], next_3_months_sales, color='g', label="Predicted Sales for the next 3 months")
plt.title("Customer Sales Forecast for the Next 3 Months using Random Forest")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales", "Predicted Sales for the next 7 months"])
plt.show()
# print('Predicted sales for December 2023:', dec_2023_sales[0])


# plt.figure(figsize=(15,7))
# plt.plot(monthly_sales.index, monthly_sales['total_sales'], label="Original Sales")
# plt.axvline(mdates.date2num(pd.to_datetime('2023-11-30')), color='r', linestyle='--', label='Prediction Start')
# plt.scatter(mdates.date2num(pd.to_datetime('2023-12-31')), dec_2023_sales, color='g', label="Predicted Sales for Dec 2023")
# plt.title("Customer Sales Forecast using Random Forest")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend()
# plt.show()