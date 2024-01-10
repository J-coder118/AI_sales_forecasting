import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from datetime import datetime, date
from flask import Flask, request, jsonify, Response


store_sales = pd.read_csv("../doc/data.csv")

# store_sales['day'] = pd.to_datetime(store_sales['day'])
# store_sales['day'] = store_sales['day'].dt.to_period('M')
# monthly_sales = store_sales.groupby('day').sum().reset_index()

# monthly_sales['day'] = monthly_sales['day'].dt.to_timestamp()
# monthly_sales.head(10)

store_sales['sales_diff'] = store_sales['total_sales'].diff()
store_sales = store_sales.dropna()

supverised_data = store_sales.drop(['day','total_sales', 'average_order_value', 'orders'], axis=1)




# supverised_data = monthly_sales
print("first", supverised_data)
for i in range(1,32):
    col_name = 'day_' + str(i)
    supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
supverised_data = supverised_data.dropna().reset_index(drop=True)

print(supverised_data)

train_data = supverised_data[:-1]
test_data = supverised_data[-1400:]
print('Train Data Shape:', train_data.shape)
print('Test Data Shape:', test_data.shape)

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()
print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Shape:', y_test.shape)


sales_dates = store_sales['day'][-1400:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
print(predict_df)

act_sales = store_sales['total_sales'][-1401:].to_list()

rf_model = RandomForestRegressor(n_estimators=200, max_depth=20)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_pred = rf_pred.reshape(-1,1)
rf_pred_test_set = np.concatenate([rf_pred,X_test], axis=1)
rf_pred_test_set = scaler.inverse_transform(rf_pred_test_set)

result_list = []
for index in range(0, len(rf_pred_test_set)):
    result_list.append(rf_pred_test_set[index][0] + act_sales[index])
rf_pred_series = pd.Series(result_list, name='rf_pred')
predict_df = predict_df.merge(rf_pred_series, left_index=True, right_index=True)



# # # Load the sales data from the CSV file
# store_sales = pd.read_csv("doc/data.csv")

# # Preprocess the date column
# store_sales['day'] = pd.to_datetime(store_sales['day'])

# # Aggregate the sales data on a monthly basis
# next_monthly_sales = store_sales.resample('M', on='day').sum()

# # Create a feature for the month
# next_monthly_sales['month'] = next_monthly_sales.index.month

# # Create lag features for time series forecasting
# for i in range(1, 13):
#     next_monthly_sales[f'sales_lag_{i}'] = next_monthly_sales['total_sales'].shift(i)

# # Drop null values due to lag features
# next_monthly_sales = next_monthly_sales.dropna()

# print("------------------------")
# print(next_monthly_sales)

# X = next_monthly_sales.drop(['total_sales', 'month'], axis=1)
# y = next_monthly_sales['total_sales']

# # Split the data into training and testing sets
# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# # Initialize and fit the RandomForestRegressor model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Make predictions for December 2023
# dec_2023_features = X.loc['2023-11-30'].values.reshape(1, -1)
# dec_2023_sales = rf_model.predict(dec_2023_features)

# next_3_months_features = X.iloc[-1:, :].values  # Select the last 3 months' features
# next_3_months_sales = rf_model.predict(next_3_months_features)






# Read the data
# store_sales = pd.read_csv("doc/data.csv")

# sales_data = pd.read_csv('doc/data.csv')

# # Convert date to numerical representation
# sales_data['day'] = pd.to_datetime(sales_data['day'])
# sales_data['date_numeric'] = (sales_data['day'] - datetime(1970, 1, 1)).dt.total_seconds()

# # Assuming the dataset has 'date' and 'sales' columns
# # Replace with actual column names if different
# X = sales_data['date_numeric'].values.reshape(-1, 1)
# y = sales_data['total_sales'].values

# # Initialize the TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=5)

# # Initialize the RandomForestRegressor
# model = RandomForestRegressor()

# # Perform TimeSeriesSplit and train the model
# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model.fit(X_train, y_train)

# # Predict the 2023-12 daily sales data
# # Convert '2023-12' to numerical representation
# prediction_date_numeric = (datetime(2023, 12, 1) - datetime(1970, 1, 1)).total_seconds()
# predicted_sales = model.predict(np.array([[prediction_date_numeric]]))

# print(predicted_sales)
# # Draw the graph
# plt.plot(sales_data['day'], sales_data['total_sales'], label='Actual Sales')
# plt.axvline(x=prediction_date_numeric, color='r', linestyle='--', label='Prediction Point')
# plt.scatter(prediction_date_numeric, predicted_sales, color='r', label='Predicted Sales')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.title('Daily Sales Data')
# plt.legend()
# plt.show()


# Load the data
store_sales = pd.read_csv("../doc/data.csv")

# Preprocess the date column
store_sales['day'] = pd.to_datetime(store_sales['day'])

# Create lag features for time series forecasting
for i in range(1, 32):  # For daily sales forecasting
    store_sales[f'sales_lag_{i}'] = store_sales['total_sales'].shift(i)

# Drop null values due to lag features
store_sales = store_sales.dropna()

# store_sales = store_sales.iloc[:, 1:]
print("-------------")
print(store_sales)

X = store_sales.drop(['total_sales', 'orders'], axis=1)
y = store_sales['total_sales']


# Convert datetime columns to numerical values
X_numeric = X.select_dtypes(include=['number'])
X_datetime = X.select_dtypes(include=['datetime64'])
X_datetime_numeric = X_datetime.apply(lambda x: x.astype('int64') // 10**9)  # Convert datetime to numerical
X = pd.concat([X_numeric, X_datetime_numeric], axis=1)


# Split the data into training and testing sets
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print(X)


next_3_months_features = X.iloc[-10:, :].values  # Select the last 3 months' features
next_3_months_sales = rf_model.predict(next_3_months_features)


# Plot the results
# plt.figure(figsize=(15,7))
# plt.plot(store_sales['day'], store_sales['total_sales'], label='Actual Sales')
# # plt.plot(pd.date_range(start='2023-11-30', periods=32, freq='D')[1:], rf_model.predict(X), label='Predicted Sales')
# # plt.scatter(pd.date_range(start='2023-11-30', periods=32, freq='D')[31:], dec_2023_sales, color='g', label="Predicted Sales for December 2023")
# plt.scatter(pd.date_range(start=store_sales.index[-1], periods=32, freq='D')[31:], next_3_months_sales, color='g', label="Predicted Sales for December 2023")

# plt.title("Customer Sales Forecast using Random Forest")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend()
# plt.show()

# Plot the results
# plt.figure(figsize=(15,7))
# plt.plot(store_sales['day'], store_sales['total_sales'], label='Actual Sales')
# plt.scatter(pd.date_range(start=store_sales.index[-1], periods=32, freq='D')[31:], next_3_months_sales, color='g', label="Predicted Sales for December 2023")
# plt.title("Customer Sales Forecast using Random Forest")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend()
# plt.show()
next_3_months_dates = pd.date_range(start=store_sales['day'].iloc[-1], periods=31, freq='D')[31:]
print("---------------------------------")
print(next_3_months_dates, next_3_months_sales)
plt.figure(figsize=(15,7))
plt.plot(store_sales['day'], store_sales['total_sales'], label='Actual Sales')
plt.plot(next_3_months_dates, next_3_months_sales, color='g', label="Predicted Sales for December 2023")
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()