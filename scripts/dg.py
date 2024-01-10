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

store_sales['day'] = pd.to_datetime(store_sales['day'])
store_sales['day'] = store_sales['day'].dt.to_period('M')
monthly_sales = store_sales.groupby('day').sum().reset_index()

monthly_sales['day'] = monthly_sales['day'].dt.to_timestamp()
monthly_sales.head(10)

monthly_sales['sales_diff'] = monthly_sales['total_sales'].diff()
monthly_sales = monthly_sales.dropna()

supverised_data = monthly_sales.drop(['day','total_sales', 'average_order_value', 'orders'], axis=1)




# supverised_data = monthly_sales
print("first", supverised_data)
for i in range(1,13):
    col_name = 'month_' + str(i)
    supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
supverised_data = supverised_data.dropna().reset_index(drop=True)

print(supverised_data)

train_data = supverised_data[:-1]
test_data = supverised_data[-35:]
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


sales_dates = monthly_sales['day'][-35:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
print(predict_df)

act_sales = monthly_sales['total_sales'][-36:].to_list()

rf_model = RandomForestRegressor(n_estimators=9000, max_depth=20)
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



# Load the sales data from the CSV file
store_sales = pd.read_csv("../doc/data.csv")

# Preprocess the date column
store_sales['day'] = pd.to_datetime(store_sales['day'])

# Aggregate the sales data on a monthly basis
next_monthly_sales = store_sales.resample('M', on='day').sum()

# Create a feature for the month
next_monthly_sales['month'] = next_monthly_sales.index.month

# Create lag features for time series forecasting
for i in range(1, 13):
    next_monthly_sales[f'sales_lag_{i}'] = next_monthly_sales['total_sales'].shift(i)

# Drop null values due to lag features
next_monthly_sales = next_monthly_sales.dropna()

print("-------===========")
print(next_monthly_sales)

X = next_monthly_sales.drop(['total_sales', 'month'], axis=1)
y = next_monthly_sales['total_sales']

# Split the data into training and testing sets
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print("1-------------------")
print(X_train)
print("2--------------")
print(y_train)
# Initialize and fit the RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print(X)
# Make predictions for December 2023
dec_2023_features = X.loc['2023-11-30'].values.reshape(1, -1)
dec_2023_sales = rf_model.predict(dec_2023_features)

next_3_months_features = X.iloc[-7:, :].values  # Select the last 3 months' features
next_3_months_sales = rf_model.predict(next_3_months_features)


# rf_rmse = np.sqrt(mean_squared_error(predict_df['rf_pred'], monthly_sales['total_sales'][-31:]))
# rf_mae = mean_absolute_error(predict_df['rf_pred'], monthly_sales['total_sales'][-31:])
# rf_r2 = r2_score(predict_df['rf_pred'], monthly_sales['total_sales'][-31:])
# print('Random Forest RMSE: ', rf_rmse)
# print('Random Forest MAE: ', rf_mae)
# print('Random Forest R2 Score: ', rf_r2)


plt.figure(figsize=(15,7))
plt.plot(monthly_sales['day'], monthly_sales['total_sales'])
plt.plot(predict_df['day'], predict_df['rf_pred'])
plt.scatter(pd.date_range(start=next_monthly_sales.index[-1], periods=8, freq='M')[1:], next_3_months_sales, color='g', label="Predicted Sales for the next 7 months")
plt.title("Customer Sales Forecast using Random Forest")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales", "Predicted Sales for the next 7 months"])
plt.show()

# act_sales = monthly_sales['total_sales'][-13:].to_list()

