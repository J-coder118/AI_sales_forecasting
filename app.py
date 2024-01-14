import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from quart_cors import cors
from quart import Quart, jsonify, request, websocket, g, abort, Response
import pandas as pd
import numpy as np
import uvicorn
import psycopg2
import csv
import io
import json
from datetime import datetime, date

app = Quart(__name__)
app = cors(app)

def create_connection():
    try:
        print("connection completed")
        return psycopg2.connect(
            host='localhost',
            port='5433',
            user='postgres',
            password='postgres',
            database='Data-Projection'
        )
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")


# Hello
def load_data():
    try:
        store_sales = pd.read_csv("doc/data.csv")
        store_sales['day'] = pd.to_datetime(store_sales['day'])
        return store_sales, "2023-11-30"
    except Exception as e:
        print(f"Error retrieving sales data: {e}")

    try:
        connection = create_connection()
        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM sales_data"
        )
        result = cursor.fetchall()
        cursor.close()
        connection.close()

        csv_data = io.StringIO()
        csv_writer = csv.writer(csv_data)
        
        # Write the header row
        csv_writer.writerow(['day', 'total_sales', 'average_order_value', 'orders'])
        
        # Write the data rows
        csv_writer.writerows(result)
        response = Response(csv_data.getvalue(), content_type='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=sales_data.csv'
        
        return response, "2023-11-30"
        
    except Exception as e:
        print(f"Error retrieving sales data: {e}")


def daily_sales(data):
    return ""

def monthly_sales(data):
    """Returns a dataframe where each row represents total sales for a given
    month. Columns include 'date' by month and 'sales'.
    """
    monthly_data = data.copy()
    # Drop the day indicator from the date column
    monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-3])
    # Sum sales per month
    monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
    monthly_data.date = pd.to_datetime(monthly_data.date)
    monthly_data.to_csv('../data/monthly_data.csv')

    return monthly_data

def get_diff(data):
    """Returns the dataframe with a column for sales difference between each
    month. Results in a stationary time series dataframe. Prior EDA revealed
    that the monthly data was not stationary as it had a time-dependent mean.
    """
    data['sales_diff'] = data.sales.diff()
    data = data.dropna()
    data.to_csv('../data/stationary_df.csv')

    return data


def generate_supervised(data):
    """Generates a csv file where each row represents a month and columns
    include sales, the dependent variable, and prior sales for each lag. Based
    on EDA, 12 lag features are generated. Data is used for regression modeling.

    Output df:
    month1  sales  lag1  lag2  lag3 ... lag11 lag12
    month2  sales  lag1  lag2  lag3 ... lag11 lag12
    """
    supervised_df = data.copy()
    #create column for each lag
    for i in range(1, 13):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['sales_diff'].shift(i)

    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    supervised_df.to_csv('../data/model_df.csv', index=False)


def tts(data):
    """Splits the data into train and test. Test set consists of the last 12
    months of data.
    """
    data = data.drop(['sales', 'date'], axis=1)
    train, test = data[0:-12].values, data[-12:].values

    return train, test

def scale_data(train_set, test_set):
    """Scales data using MinMaxScaler and separates data into X_train, y_train,
    X_test, and y_test.

    Keyword Arguments:
    -- train_set: dataset used to train the model
    -- test_set: dataset used to test the model
    """

    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

    return X_train, y_train, X_test, y_test, scaler



@app.route('/sales-data', methods=['GET'])
async def sales_data():
    try:
        start = request.args.get('start_date')
        end = request.args.get('end_date')

        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")

        store_sales, last_db_date = load_data()
        last_db_date = datetime.strptime(last_db_date, "%Y-%m-%d")

        # Calculate the difference in days
        num_days = (end_date - last_db_date).days

        for i in range(1, 32):  # For daily sales forecasting
            store_sales[f'sales_lag_{i}'] = store_sales['total_sales'].shift(i)
        # Drop null values due to lag features
        store_sales = store_sales.dropna()

        X = store_sales.drop(['total_sales', 'orders'], axis=1)
        y = store_sales['total_sales']

        # Convert datetime columns to numerical values
        X_numeric = X.select_dtypes(include=['number'])
        X_datetime = X.select_dtypes(include=['datetime64'])
        X_datetime_numeric = X_datetime.apply(lambda x: x.astype('int64') // 10**9)  # Convert datetime to numerical
        X = pd.concat([X_numeric, X_datetime_numeric], axis=1)

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        predict_day_features = X.iloc[-num_days:, :].values  # Select the last 3 months' features
        predict_day_sales = rf_model.predict(predict_day_features)

        return jsonify({
            'status': "success",
            'response': predict_day_sales.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)