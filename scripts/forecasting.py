import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from datetime import datetime, date
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        else:
            return super().default(obj)


@app.route('/', methods=['GET'])
def predict():
    store_sales = pd.read_csv("doc/data.csv")

    store_sales['day'] = pd.to_datetime(store_sales['day'])
    store_sales['day'] = store_sales['day'].dt.to_period('M')
    monthly_sales = store_sales.groupby('day').sum().reset_index()

    monthly_sales['day'] = monthly_sales['day'].dt.to_timestamp()
    monthly_sales.head(10)

    # print(monthly_sales)


    # plt.figure(figsize=(15,5))
    # plt.plot(monthly_sales['day'], monthly_sales['total_sales'])
    # plt.xlabel('Date')
    # plt.xlabel('Sales')
    # plt.title("Monthly Customer Sales")
    # plt.show()

    monthly_sales['sales_diff'] = monthly_sales['total_sales'].diff()
    monthly_sales = monthly_sales.dropna()


    print(monthly_sales)


    # plt.figure(figsize=(15,5))
    # plt.plot(monthly_sales['day'], monthly_sales['sales_diff'])
    # plt.xlabel('Date')
    # plt.xlabel('Sales')
    # plt.title("Monthly Customer Sales Diff")
    # plt.show()


    supverised_data = monthly_sales.drop(['day','total_sales', 'average_order_value', 'orders'], axis=1)




    # supverised_data = monthly_sales

    for i in range(1,13):
        col_name = 'month_' + str(i)
        supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
    supverised_data = supverised_data.dropna().reset_index(drop=True)

    print(supverised_data)

    train_data = supverised_data[:-12]
    test_data = supverised_data[-12:]
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


    sales_dates = monthly_sales['day'][-12:].reset_index(drop=True)
    predict_df = pd.DataFrame(sales_dates)
    print(predict_df)


    act_sales = monthly_sales['total_sales'][-13:].to_list()

    # linreg_model = LinearRegression()
    # linreg_model.fit(X_train, y_train)
    # linreg_pred = linreg_model.predict(X_test)


    # linreg_pred = linreg_pred.reshape(-1, 1)
    # linreg_pred_test_set = np.concatenate([linreg_pred, X_test], axis=1)
    # linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

    # result_list = []
    # for index in range(0, len(linreg_pred_test_set)):
    #     result_list.append(linreg_pred_test_set[index][0] + act_sales[index])
    # linreg_pred_series = pd.Series(result_list, name="linreg_pred")
    # predict_df = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)


    # linreg_rmse = np.sqrt(mean_squared_error(predict_df['linreg_pred'], monthly_sales['total_sales'][-12:]))
    # linreg_mae = mean_absolute_error(predict_df['linreg_pred'], monthly_sales['total_sales'][-12:])
    # linreg_r2 = r2_score(predict_df['linreg_pred'], monthly_sales['total_sales'][-12:])
    # print('Linear Regression RMSE: ', linreg_rmse)
    # print('Linear Regression MAE: ', linreg_mae)
    # print('Linear Regression R2 Score: ', linreg_r2)

    # print(monthly_sales)
    # print(predict_df)
    # plt.figure(figsize=(15,5))
    # plt.plot(monthly_sales['day'], monthly_sales['sales_diff'])
    # plt.plot(predict_df['day'], predict_df['linreg_pred'])
    # plt.title("Customer Sales Forecast using Linear Regression")
    # plt.xlabel("Date")
    # plt.ylabel("Sales")
    # plt.legend(["Original Sales", "Predicted Sales"])
    # plt.show()



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


    rf_rmse = np.sqrt(mean_squared_error(predict_df['rf_pred'], monthly_sales['total_sales'][-12:]))
    rf_mae = mean_absolute_error(predict_df['rf_pred'], monthly_sales['total_sales'][-12:])
    rf_r2 = r2_score(predict_df['rf_pred'], monthly_sales['total_sales'][-12:])
    print('Random Forest RMSE: ', rf_rmse)
    print('Random Forest MAE: ', rf_mae)
    print('Random Forest R2 Score: ', rf_r2)


    plt.figure(figsize=(15,7))
    plt.plot(monthly_sales['day'], monthly_sales['total_sales'])
    plt.plot(predict_df['day'], predict_df['rf_pred'])
    plt.title("Customer Sales Forecast using Random Forest")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(["Original Sales", "Predicted Sales"])
    plt.show()


    # xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror')
    # xgb_model.fit(X_train, y_train)
    # xgb_pred = xgb_model.predict(X_test)


    # xgb_pred = xgb_pred.reshape(-1,1)
    # xgb_pred_test_set = np.concatenate([xgb_pred,X_test], axis=1)
    # xgb_pred_test_set = scaler.inverse_transform(xgb_pred_test_set)

    # result_list = []
    # for index in range(0, len(xgb_pred_test_set)):
    #     result_list.append(xgb_pred_test_set[index][0] + act_sales[index])
    # xgb_pred_series = pd.Series(result_list, name='xgb_pred')
    # predict_df = predict_df.merge(xgb_pred_series, left_index=True, right_index=True)


    # xgb_rmse = np.sqrt(mean_squared_error(predict_df['xgb_pred'], monthly_sales['total_sales'][-12:]))
    # xgb_mae = mean_absolute_error(predict_df['xgb_pred'], monthly_sales['total_sales'][-12:])
    # xgb_r2 = r2_score(predict_df['xgb_pred'], monthly_sales['total_sales'][-12:])
    # print('XG Boost RMSE: ', xgb_rmse)
    # print('XG Boost MAE: ', xgb_mae)
    # print('XG Boost R2 Score: ', xgb_r2)


    # plt.figure(figsize=(15,7))
    # plt.plot(monthly_sales['day'], monthly_sales['total_sales'])
    # plt.plot(predict_df['day'], predict_df['xgb_pred'])
    # plt.title("Customer Sales Forecast using XG Boost")
    # plt.xlabel("Date")
    # plt.ylabel("Sales")
    # plt.legend(["Original Sales", "Predicted Sales"])
    # plt.show()


    # Step 1: Extract Data
    date_values = predict_df['day'].tolist()
    original_sales_values = monthly_sales['total_sales'].tolist()
    predicted_sales_values = predict_df['rf_pred'].tolist()

    # Step 2: Prepare JSON Data
    sales_data = {
        "date": date_values,
        "original_sales": original_sales_values,
        "predicted_sales": predicted_sales_values
    }

    print(sales_data)
    def serialize_timestamp(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()

    # Assuming sales_data is a list of dictionaries
    for item in sales_data:
        if isinstance(item, dict) and 'date' in item:
            item['timestamp'] = item['date']

    # Now you can serialize the data to JSON using the custom function
    sales_data_json = json.dumps(sales_data, default=serialize_timestamp)

    print(sales_data_json)

    return Response(
        response=sales_data_json
    )

@app.route('/assumption', methods=['POST'])
def get_assumptions():
    return ""

if __name__ == '__main__':
    app.run(debug=False)