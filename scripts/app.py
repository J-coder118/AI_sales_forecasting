from quart import Quart, jsonify, request
import psycopg2
import datetime
import uvicorn

app = Quart(__name__)

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

async def get_sales_data(start, end):
    try:
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()

        connection = create_connection()
        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM sales_data WHERE day BETWEEN %s AND %s",
            (start_date, end_date)
        )
        result = cursor.fetchall()
        cursor.close()
        connection.close()

        return [
            {
                'day': row[0].strftime("%m-%d-%Y"),
                'total_sales': int(row[1]),
                'average_order_value': int(row[2]),
                'orders': int(row[3]),
            }
            for row in result
        ]
    except Exception as e:
        print(f"Error retrieving sales data: {e}")

@app.route('/sales-data', methods=['GET'])
async def sales_data():
    try:
        start = request.args.get('start')
        end = request.args.get('end')

        # Validate start and end parameters here if needed

        data = await get_sales_data(start, end)
        response = jsonify({'data': data})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Allow requests from any origin
        return response

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)