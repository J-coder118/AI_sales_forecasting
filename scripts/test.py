from datetime import datetime

start_date_str = "2023.11.30"
end_date_str = "2024.02.03"

# Convert the date strings to datetime objects
start_date = datetime.strptime(start_date_str, "%Y.%m.%d")
end_date = datetime.strptime(end_date_str, "%Y.%m.%d")

# Calculate the difference in days
num_days = (end_date - start_date).days

print("The number of days between the start date and end date is:", num_days)