import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a DataFrame with user input data
user_data = {
    'CPU_Usage': float(input("Enter CPU Usage (%): ")),
    'RAM_Usage': float(input("Enter RAM Usage (%): ")),
    'Storage_Usage': float(input("Enter Storage Usage (%): ")),
    'Charging_Cycles': int(input("Enter Charging Cycles: ")),
    'Battery_Health': float(input("Enter Battery Health (%): ")),
    'Battery_Life': float(input("Enter Actual Battery Life (hours): "))
}
user_df = pd.DataFrame([user_data])

# Ensure valid and non-null feature names
user_df.columns = ['CPU_Usage', 'RAM_Usage', 'Storage_Usage', 'Charging_Cycles', 'Battery_Health', 'Battery_Life']

# Split the user data into features (X) and target variable (y)
X_user = user_df.drop(['Battery_Life'], axis=1)
y_user = user_df['Battery_Life']

# Load the full dataset (replace 'data.csv' with your actual dataset)
data = pd.read_csv('data.csv')

# Split the full data into features (X) and target variable (y)
X = data.drop(['Battery_Life'], axis=1)
y = data['Battery_Life']

# Initialize and train the linear regression model using the full dataset
model = LinearRegression()
model.fit(X, y)

# Predict the battery life for the user's data
predicted_battery_life = model.predict(X_user)

# Print the prediction in words
print(f"Estimated Battery Life: {predicted_battery_life[0]:.2f} hours")
