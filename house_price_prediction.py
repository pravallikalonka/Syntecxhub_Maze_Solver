# House Price Prediction using Linear Regression

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
# Replace with your dataset file name
data = pd.read_csv("housing.csv")

# Display first 5 rows
print("Dataset Preview:\n")
print(data.head())

# Check missing values
print("\nMissing Values:\n")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

# Select features and target
# Change column names according to your dataset
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("RMSE:", rmse)
print("R2 Score:", r2)

# Model coefficients
print("\nModel Coefficients")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Save model
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")

# Example prediction
example = [[2000, 3, 2]]  # area, bedrooms, bathrooms
prediction = model.predict(example)

print("\nExample Prediction")
print("Predicted House Price:", prediction[0])