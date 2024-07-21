import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data = pd.read_csv('house_price.csv')

# Explore the dataset
print(data.head())

# Define features and target
X = data[['size', 'bathrooms', 'bedrooms']]
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compare predictions with actual values
print('Predicted prices:', y_pred)
print('Actual prices:', y_test.values)

# Plot the results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['size'], data['bathrooms'], data['price'], color='blue', label='Bathrooms')
ax.scatter(data['size'], data['bedrooms'], data['price'], color='green', label='Bedrooms')

ax.set_xlabel('Size (sq ft)')
ax.set_ylabel('Bathrooms')
ax.set_zlabel('Price ($)')

plt.title('House Price Prediction')
plt.legend()
plt.show()

# Print model coefficients
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
