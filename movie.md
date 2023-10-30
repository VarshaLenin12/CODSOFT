import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your movie dataset (replace 'your_movie_data.csv' with your dataset file)
data = pd.read_csv('your_movie_data.csv')

# Data Preprocessing
# Perform data cleaning, handle missing values, encode categorical variables, and feature engineering as needed.

# Data Exploration
# Explore the dataset to gain insights into relationships between features and ratings. Use visualizations for analysis.

# Feature Selection
# Determine which features are most relevant for predicting movie ratings.

# Split the data into features (X) and target variable (y)
X = data[['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3', 'Budget', 'Runtime', 'ReleaseYear']]
y = data['Rating']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a regression model (Random Forest Regressor in this example)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Deploy the model for predicting movie ratings

# You can also visualize the predicted ratings against the actual ratings
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs. Predicted Ratings")
plt.show()
