# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Mount Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')

# Load Titanic dataset
df1 = pd.read_csv('/content/drive/MyDrive/titanic_dataset.csv')

# Display the first few rows and columns of the dataset
print(df1.head())
print(df1.columns)

# Plot histogram for passenger age distribution
sns.histplot(data=df1, x='Age', bins=20)
plt.title('Passenger Age Distribution')
plt.show()

# Bar plot for survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df1)
plt.title('Survival Rate by Gender')
plt.show()

# Count plot for the number of Siblings/Spouses Aboard
sns.countplot(x='SibSp', data=df1)
plt.title('Number of Siblings/Spouses Aboard')
plt.show()

# Count plot for the number of Parents/Children Aboard
sns.countplot(x='Parch', data=df1)
plt.title('Number of Parents/Children Aboard')
plt.show()

# Fill missing values in 'Age' and 'Fare' columns with median values
df1['Age'].fillna(df1['Age'].median(), inplace=True)
df1['Fare'].fillna(df1['Fare'].median(), inplace=True)

# Map 'Sex' to numeric values ('male': 0, 'female': 1)
df1['Sex'] = df1['Sex'].map({'male': 0, 'female': 1})

# Split the dataset into features (X) and target variable (y)
X = df1.drop('Survived', axis=1)
y = df1['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Drop rows with missing values in 'Age' and 'Fare' columns
df1.dropna(subset=['Age', 'Fare'], inplace=True)

# Impute missing values in the training and testing sets with median values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train the RandomForestClassifier on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
