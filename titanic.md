import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from google.colab import drive
drive.mount('/content/drive')
df1=pd.read_csv('/content/drive/MyDrive/titanic_dataset.csv')
print(df1.head())
print(df1.keys())
sns.histplot(data=df1, x='Age', bins=20)
plt.title('Passenger gender Distribution')
plt.show()
sns.barplot(x='Sex', y='Survived', data=df1)
plt.title('Survival Rate by Gender')
plt.show()
sns.countplot(x='SibSp', data=df1)
plt.title('Number of Siblings/Spouses Aboard')
plt.show()

sns.countplot(x='Parch', data=df1)
plt.title('Number of Parents/Children Aboard')
plt.show()

df1['Age'].fillna(df1['Age'].median(), inplace=True)
df1['Fare'].fillna(df1['Fare'].median(), inplace=True)
df1['Sex'] = df1['Sex'].map({'male': 0, 'female': 1})
X = df1.drop('Survived', axis=1)
y = df1['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'model = RandomForestClassifier(n_estimators=100, random_state=42)
