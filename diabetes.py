import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle  # ✅ Added

# Load dataset
data = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(data)

# Quick info
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.columns)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model using pickle
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the model back (optional test)
with open("diabetes_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Predict on test set
y_pred = loaded_model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict user input
print("\n🔍 Enter the following values to predict diabetes:")
Pregnancies = float(input("Pregnancies: "))
Glucose = float(input("Glucose: "))
BloodPressure = float(input("BloodPressure: "))
SkinThickness = float(input("SkinThickness: "))
Insulin = float(input("Insulin: "))
BMI = float(input("BMI: "))
DiabetesPedigreeFunction = float(input("DiabetesPedigreeFunction: "))
Age = float(input("Age: "))

user_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
              Insulin, BMI, DiabetesPedigreeFunction, Age]]

user_result = loaded_model.predict(user_data)

if user_result[0] == 1:
    print("\n✅ Prediction: Patient **has diabetes**.")
else:
    print("\n✅ Prediction: Patient **does not have diabetes**.")
