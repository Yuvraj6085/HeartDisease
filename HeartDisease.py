import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

csv_url = "https://raw.githubusercontent.com/plaguedoc000/framingham-heart-study-dataset/main/framingham.csv"
disease_df = pd.read_csv(csv_url)
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)

print(disease_df)
print(disease_df.TenYearCHD.value_counts())

X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.3, random_state=4)

print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# ✅ Save the model and scaler using pickle
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(logreg, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Load back for testing
with open("heart_disease_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)

# Prediction
y_pred = loaded_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('Accuracy of the model is =', accuracy_score(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
