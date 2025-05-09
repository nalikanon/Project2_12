# Project2_GroupX.ipynb
# Group Members:
# 1. 65123456 - สมชาย ใจดี
# 2. 65123457 - สุพรรณษา ดีงาม

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(['customerID'], axis=1, inplace=True)

# Encode 'Churn'
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# แยก X, y
y = df['Churn']
X = df.drop('Churn', axis=1)
X = pd.get_dummies(X, drop_first=True)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สเกลข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# เทรนโมเดล
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ประเมินผล
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# บันทึกโมเดล
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
