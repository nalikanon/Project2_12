# Project2_12.ipynb

# Group Members:
# 1. 6531501132 - Ananya Samyo
# 2. 6531501136 - Apisit Yambangyang
# 3. 6531501138 - Kullaporn promlachai
# 4. 6531501145 - Nathida Thanauppatham
# 5. 6531501163 - Peerawit Kubang

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

# project2_groupX.py
import streamlit as st

# โหลดโมเดล
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# โหลดข้อมูลเพื่อดู features
df = pd.read_csv("Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(['customerID'], axis=1, inplace=True)
df['Churn'] = pd.factorize(df['Churn'])[0]
X = df.drop('Churn', axis=1)
X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns

# UI ของแอป
st.title("📊 Project2_GroupX - พยากรณ์การยกเลิกบริการลูกค้า")

# ใส่ข้อมูลลูกค้า
input_data = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

# ปุ่มทำนาย
if st.button("🔮 ทำนาย"):
    user_input = np.array(input_data).reshape(1, -1)
    scaled = scaler.transform(user_input)
    prediction = model.predict(scaled)[0]
    st.success("ลูกค้า **จะยกเลิกบริการ**" if prediction == 1 else "ลูกค้า **จะไม่ยกเลิกบริการ**")
