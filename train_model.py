import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

# Load dataset
df = pd.read_csv('car_data_2026.csv')

# Clean columns
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

df.dropna(inplace=True)

# Feature engineering
current_year = 2026
df['car_age'] = current_year - df['year']

# Encode categorical columns
encoders = {}
for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & target
X = df[['name', 'car_age', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = df['selling_price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"R²: {r2_score(y_test, preds):.3f}")
print(f"MAE: ₹{mean_absolute_error(y_test, preds):,.0f}")

# Save model
os.makedirs('model', exist_ok=True)

with open('model/car_price_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoders': encoders}, f)

print("✅ Model saved")