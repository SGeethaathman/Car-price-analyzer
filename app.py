import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Car Analyzer 2026 🚗", layout="centered")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open('model/car_price_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoders']

model, encoders = load_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv('car_data_2026.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.dropna(inplace=True)

    # Split brand & model
    df[['brand', 'model']] = df['name'].str.split(' ', n=1, expand=True)

    return df

df = load_data()

# =========================
# PRICE CONVERTER
# =========================
def convert_price(price):
    price = str(price).lower().replace(',', '').replace('₹', '').strip()

    # Handle range (take max)
    if '-' in price:
        price = price.split('-')[-1].strip()

    if 'lakh' in price or 'la' in price:
        return float(price.replace('lakh', '').replace('la', '').strip()) * 100000
    elif 'crore' in price or 'cr' in price:
        return float(price.replace('crore', '').replace('cr', '').strip()) * 10000000
    else:
        return float(price)

# =========================
# UI
# =========================
st.title("Car Prize Analyzer (2026)")
st.markdown("Complete car insights using ML + real-world data")
st.divider()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(df['brand'].unique()))
    model_name = st.selectbox(
        "Model",
        sorted(df[df['brand'] == brand]['model'].dropna().unique())
    )

    car_name = brand + " " + model_name

    year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True))
    km_driven = st.number_input("KM Driven", 100, 500000, 45000, 1000)

with col2:
    fuel = st.selectbox("Fuel", df['fuel'].unique())
    seller_type = st.selectbox("Seller Type", df['seller_type'].unique())
    transmission = st.selectbox("Transmission", df['transmission'].unique())
    owner = st.selectbox("Owner", df['owner'].unique())

st.divider()

# =========================
# PREDICTION
# =========================
if st.button("🚀 Analyze Car", use_container_width=True):
    try:
        current_year = 2026
        car_age = current_year - year

        # Encode inputs
        name_enc = encoders['name'].transform([car_name])[0]
        fuel_enc = encoders['fuel'].transform([fuel])[0]
        seller_enc = encoders['seller_type'].transform([seller_type])[0]
        trans_enc = encoders['transmission'].transform([transmission])[0]
        owner_enc = encoders['owner'].transform([owner])[0]

        features = np.array([[name_enc, car_age, km_driven,
                              fuel_enc, seller_enc, trans_enc, owner_enc]])

        predicted_price = model.predict(features)[0]

        # =========================
        # CURRENT PRICE + STATUS
        # =========================
        car_row = df[df['name'].str.contains(model_name, case=False, na=False)]

        if not car_row.empty:
            current_price = car_row.iloc[0]['current_price']
            status = "Running ✅"
        else:
            current_price = "N/A"
            status = "Discontinued ❌"

        # Convert safely
        try:
            current_price_val = convert_price(current_price)
            formatted_price = f"₹{current_price_val:,.0f}"
        except:
            formatted_price = str(current_price)

        # =========================
        # MILEAGE
        # =========================
        if 'mileage' in df.columns and not car_row.empty:
            mileage_val = car_row.iloc[0]['mileage']
        else:
            mileage_map = {
                'Petrol': '15–18 km/l',
                'Diesel': '18–22 km/l',
                'CNG': '20–25 km/kg'
            }
            mileage_val = mileage_map.get(fuel, "N/A")

        # =========================
        # DISPLAY METRICS
        # =========================
        st.success("Analysis Complete")

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Resale Price", f"₹{predicted_price:,.0f}")
        col2.metric("🚗 Status", status)
        col3.metric("🏷 Current Price", formatted_price)

        col4, col5, col6 = st.columns(3)
        col4.metric("📅 Age", f"{car_age} yrs")
        col5.metric("🛣 KM Driven", f"{km_driven:,}")
        col6.metric("⛽ Mileage", mileage_val)

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        st.divider()
        st.subheader("📊 What affects the price most?")

        feature_names = [
            'Car Name', 'Car Age', 'Km Driven',
            'Fuel', 'Seller Type', 'Transmission', 'Owner'
        ]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)

            st.bar_chart(imp_df.set_index('Feature'))
        else:
            st.info("Feature importance not available.")

    except Exception as e:
        st.error(f"Error: {e}")

# =========================
# MODEL COMPARISON
# =========================
with st.expander("📊 Model Comparison (from training)"):

    st.markdown("Three models were trained and compared. Best one was saved automatically.")

    comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
        'R² Score': ['~0.67', '~0.91', '~0.89'],
        'Best For': ['Baseline', '✅ Best Accuracy', 'Close second']
    })

    st.dataframe(comparison, use_container_width=True)