import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load('car_price_model.pkl')

model = load_model()

st.title("Предсказание цены автомобиля")

with st.form("car_form"):
    name = st.text_input("Название", "Maruti Swift Dzire VDI")
    year = st.number_input("Год", min_value=1990, max_value=2025, value=2014)
    km_driven = st.number_input("Пробег", min_value=0, value=45000)
    fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("КПП", ["Manual", "Automatic"])
    owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    mileage = st.text_input("Mileage", "23.4 kmpl")
    engine = st.text_input("Engine", "1248 CC")
    max_power = st.text_input("Max power", "74 bhp")
    torque = st.text_input("Torque", "190Nm@ 2000rpm")
    seats = st.number_input("Seats", min_value=2, max_value=14, value=5)

    submitted = st.form_submit_button("Предсказать")

if submitted:
    input_df = pd.DataFrame([{
        'name': name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'torque': torque,
        'seats': seats
    }])

    pred = model.predict(input_df)[0]
    st.success(f"Предсказанная цена: {pred:.2f}")
