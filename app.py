import streamlit as st
import pandas as pd
import joblib

model = joblib.load('car_price_model.pkl')

st.title('Предсказание цены автомобиля')

st.write('Введите параметры автомобиля для получения прогноза')

name = st.text_input('Название машины', 'Maruti Swift')
year = st.number_input('Год выпуска', min_value=1990, max_value=2025, value=2014)
km_driven = st.number_input('Пробег, км', min_value=0, value=45000)
fuel = st.selectbox('Тип топлива', ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Тип продавца', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Коробка передач', ['Manual', 'Automatic'])
owner = st.selectbox('Владелец', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.text_input('Mileage', '23.4 kmpl')
engine = st.text_input('Engine', '1248 CC')
max_power = st.text_input('Max power', '74 bhp')
torque = st.text_input('Torque', '190Nm@ 2000rpm')
seats = st.number_input('Seats', min_value=1.0, max_value=20.0, value=5.0, step=1.0)

if st.button('Предсказать цену'):
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

    prediction = model.predict(input_df)
    st.success(f'Примерная цена автомобиля: {prediction[0]:,.0f}')
