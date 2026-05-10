import streamlit as st
import pandas as pd
import joblib
import numpy as np

def car_data_cleaner(X):
    df = X.copy()
    
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str).apply(lambda x: x.split()[0])
    
    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(r'[^0-9.]', '', regex=True)
                       .replace('', np.nan)
                       .astype(float))
            
    if 'torque' in df.columns:
        s = df['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
        is_kgm = s.str.contains('kgm', na=False)
        
        df['max_torque_rpm'] = s.str.findall(r'\d+').str[-1].apply(lambda x: float(x) if isinstance(x, str) else np.nan)
        df['torque'] = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
        
        df.loc[is_kgm, 'torque'] *= 9.8
   
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(str).replace('nan', np.nan)
        
    return df

def load_model():
    return joblib.load('car_price_model.pkl')

model = load_model()

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
