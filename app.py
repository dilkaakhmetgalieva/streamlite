import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from custom_transformers import CarDataCleaner  # Импорт нашего класса

st.set_page_config(page_title="Car Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    # Файл должен лежать в корне вместе с app.py
    with open("car_model_pipeline.pickle", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")
    st.stop()

st.title("🚗 Предсказание стоимости автомобиля")

tab1, tab2 = st.tabs(["Ввод данных вручную", "Загрузка файла"])

input_df = pd.DataFrame()

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Марка и модель", "Maruti Swift Dzire VDI")
        year = st.slider("Год выпуска", 1990, 2024, 2014)
        km_driven = st.number_input("Пробег (км)", 0, 1000000, 50000)
        fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "LPG", "CNG"])
        seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
    
    with col2:
        transmission = st.selectbox("КПП", ["Manual", "Automatic"])
        owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
        mileage = st.text_input("Расход (mileage)", "23.4 kmpl")
        engine = st.text_input("Двигатель (engine)", "1248 CC")
        max_power = st.text_input("Мощность (max_power)", "74 bhp")
        torque = st.text_input("Крутящий момент (torque)", "190Nm@ 2000rpm")
        seats = st.number_input("Мест", 2, 10, 5)

    if st.button("Рассчитать стоимость"):
        input_df = pd.DataFrame([{
            "name": name, "year": year, "km_driven": km_driven, "fuel": fuel,
            "seller_type": seller_type, "transmission": transmission,
            "owner": owner, "mileage": mileage, "engine": engine,
            "max_power": max_power, "torque": torque, "seats": seats
        }])

with tab2:
    uploaded_test = st.file_uploader("Загрузите CSV файл", type="csv")
    if uploaded_test is not None:
        input_df = pd.read_csv(uploaded_test)

if not input_df.empty:
    try:
        preds = model.predict(input_df)
        st.success("Результат готов!")
        if len(preds) == 1:
            st.metric("Прогноз цены", f"{round(preds[0], 2)} руб.")
        else:
            res_df = input_df.copy()
            res_df["predicted_price"] = preds
            st.dataframe(res_df)
    except Exception as e:
        st.error(f"Ошибка при работе модели: {e}")

st.divider()
st.header("📊 Интерпретация модели")
if st.button("Показать веса признаков"):
    try:
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['model']
        
        features = preprocessor.get_feature_names_out()
        coefs = regressor.coef_
        
        weights_df = pd.DataFrame({'Feature': features, 'Weight': coefs}).sort_values(by='Weight', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_weights = pd.concat([weights_df.head(10), weights_df.tail(10)])
        sns.barplot(data=top_weights, x='Weight', y='Feature', palette='coolwarm', ax=ax)
        st.pyplot(fig)
    except:
        st.warning("Веса доступны только для линейных моделей с именованным шагом 'preprocessor' и 'model'")
