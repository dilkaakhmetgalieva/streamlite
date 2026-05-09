import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Предсказание стоимости автомобиля", layout="wide")

@st.cache_resource
def load_model():
    with open("car_model_pipeline.pickle", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Файл 'car_model_pipeline.pickle' не найден. Пожалуйста, загрузите его в директорию с приложением.")
    st.stop()

st.title("🚗 Приложение для анализа и предсказания стоимости авто")


st.header("1. Исследовательский анализ данных (EDA)")
uploaded_train = st.file_uploader("Загрузите обучающий CSV-файл для анализа", type="csv")

if uploaded_train is not None:
    df_train = pd.read_csv(uploaded_train)
    

    st.info(f"Количество строк до удаления дубликатов: {len(df_train)}")
    df_train = df_train.drop_duplicates()
    st.success(f"Количество строк после удаления дубликатов: {len(df_train)}")
    
    st.subheader("Обзор данных (первые 5 строк)")
    st.dataframe(df_train.head())


    numeric_train = df_train.select_dtypes(include='number')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pairplot числовых признаков")
        fig_pair = sns.pairplot(numeric_train, diag_kind='kde')
        st.pyplot(fig_pair.figure)

    with col2:
        st.subheader("Тепловая карта корреляции")

        pearsoner_corr = numeric_train.corr()
        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
        sns.heatmap(pearsoner_corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_heat)
        plt.title('Тепловая карта числовых признаков')
        st.pyplot(fig_heat)

st.header("2. Предсказание стоимости")
input_method = st.radio("Способ ввода данных:", ["Вручную", "Загрузить CSV файл"])

input_df = pd.DataFrame()

if input_method == "Вручную":
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        name = st.text_input("Марка и модель", "Maruti Swift SZ5")
        year = st.number_input("Год выпуска", 1990, 2024, 2018)
        km_driven = st.number_input("Пробег (км)", 0, 1000000, 50000)
    with col_b:
        fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG"])
        seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("КПП", ["Manual", "Automatic"])
    with col_c:
        owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        mileage = st.text_input("Расход (mileage)", "20.0 kmpl")
        engine = st.text_input("Двигатель (engine)", "1200 CC")
    
    max_power = st.text_input("Мощность (max_power)", "75 bhp")
    torque = st.text_input("Крутящий момент (torque)", "190Nm@ 2000rpm")
    seats = st.number_input("Места", 2, 10, 5)

    if st.button("Рассчитать стоимость"):
        input_df = pd.DataFrame([{
            "name": name, "year": year, "km_driven": km_driven, "fuel": fuel,
            "seller_type": seller_type, "transmission": transmission,
            "owner": owner, "mileage": mileage, "engine": engine,
            "max_power": max_power, "torque": torque, "seats": seats
        }])

else:
    uploaded_test = st.file_uploader("Загрузите CSV с признаками для предсказания", type="csv")
    if uploaded_test is not None:
        input_df = pd.read_csv(uploaded_test)

if not input_df.empty:
    try:
        preds = model.predict(input_df)
        st.subheader("Результаты предсказания:")
        if len(preds) == 1:
            st.metric("Прогноз цены", f"{round(preds[0], 2)}")
        else:
            res_df = input_df.copy()
            res_df["predicted_price"] = preds
            st.dataframe(res_df)
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")

st.header("3. Визуализация весов модели")
if st.button("Показать влияние признаков"):
    try:
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['model']
        
        features = preprocessor.get_feature_names_out()
        coefs = regressor.coef_
        
        weights_df = pd.DataFrame({'Feature': features, 'Weight': coefs})
        weights_df = weights_df.sort_values(by='Weight', ascending=False)
        
        fig_weights, ax_weights = plt.subplots(figsize=(10, 12))
        top_weights = pd.concat([weights_df.head(10), weights_df.tail(10)])
        sns.barplot(data=top_weights, x='Weight', y='Feature', palette='vlag', ax=ax_weights)
        ax_weights.set_title("Самые важные признаки (Топ положительных и отрицательных)")
        st.pyplot(fig_weights)
        
        st.write("Таблица всех весов:", weights_df)
    except Exception as e:
        st.warning(f"Не удалось извлечь веса (проверьте структуру пайплайна): {e}")
