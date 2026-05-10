import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Прогноз стоимости автомобилей", layout="wide")

# 1. Функция очистки данных
def clean_data(df):
    df = df.copy()

    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.extract(r'(\d+\.?\d*)')[0]
                .astype(float)
            )

    if 'torque' in df.columns:
        df['torque'] = df['torque'].astype(str).str.lower()
        df['max_torque_rpm'] = df['torque'].str.findall(r'\d+').str[-1].apply(
            lambda x: float(x) if isinstance(x, str) else np.nan
        )
        df['torque'] = (
            df['torque']
            .str.extract(r'(\d+\.?\d*)')[0]
            .astype(float)
        )

    if 'name' in df.columns:
        df['name'] = df['name'].astype(str).str.split().str[0]

    return df

# 2. Загрузка модели
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "car_price_model.pkl")

    st.write("Папка app.py:", base_dir)
    st.write("Путь к модели:", model_path)
    st.write("Модель существует:", os.path.exists(model_path))

    return joblib.load(model_path)

try:
    model_pipeline = load_model()
    st.success("Модель успешно загружена")
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")
    st.stop()

st.title('🚗 Прогноз стоимости автомобилей')

# --- БЛОК 1: EDA ---
st.header('📊 Анализ данных (EDA)')
uploaded_train = st.file_uploader("Загрузите тренировочный CSV для анализа", type="csv", key='eda')

if uploaded_train:
    try:
        df_eda = pd.read_csv(uploaded_train)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Цены")
            if "selling_price" in df_eda.columns:
                fig1 = px.histogram(df_eda, x="selling_price", title="Распределение цен")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Нет колонки selling_price")

        with col2:
            st.subheader("Пробег vs Цена")
            if "km_driven" in df_eda.columns and "selling_price" in df_eda.columns:
                fig2 = px.scatter(df_eda, x="km_driven", y="selling_price", color="fuel" if "fuel" in df_eda.columns else None)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Нет нужных колонок для scatter")

        with col3:
            st.subheader("Тип топлива")
            if "fuel" in df_eda.columns:
                fig3 = px.pie(df_eda, names='fuel', title="Доля типов топлива")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Нет колонки fuel")

    except Exception as e:
        st.error(f"Ошибка при анализе CSV: {e}")

# --- БЛОК 2: Предсказание ---
st.header('🤖 Предсказание цены')

tab1, tab2 = st.tabs(["Ввод вручную", "Загрузка файла"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        name = st.selectbox('Марка', ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra'])
        year = st.slider('Год выпуска', 1990, 2022, 2015)
        km_driven = st.number_input('Пробег (км)', value=50000)
        fuel = st.selectbox('Топливо', ['Diesel', 'Petrol', 'LPG', 'CNG'])

    with col_b:
        engine = st.number_input('Объем двигателя (CC)', value=1200)
        max_power = st.number_input('Мощность (bhp)', value=80.0)
        seats = st.selectbox('Мест', [4, 5, 7, 8])
        transmission = st.radio('КПП', ['Manual', 'Automatic'])

    input_data = pd.DataFrame([[
        name, year, km_driven, fuel, engine, max_power, seats, transmission,
        'Individual', 'First Owner', '20 kmpl', '100Nm@ 2000rpm'
    ]], columns=[
        'name', 'year', 'km_driven', 'fuel', 'engine', 'max_power',
        'seats', 'transmission', 'seller_type', 'owner', 'mileage', 'torque'
    ])

    input_data = clean_data(input_data)

    if st.button('Рассчитать цену'):
        try:
            prediction = model_pipeline.predict(input_data)
            st.balloons()
            st.success(f'Предполагаемая цена: {round(float(prediction[0]), 2)} руб.')
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

with tab2:
    uploaded_file = st.file_uploader("Загрузите CSV с признаками для предсказания", type="csv", key='predict')

    if uploaded_file:
        try:
            test_df = pd.read_csv(uploaded_file)
            test_df = clean_data(test_df)

            preds = model_pipeline.predict(test_df)
            test_df['predicted_price'] = preds

            st.write(test_df.head())
            st.download_button(
                "Скачать результат",
                test_df.to_csv(index=False),
                "predictions.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"Ошибка при предсказании по файлу: {e}")

# --- БЛОК 3: Веса модели ---
st.header('⚖️ Веса модели (Feature Importance)')

if st.checkbox('Показать важность признаков'):
    try:
        if hasattr(model_pipeline, 'named_steps'):
            model = model_pipeline.named_steps.get('model', model_pipeline)
        else:
            model = model_pipeline

        if hasattr(model, 'coef_'):
            weights = model.coef_
            fig_weights, ax_weights = plt.subplots(figsize=(10, 6))
            top_weights = pd.Series(weights).sort_values(ascending=False).head(15)
            sns.barplot(x=top_weights.values, y=top_weights.index, ax=ax_weights)
            ax_weights.set_title("Важность признаков")
            st.pyplot(fig_weights)
            st.write("На графике показаны коэффициенты модели.")
        else:
            st.info("У модели нет коэффициентов для отображения.")
    except Exception as e:
        st.error(f"Ошибка при построении важности признаков: {e}")
