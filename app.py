import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Функция очистки данных (дублируем логику из обучения)
def clean_data(df):
    df = df.copy()
    # Обработка числовых колонок из строк (mileage, engine, max_power)
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Обработка torque
    if 'torque' in df.columns:
        df['torque'] = df['torque'].astype(str).str.lower()
        df['max_torque_rpm'] = df['torque'].str.findall(r'\d+').str[-1].apply(lambda x: float(x) if isinstance(x, str) else np.nan)
        # Упрощенная логика для примера
        df['torque'] = df['torque'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Извлечение марки авто
    if 'name' in df.columns:
        df['name'] = df['name'].str.split().str[0]
        
    return df

# Загрузка модели
@st.cache_resource
def load_model():
    # Убедитесь, что файл car_price_model.pkl лежит в той же папке
    return joblib.load('car_price_model.pkl')

try:
    model_pipeline = load_model()
    st.success("Модель успешно загружена")
except:
    st.error("Файл car_price_model.pkl не найден. Пожалуйста, обучите модель и сохраните её.")

st.title('🚗 Прогноз стоимости автомобилей')

# --- БЛОК 1: EDA (3 балла) ---
st.header('📊 Анализ данных (EDA)')
uploaded_train = st.file_uploader("Загрузите тренировочный CSV для анализа", type="csv", key='eda')

if uploaded_train:
    df_eda = pd.read_csv(uploaded_train)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Цены")
        fig1 = px.histogram(df_eda, x="selling_price", title="Распределение цен")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Пробег vs Цена")
        fig2 = px.scatter(df_eda, x="km_driven", y="selling_price", color="fuel")
        st.plotly_chart(fig2, use_container_width=True)
        
    with col3:
        st.subheader("Тип топлива")
        fig3 = px.pie(df_eda, names='fuel', title="Доля типов топлива")
        st.plotly_chart(fig3, use_container_width=True)

# --- БЛОК 2: Предсказание (3 балла) ---
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

    input_data = pd.DataFrame([[name, year, km_driven, fuel, engine, max_power, seats, transmission, 'Individual', 'First Owner', '20 kmpl', '100Nm@ 2000rpm']], 
                              columns=['name', 'year', 'km_driven', 'fuel', 'engine', 'max_power', 'seats', 'transmission', 'seller_type', 'owner', 'mileage', 'torque'])

    if st.button('Рассчитать цену'):
        prediction = model_pipeline.predict(input_data)
        st.balloons()
        st.success(f'Предполагаемая цена: {round(prediction[0], 2)} руб.')

with tab2:
    uploaded_file = st.file_uploader("Загрузите CSV с признаками для предсказания", type="csv", key='predict')
    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)
        preds = model_pipeline.predict(test_df)
        test_df['predicted_price'] = preds
        st.write(test_df.head())
        st.download_button("Скачать результат", test_df.to_csv(index=False), "predictions.csv")

# --- БЛОК 3: Веса модели (3 балла) ---
st.header('⚖️ Веса модели (Feature Importance)')
if st.checkbox('Показать важность признаков'):
    try:
        # Пытаемся достать веса из Ridge/Lasso внутри пайплайна
        model = model_pipeline.named_steps['model']
        # Получаем имена колонок после OneHotEncoding
        # Это может варьироваться от версии sklearn, но логика такая:
        if hasattr(model, 'coef_'):
            # Если это Ridge/LinearRegression
            weights = model.coef_
            # Для упрощения визуализируем топ-10 самых важных
            fig_weights, ax_weights = plt.subplots(figsize=(10, 6))
            # Берем абсолютные значения для важности
            top_weights = pd.Series(weights).sort_values(ascending=False).head(15)
            sns.barplot(x=top_weights.values, y=top_weights.index, ax=ax_weights)
            st.pyplot(fig_weights)
            st.write("На графике отображены коэффициенты модели для наиболее значимых признаков.")
    except Exception as e:
        st.info("Для визуализации весов убедитесь, что в пайплайне используется линейная модель.")
