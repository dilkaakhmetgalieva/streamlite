import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# --- КОПИРУЕМ КЛАСС CLEANER (обязательно для загрузки pickle) ---
class Cleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        cols_to_clean = ['mileage', 'engine', 'max_power']
        for col in cols_to_clean:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True).replace('', np.nan).astype(float)
        if 'torque' in df.columns and df['torque'].dtype == object:
            s = df['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
            is_kgm = s.str.contains('kgm', na=False)
            df['torque'] = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
            df.loc[is_kgm, 'torque'] *= 9.8
            df['max_torque_rpm'] = s.str.findall(r'\d+').str[-1].astype(float)
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str).str.split().str[0]
        return df

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    with open('car_price_model.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_model()

# --- ЗАГОЛОВОК ---
st.title("🚗 Прогноз стоимости автомобилей")

# --- ВКЛАДКИ ДЛЯ РАЗДЕЛОВ ---
tab1, tab2, tab3 = st.tabs(["📊 Аналитика (EDA)", "🔮 Предсказание", "📈 Веса модели"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Анализ данных")
    # Для демонстрации графиков загружаем тренировочный датасет, если он есть, 
    # или просто показываем примеры на основе логики
    uploaded_data = st.file_uploader("Загрузите CSV для анализа", type="csv", key="eda_upload")
    if uploaded_data:
        df_eda = pd.read_csv(uploaded_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Распределение цен")
            fig, ax = plt.subplots()
            sns.histplot(df_eda['selling_price'], kde=True, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("Тип топлива")
            fig, ax = plt.subplots()
            df_eda['fuel'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)
            
        st.write("Зависимость цены от пробега")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(data=df_eda, x='km_driven', y='selling_price', alpha=0.5)
        st.pyplot(fig)

# --- TAB 2: ПРЕДСКАЗАНИЕ ---
with tab2:
    st.header("Ввод данных")
    
    input_mode = st.radio("Как ввести данные?", ["Вручную", "Загрузить файл CSV"])
    
    if input_mode == "Вручную":
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.number_input("Год выпуска", 1990, 2024, 2015)
            km_driven = st.number_input("Пробег (км)", 0, 1000000, 50000)
        with col2:
            fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG"])
            seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
        with col3:
            transmission = st.selectbox("КПП", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner"])

        # Обязательно добавляем те столбцы, которые ожидает Cleaner и Preprocessor
        input_df = pd.DataFrame([{
            'name': 'Maruti', # дефолтное значение для корректной работы Cleaner
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': '20.0 kmpl',
            'engine': '1200 CC',
            'max_power': '70 bhp',
            'torque': '100 Nm@ 2000rpm',
            'seats': 5.0
        }])
        
        if st.button("Рассчитать стоимость"):
            res = pipeline.predict(input_df)
            st.success(f"Оценочная стоимость: {round(res[0], 2)} руб.")

    else:
        file = st.file_uploader("Загрузите CSV с признаками", type="csv")
        if file:
            test_df = pd.read_csv(file)
            preds = pipeline.predict(test_df)
            test_df['predicted_price'] = preds
            st.write(test_df)
            st.download_button("Скачать результат", test_df.to_csv(index=False), "preds.csv")

# --- TAB 3: ВЕСА МОДЕЛИ ---
with tab3:
    st.header("Интерпретация модели")
    
    # Извлекаем названия признаков после OneHotEncoder
    try:
        # Получаем имена колонок после трансформации
        cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out()
        num_features = pipeline.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out()
        all_features = np.concatenate([num_features, cat_features])
        
        # Получаем веса Ridge
        weights = pipeline.named_steps['model'].coef_
        
        # Создаем таблицу весов
        coef_df = pd.DataFrame({'Feature': all_features, 'Weight': weights})
        coef_df = coef_df.sort_values(by='Weight', ascending=False)
        
        st.write("Топ влияющих признаков:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=coef_df.head(10).append(coef_df.tail(10)), x='Weight', y='Feature', ax=ax)
        st.pyplot(fig)
        
        st.dataframe(coef_df)
    except:
        st.error("Не удалось визуализировать веса. Убедитесь, что модель обучена корректно.")
