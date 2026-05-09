import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

st.set_page_config(page_title="Прогноз стоимости автомобилей", layout="wide")

# =========================
# 1. Классы и загрузка
# =========================
class CarDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        cols_to_clean = ['mileage', 'engine', 'max_power']
        for col in cols_to_clean:
            if col in X.columns and X[col].dtype == object:
                X[col] = (
                    X[col].astype(str)
                    .str.replace(r'[^0-9.]', '', regex=True)
                    .replace('', np.nan)
                    .replace('nan', np.nan)
                    .astype(float)
                )

        if 'torque' in X.columns and X['torque'].dtype == object:
            s = X['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
            is_kgm = s.str.contains('kgm', na=False)
            extracted_torque = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
            X['max_torque_rpm'] = s.str.findall(r'\d+').str[-1].astype(float)
            X['torque'] = extracted_torque
            X.loc[is_kgm, 'torque'] *= 9.8
            X.loc[s.eq('nan'), ['torque', 'max_torque_rpm']] = np.nan

        if 'name' in X.columns:
            X['name'] = X['name'].astype(str).str.split().str[0]

        return X


@st.cache_resource
def load_pipeline():
    return joblib.load("car_price_pipeline.pkl")


@st.cache_data
def load_train_data():
    return joblib.load("df_train.pkl")


pipeline = load_pipeline()
df_train = load_train_data()

# =========================
# 2. Заголовок
# =========================
st.title("Приложение для прогнозирования стоимости автомобилей")
st.write("Приложение показывает EDA, принимает данные автомобиля и предсказывает его стоимость с помощью обученной линейной модели.")

# =========================
# 3. EDA
# =========================
st.header("EDA: исследовательский анализ данных")

tab1, tab2, tab3 = st.tabs(["Распределения", "Корреляция", "Категориальные признаки"])

with tab1:
    st.subheader("Распределение цены")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_train['selling_price'], kde=True, ax=ax)
    ax.set_title("Распределение selling_price")
    st.pyplot(fig)

    st.subheader("Распределение пробега")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_train['km_driven'], kde=True, ax=ax)
    ax.set_title("Распределение km_driven")
    st.pyplot(fig)

    st.subheader("Распределение года выпуска")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_train['year'], kde=False, bins=20, ax=ax)
    ax.set_title("Распределение year")
    st.pyplot(fig)

with tab2:
    st.subheader("Тепловая карта корреляций числовых признаков")

    df_corr = df_train.copy()

    cleaner = CarDataCleaner()
    df_corr = cleaner.transform(df_corr)

    numeric_cols = df_corr.select_dtypes(include=['number']).columns
    corr = df_corr[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Корреляция числовых признаков")
    st.pyplot(fig)

with tab3:
    st.subheader("Средняя цена по типу топлива")
    fig, ax = plt.subplots(figsize=(8, 5))
    fuel_price = df_train.groupby('fuel')['selling_price'].mean().sort_values(ascending=False)
    sns.barplot(x=fuel_price.index, y=fuel_price.values, ax=ax)
    ax.set_title("Средняя цена по fuel")
    ax.set_xlabel("fuel")
    ax.set_ylabel("Средняя цена")
    st.pyplot(fig)

    st.subheader("Средняя цена по типу коробки передач")
    fig, ax = plt.subplots(figsize=(8, 5))
    tr_price = df_train.groupby('transmission')['selling_price'].mean().sort_values(ascending=False)
    sns.barplot(x=tr_price.index, y=tr_price.values, ax=ax)
    ax.set_title("Средняя цена по transmission")
    ax.set_xlabel("transmission")
    ax.set_ylabel("Средняя цена")
    st.pyplot(fig)

# =========================
# 4. Предсказание
# =========================
st.header("Предсказание стоимости автомобиля")

input_mode = st.radio(
    "Выберите способ ввода данных:",
    ["Ручной ввод", "Загрузка CSV"]
)

feature_columns = [
    'name', 'year', 'km_driven', 'fuel', 'seller_type',
    'transmission', 'owner', 'mileage', 'engine',
    'max_power', 'torque', 'seats'
]

if input_mode == "Ручной ввод":
    st.subheader("Введите признаки автомобиля")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Марка/модель", value="Maruti Swift")
        year = st.number_input("Год выпуска", min_value=1980, max_value=2030, value=2015)
        km_driven = st.number_input("Пробег (km_driven)", min_value=0, value=50000)
        fuel = st.selectbox("Тип топлива", options=sorted(df_train['fuel'].dropna().unique()))
        seller_type = st.selectbox("Тип продавца", options=sorted(df_train['seller_type'].dropna().unique()))
        transmission = st.selectbox("Коробка передач", options=sorted(df_train['transmission'].dropna().unique()))

    with col2:
        owner = st.selectbox("Тип владельца", options=sorted(df_train['owner'].dropna().unique()))
        mileage = st.text_input("Mileage", value="20.4 kmpl")
        engine = st.text_input("Engine", value="1197 CC")
        max_power = st.text_input("Max power", value="81.80 bhp")
        torque = st.text_input("Torque", value="113Nm@4200rpm")
        seats = st.number_input("Количество мест", min_value=2, max_value=20, value=5)

    if st.button("Предсказать цену"):
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

        pred = pipeline.predict(input_df)[0]
        st.success(f"Предсказанная стоимость: {pred:,.2f}")

else:
    st.subheader("Загрузите CSV-файл")

    uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Первые строки загруженного файла:")
        st.dataframe(input_df.head())

        missing_cols = [col for col in feature_columns if col not in input_df.columns]

        if missing_cols:
            st.error(f"В файле отсутствуют столбцы: {missing_cols}")
        else:
            preds = pipeline.predict(input_df[feature_columns])
            result_df = input_df.copy()
            result_df['predicted_price'] = preds

            st.write("Результат предсказания:")
            st.dataframe(result_df)

            csv_result = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать результат CSV",
                data=csv_result,
                file_name="predictions.csv",
                mime="text/csv"
            )

# =========================
# 5. Визуализация весов модели
# =========================
st.header("Визуализация весов обученной модели")

if hasattr(pipeline.named_steps['regressor'], 'coef_'):
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']

    feature_names = preprocessor.get_feature_names_out()
    coefs = regressor.coef_

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'weight': coefs
    })

    coef_df['abs_weight'] = coef_df['weight'].abs()
    coef_df = coef_df.sort_values('abs_weight', ascending=False)

    st.subheader("Топ-20 самых значимых признаков")
    top_coef_df = coef_df.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_coef_df, y='feature', x='weight', ax=ax)
    ax.set_title("Топ-20 весов линейной модели")
    st.pyplot(fig)

    st.subheader("Таблица весов")
    st.dataframe(coef_df[['feature', 'weight']])
else:
    st.warning("У текущей модели нет атрибута coef_ для визуализации весов.")
