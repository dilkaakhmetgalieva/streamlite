import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price App", layout="wide")

st.title("Прогнозирование цены автомобиля")

@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

model = load_model()

@st.cache_data
def load_data():
    return pd.read_csv("https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv")

df = load_data()
st.header("EDA")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['selling_price'], kde=True, ax=ax)
    ax.set_title("Распределение цены автомобиля")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df['fuel'], y=df['selling_price'], ax=ax)
    ax.set_title("Цена по типу топлива")
    plt.xticks(rotation=30)
    st.pyplot(fig)
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='km_driven', y='selling_price', ax=ax)
    ax.set_title("Зависимость цены от пробега")
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x='transmission', y='selling_price', ax=ax)
    ax.set_title("Цена по типу коробки передач")
    st.pyplot(fig)
st.header("Ручной ввод признаков")

with st.form("input_form"):
    name = st.text_input("Название авто", "Hyundai Grand i10 Sportz")
    year = st.number_input("Год выпуска", min_value=1990, max_value=2025, value=2017)
    km_driven = st.number_input("Пробег", min_value=0, value=35000)
    fuel = st.selectbox("Тип топлива", df['fuel'].dropna().unique().tolist())
    seller_type = st.selectbox("Тип продавца", df['seller_type'].dropna().unique().tolist())
    transmission = st.selectbox("Трансмиссия", df['transmission'].dropna().unique().tolist())
    owner = st.selectbox("Владелец", df['owner'].dropna().unique().tolist())
    mileage = st.text_input("Mileage", "18.9 kmpl")
    engine = st.text_input("Engine", "1197 CC")
    max_power = st.text_input("Max power", "82 bhp")
    torque = st.text_input("Torque", "114Nm@ 4000rpm")
    seats = st.number_input("Seats", min_value=1, max_value=20, value=5)

    submitted = st.form_submit_button("Предсказать цену")
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

    prediction = model.predict(input_df)[0]
    st.success(f"Предсказанная цена: {prediction:,.0f}")
st.header("Загрузка CSV-файла")

uploaded_file = st.file_uploader("Загрузите CSV с признаками", type=["csv"])

if uploaded_file is not None:
    csv_df = pd.read_csv(uploaded_file)

    st.write("Первые строки загруженного файла:")
    st.dataframe(csv_df.head())

    try:
        preds = model.predict(csv_df)
        csv_df["predicted_selling_price"] = preds
        st.success("Предсказания успешно получены")
        st.dataframe(csv_df)
        st.download_button(
            label="Скачать результаты",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
st.header("Веса модели")

if st.button("Показать веса модели"):
    preprocessor = model.named_steps['preprocessor']
    ridge_model = model.named_steps['model']

    feature_names = preprocessor.get_feature_names_out()
    coefficients = ridge_model.coef_

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'weight': coefficients
    }).sort_values(by='weight', key=lambda x: np.abs(x), ascending=False)

    st.dataframe(coef_df)

    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 20
    plot_df = coef_df.head(top_n).sort_values("weight")

    ax.barh(plot_df['feature'], plot_df['weight'])
    ax.set_title(f"Топ-{top_n} весов модели")
    ax.set_xlabel("Вес")
    ax.set_ylabel("Признак")
    st.pyplot(fig)
