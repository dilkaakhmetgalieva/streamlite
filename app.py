import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from custom_transformers import CarDataCleaner

st.set_page_config(page_title="Car Price Predictor", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("car_price_pipeline.pkl")

@st.cache_data
def load_data():
    return joblib.load("df_train.pkl")

pipeline = load_pipeline()
df = load_data()

st.title("Предсказание стоимости автомобиля")

tab1, tab2, tab3, tab4 = st.tabs([
    "Ручной ввод",
    "Загрузка CSV",
    "EDA",
    "Веса признаков"
])

with tab1:
    st.header("Ручной ввод параметров")

    name = st.selectbox("Марка/модель", sorted(df["name"].dropna().unique()))
    year = st.number_input("Год выпуска", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("Пробег", min_value=0, max_value=1000000, value=50000)
    fuel = st.selectbox("Тип топлива", sorted(df["fuel"].dropna().unique()))
    seller_type = st.selectbox("Тип продавца", sorted(df["seller_type"].dropna().unique()))
    transmission = st.selectbox("Коробка передач", sorted(df["transmission"].dropna().unique()))
    owner = st.selectbox("Владельцы", sorted(df["owner"].dropna().unique()))
    mileage = st.text_input("Пробег (например 18.9 kmpl)", "18.9 kmpl")
    engine = st.text_input("Объём двигателя (например 1197 CC)", "1197 CC")
    max_power = st.text_input("Мощность (например 82 bhp)", "82 bhp")
    torque = st.text_input("Крутящий момент (например 113Nm@4200rpm)", "113Nm@4200rpm")
    seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5)

    if st.button("Предсказать цену"):
        input_df = pd.DataFrame([{
            "name": name,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "torque": torque,
            "seats": seats
        }])

        prediction = pipeline.predict(input_df)[0]
        st.success(f"Оценочная стоимость: {prediction:,.2f}")

with tab2:
    st.header("Пакетное предсказание из CSV")

    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.subheader("Предпросмотр данных")
        st.dataframe(batch_df.head())

        if st.button("Сделать предсказание по CSV"):
            preds = pipeline.predict(batch_df)
            result_df = batch_df.copy()
            result_df["predicted_price"] = preds

            st.subheader("Результаты")
            st.dataframe(result_df.head())

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать результаты CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

with tab3:
    st.header("EDA")

    st.subheader("Первые строки датасета")
    st.dataframe(df.head())

    st.subheader("Размер датасета")
    st.write(df.shape)

    st.subheader("Типы данных")
    st.write(df.dtypes)

    st.subheader("Пропуски")
    st.write(df.isna().sum())

    st.subheader("Распределение цены")
    fig, ax = plt.subplots()
    sns.histplot(df["selling_price"], kde=True, ax=ax)
    ax.set_title("Распределение selling_price")
    st.pyplot(fig)

    st.subheader("Корреляция числовых признаков")
    numeric_df = df.select_dtypes(include=["number"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Boxplot цены")
    fig, ax = plt.subplots()
    sns.boxplot(x=df["selling_price"], ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("Веса признаков модели")

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]

        feature_names = preprocessor.get_feature_names_out()
        coefficients = regressor.coef_

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients
        }).sort_values("coefficient", key=abs, ascending=False)

        st.subheader("Топ-20 признаков по абсолютной величине веса")
        st.dataframe(coef_df.head(20))

        fig, ax = plt.subplots(figsize=(10, 8))
        top_coef = coef_df.head(20).iloc[::-1]
        ax.barh(top_coef["feature"], top_coef["coefficient"])
        ax.set_title("Топ-20 весов признаков")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Не удалось получить веса признаков: {e}")
