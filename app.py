import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Прогноз стоимости автомобилей")

# --- 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ---
CARS_TRAIN = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
CARS_TEST = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

@st.cache_data
def load_data():
    df_train = pd.read_csv(CARS_TRAIN)
    df_test = pd.read_csv(CARS_TEST)
    cols_check = df_train.drop(columns=['selling_price']).columns
    df_train = df_train.drop_duplicates(subset=cols_check, keep='first').reset_index(drop=True)
    return df_train, df_test

df_train, df_test = load_data()

def cleaner(df):
    df = df.copy()

    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace(r'[^0-9.]', '', regex=True)
                       .replace('', np.nan)
                       .astype(float))

    if 'torque' in df.columns:
        s = df['torque'].astype(str).str.lower().str.replace(',', '', regex=False)

        torque_val = s.str.extract(r'(\d+\.?\d*)')[0]
        df['torque'] = pd.to_numeric(torque_val, errors='coerce')

        df['max_torque_rpm'] = np.nan
        rpm_list = s.str.findall(r'\d+')
        df['max_torque_rpm'] = rpm_list.apply(lambda x: float(x[-1]) if isinstance(x, list) and len(x) > 0 else np.nan)

        is_kgm = s.str.contains('kgm', na=False)
        df.loc[is_kgm, 'torque'] = df.loc[is_kgm, 'torque'] * 9.8

    return df

df_train = cleaner(df_train)
df_test = cleaner(df_test)

required_median_cols = ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
for col in required_median_cols:
    if col not in df_train.columns:
        df_train[col] = np.nan
    if col not in df_test.columns:
        df_test[col] = np.nan

medians = df_train[required_median_cols].median(numeric_only=True)

for col in required_median_cols:
    df_train[col] = df_train[col].fillna(medians[col])
    df_test[col] = df_test[col].fillna(medians[col])

y_train = df_train['selling_price']
y_test = df_test['selling_price']

X_train_cat = df_train.drop('selling_price', axis=1).copy()
X_test_cat = df_test.drop('selling_price', axis=1).copy()

X_train_cat['name'] = X_train_cat['name'].astype(str).str.split().str[0]
X_test_cat['name'] = X_test_cat['name'].astype(str).str.split().str[0]

# Явно задаём категориальные и числовые столбцы
cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train = ohe.fit_transform(X_train_cat[cat_cols])

encoded_train_df = pd.DataFrame(
    encoded_train,
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_train_cat.index
)

X_train_ready = pd.concat([X_train_cat[num_cols].reset_index(drop=True), encoded_train_df.reset_index(drop=True)], axis=1)

encoded_test = ohe.transform(X_test_cat[cat_cols])
encoded_test_df = pd.DataFrame(
    encoded_test,
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_test_cat.index
)

X_test_ready = pd.concat([X_test_cat[num_cols].reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
X_train_scaled_cat = scaler.fit_transform(X_train_ready)
X_test_scaled_cat = scaler.transform(X_test_ready)

@st.cache_resource
def train_model(X, y):
    alphas = np.logspace(-2, 3, 20)
    searcher = GridSearchCV(
        Ridge(),
        param_grid={"alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    searcher.fit(X, y)
    return searcher.best_estimator_, searcher.best_params_["alpha"]

lr_ridge_best, best_alpha = train_model(X_train_scaled_cat, y_train)
st.write(f"Best alpha = {best_alpha:.4f}")

# --- 3. ИНТЕРФЕЙС STREAMLIT ---
tabs = st.tabs(["📊 EDA", "🚗 Предсказание", "🧬 Веса модели"])

with tabs[0]:
    st.header("Анализ данных (EDA)")
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots()
        sns.histplot(y_train, kde=True, ax=ax1, color='green')
        ax1.set_title("Распределение целевой переменной")
        st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots()
        sns.heatmap(df_train[num_cols + ['selling_price']].corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

with tabs[1]:
    st.header("Ввод данных владельца")
    mode = st.radio("Метод:", ["Вручную", "Загрузить CSV"])
    
    if mode == "Вручную":
        c1, c2 = st.columns(2)
        with c1:
            name = st.selectbox("Марка", sorted(df_train['name'].str.split().str[0].unique()))
            year = st.slider("Год выпуска", 1990, 2024, 2017)
            km = st.number_input("Пробег (км)", 0, 1000000, 60000)
            fuel = st.selectbox("Тип топлива", df_train['fuel'].unique())
            seats = st.selectbox("Мест", sorted(df_train['seats'].unique()))
        with c2:
            trans = st.selectbox("Коробка передач", df_train['transmission'].unique())
            sell = st.selectbox("Тип продавца", df_train['seller_type'].unique())
            owner = st.selectbox("Владелец", df_train['owner'].unique())
            power = st.number_input("Мощность (hp)", 30, 600, 100)
            
        if st.button("Узнать цену"):
            # Создаем DataFrame для предикта с использованием медиан для пропущенных полей
            input_df = pd.DataFrame([{
                'name': name, 'year': year, 'km_driven': km, 'fuel': fuel,
                'seller_type': sell, 'transmission': trans, 'owner': owner,
                'mileage': medians['mileage'], 'engine': medians['engine'],
                'max_power': power, 'torque': medians['torque'],
                'max_torque_rpm': medians['max_torque_rpm'], 'seats': seats
            }])
            
            X_input, _, _, _, _ = get_X_y(input_df, False, ohe, std_scaler)
            prediction = model.predict(X_input)[0]
            st.metric("Рекомендованная цена", f"{round(prediction):,} ₽")

    else:
        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_file:
            test_upload = pd.read_csv(uploaded_file)
            st.write("Первые строки файла:")
            st.dataframe(test_upload.head(3))
            
            # Обработка загруженного файла
            test_proc = test_upload.copy()
            # Упрощенная очистка для демо (в реальности вызвать функции выше)
            try:
                X_up, _, _, _, _ = get_X_y(test_proc, False, ohe, std_scaler)
                preds = model.predict(X_up)
                test_upload['predicted_price'] = preds
                st.write("Результаты:")
                st.dataframe(test_upload)
            except Exception as e:
                st.error(f"Ошибка в структуре файла: {e}")

with tabs[2]:
    st.header("Анализ влияния признаков")
    importance = pd.DataFrame({
        'Feature': feat_names,
        'Weight': model.coef_
    }).sort_values(by='Weight', ascending=False)
    
    # Показываем топ-10 положительных и топ-10 отрицательных
    top_bottom = pd.concat([importance.head(10), importance.tail(10)])
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_bottom, x='Weight', y='Feature', palette='RdYlGn', ax=ax3)
    ax3.set_title("Веса коэффициентов (Top 10 max/min)")
    st.pyplot(fig3)
