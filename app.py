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

# Настройка random
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

@st.cache_data
def load_data():
    df_train = pd.read_csv(CARS_TRAIN)
    df_test = pd.read_csv(CARS_TEST)
    # Удаление дублей
    cols_check = df_train.drop(columns=['selling_price']).columns
    df_train = df_train.drop_duplicates(subset=cols_check, keep='first').reset_index(drop=True)
    return df_train, df_test

df_train, df_test = load_data()

def cleaner(df):
    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace(r'[^0-9.]', '', regex=True)
                       .replace('', np.nan).astype(float))

    # Обработка torque
    if 'torque' in df.columns and df['torque'].dtype == object:
        s = df['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
        is_kgm = s.str.contains('kgm', na=False)
        extracted_torque = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
        df['max_torque_rpm'] = s.str.findall(r'\d+').str[-1].astype(float)
        df['torque'] = extracted_torque
        df.loc[is_kgm, 'torque'] *= 9.8
        df.loc[s.eq('nan'), ['torque', 'max_torque_rpm']] = np.nan
    return df

# Применяем очистку
df_train = cleaner(df_train)
df_test = cleaner(df_test)

# Заполнение пропусков медианой
df_train['mileage'] = df_train['mileage'].fillna(df_train['mileage'].median())
df_train['engine'] = df_train['engine'].fillna(df_train['engine'].median())
df_train['max_power'] = df_train['max_power'].fillna(df_train['max_power'].median())
df_train['torque'] = df_train['torque'].fillna(df_train['torque'].median())
df_train['seats'] = df_train['seats'].fillna(df_train['seats'].median())
df_train['max_torque_rpm'] = df_train['max_torque_rpm'].fillna(df_train['max_torque_rpm'].median())

df_test['mileage'] = df_test['mileage'].fillna(df_train['mileage'].median())
df_test['engine'] = df_test['engine'].fillna(df_train['engine'].median())
df_test['max_power'] = df_test['max_power'].fillna(df_train['max_power'].median())
df_test['torque'] = df_test['torque'].fillna(df_train['torque'].median())
df_test['seats'] = df_test['seats'].fillna(df_train['seats'].median())
df_test['max_torque_rpm'] = df_test['max_torque_rpm'].fillna(df_train['max_torque_rpm'].median())

# Выделение таргета
y_train = df_train['selling_price']
y_test = df_test['selling_price']

df_train['engine'] = df_train['engine'].astype(int)
df_train['seats'] = df_train['seats'].astype(int)

# Подготовка признаков (Берем только марку из названия)
X_train_cat = df_train.drop('selling_price', axis=1).copy()
X_train_cat['name'] = X_train_cat['name'].str.split().str[0]

X_test_cat = df_test.drop('selling_price', axis=1).copy()
X_test_cat['name'] = X_test_cat['name'].str.split().str[0]

from sklearn.preprocessing import OneHotEncoder

cat_cols = X_train_cat.select_dtypes(include='object').columns.tolist()
if 'seats' not in cat_cols:
    cat_cols.append('seats')
num_cols = [col for col in X_train_cat.columns if col not in cat_cols]

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train = ohe.fit_transform(X_train_cat[cat_cols])
encoded_train_df = pd.DataFrame(
    encoded_train,
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_train_cat.index
)
X_train_ready = pd.concat([X_train_cat[num_cols], encoded_train_df], axis=1)

encoded_test = ohe.transform(X_test_cat[cat_cols])
encoded_test_df = pd.DataFrame(
    encoded_test,
    columns=ohe.get_feature_names_out(cat_cols),
    index=X_test_cat.index
)
X_test_ready = pd.concat([X_test_cat[num_cols], encoded_test_df], axis=1)

numeric_train_cat = X_train_ready.select_dtypes(include='number')
numeric_test_cat = X_test_ready.select_dtypes(include='number')

# Масштабирование
scaler = StandardScaler()

X_train_scaled_cat = scaler.fit_transform(numeric_train_cat)
X_test_scaled_cat = scaler.transform(numeric_test_cat)

# --- 2. ОБУЧЕНИЕ МОДЕЛИ ---
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

@st.cache_resource
def train_model(X, y):
    alphas = np.logspace(-2, 3, 20)
    searcher = GridSearchCV(Ridge(), [{
        "alpha": alphas
    }],
                        scoring="neg_root_mean_squared_error",
                        cv=10)
searcher.fit(X_train_scaled_cat, y_train)

best_alpha = searcher.best_params_["alpha"]
print("Best alpha = %.4f" % best_alpha)

lr_ridge_best = Ridge(alpha=best alpha)
lr_ridge_best.fit(X_train_scaled_cat, y_train)

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
    mode = st.radio("Способ ввода:", ("Вручную", "Загрузить CSV"))
    if mode == "Вручную":
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            name_input = st.selectbox("Марка", sorted(X_train_cat['name'].unique()))
            year_input = st.slider("Год", 1990, 2024, 2018)
            km_input = st.number_input("Пробег", 0, 1000000, 50000)
            fuel_input = st.selectbox("Топливо", df_train['fuel'].unique())
        with col_in2:
            trans_input = st.selectbox("КПП", df_train['transmission'].unique())
            sell_input = st.selectbox("Продавец", df_train['seller_type'].unique())
            owner_input = st.selectbox("Владелец", df_train['owner'].unique())
            power_input = st.number_input("Мощность", 30, 600, 100)
        
        if st.button("Рассчитать стоимость"):
            # Создаем строку с теми же признаками, что и в обучении
            input_row = pd.DataFrame([{
                'name': name_input, 'year': year_input, 'km_driven': km_input,
                'fuel': fuel_input, 'seller_type': sell_input, 'transmission': trans_input,
                'owner': owner_input, 'mileage': medians['mileage'], 'engine': medians['engine'],
                'max_power': power_input, 'torque': medians['torque'], 
                'max_torque_rpm': medians['max_torque_rpm'], 'seats': 5
            }])
            
            # Кодируем
            cat_part = pd.DataFrame(ohe.transform(input_row[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
            num_part = input_row[num_cols].reset_index(drop=True)
            ready_row = pd.concat([num_part, cat_part], axis=1)
            scaled_row = scaler.transform(ready_row)
            
            res = lr_ridge_best.predict(scaled_row)[0]
            st.success(f"Прогнозная цена: {round(res, 2)} ₽")

with tabs[2]:
    st.header("Влияние признаков")
    coefs = pd.DataFrame({'Feature': X_train_ready.columns, 'Weight': lr_ridge_best.coef_})
    coefs = coefs.sort_values(by='Weight', ascending=False)
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.barplot(data=pd.concat([coefs.head(10), coefs.tail(10)]), x='Weight', y='Feature', palette='viridis')
    st.pyplot(fig3)
