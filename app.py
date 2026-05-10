import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import random

# Настройка страницы
st.set_page_config(page_title="Авто-Предиктор", layout="wide")
st.title("🚗 Прогноз стоимости автомобилей")

# Константы
TRAIN_URL = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
TEST_URL = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'
RANDOM_STATE = 42

# --- 1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ ---
@st.cache_data
def load_and_clean():
    df_train = pd.read_csv(TRAIN_URL)
    df_test = pd.read_csv(TEST_URL)
    
    # Удаление дубликатов (кроме таргета)
    cols_check = [c for c in df_train.columns if c != 'selling_price']
    df_train = df_train.drop_duplicates(subset=cols_check).reset_index(drop=True)

    def process_strings(df):
        temp = df.copy()
        # Извлекаем числа из строк
        for col in ['mileage', 'engine', 'max_power']:
            if col in temp.columns:
                temp[col] = temp[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        
        # Обработка torque
        if 'torque' in temp.columns:
            # Важно: работаем только с непустыми строками
            s = temp['torque'].astype(str).str.lower()
            
            # Извлекаем само значение момента
            torque_val = s.str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
            
            # Конвертация kgm в Nm
            is_kgm = s.str.contains('kgm', na=False)
            torque_val[is_kgm] = torque_val[is_kgm] * 9.8
            
            # Извлечение RPM (безопасный способ)
            # findall вернет пустой список, если ничего не найдено
            list_of_numbers = s.str.findall(r'\d+')
            
            def get_last_num(x):
                if isinstance(x, list) and len(x) > 0:
                    try:
                        return float(x[-1])
                    except:
                        return np.nan
                return np.nan

            rpm = list_of_numbers.apply(get_last_num)
            
            temp['torque'] = torque_val
            temp['max_torque_rpm'] = rpm
            
        return temp

    df_train = process_strings(df_train)
    df_test = process_strings(df_test)
    
    # Заполнение пропусков медианой
    num_features = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
    medians = df_train[num_features].median()
    df_train[num_features] = df_train[num_features].fillna(medians)
    df_test[num_features] = df_test[num_features].fillna(medians)
    
    return df_train, df_test, medians

df_train, df_test, medians = load_and_clean()

# --- 2. ПОДГОТОВКА ПРИЗНАКОВ ---
cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']

def get_X_y(df, is_train=True, encoder=None, scaler=None):
    df = df.copy()
    # Берем только марку авто
    df['name'] = df['name'].str.split().str[0]
    
    y = df['selling_price'] if 'selling_price' in df.columns else None
    
    if is_train:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_enc = encoder.fit_transform(df[cat_cols])
        
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(df[num_cols])
    else:
        cat_enc = encoder.transform(df[cat_cols])
        num_scaled = scaler.transform(df[num_cols])
        
    X = np.hstack([num_scaled, cat_enc])
    feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))
    
    return X, y, encoder, scaler, feature_names

X_train, y_train, ohe, std_scaler, feat_names = get_X_y(df_train)

# --- 3. ОБУЧЕНИЕ МОДЕЛИ ---
@st.cache_resource
def train_ridge(X, y):
    params = {'alpha': np.logspace(-2, 3, 10)}
    grid = GridSearchCV(Ridge(), params, cv=5, scoring='neg_root_mean_squared_error')
    grid.fit(X, y)
    return grid.best_estimator_

model = train_ridge(X_train, y_train)

# --- 4. ИНТЕРФЕЙС STREAMLIT ---
tabs = st.tabs(["📊 Анализ данных (EDA)", "🔮 Предсказание", "🧮 Веса модели"])

with tabs[0]:
    st.header("Анализ тренировочного набора")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(y_train, kde=True, color='green', ax=ax1)
        ax1.set_title("Распределение цены")
        st.pyplot(fig1)
        
    with col2:
        fig2, ax2 = plt.subplots()
        corr_matrix = df_train[num_cols + ['selling_price']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        ax2.set_title("Корреляция признаков")
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
