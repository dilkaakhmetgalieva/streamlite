import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Настройка страницы
st.set_page_config(page_title="Авто-Предиктор", layout="wide")
st.title("🚗 Прогноз стоимости автомобилей")

# Константы
TRAIN_URL = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
TEST_URL = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'

# --- 1. ФУНКЦИИ ОЧИСТКИ ---

def process_strings(df):
    temp = df.copy()
    # Чистим числовые колонки
    for col in ['mileage', 'engine', 'max_power']:
        if col in temp.columns:
            temp[col] = temp[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Обработка torque
    if 'torque' in temp.columns:
        s = temp['torque'].astype(str).str.lower()
        # Значение момента
        torque_val = s.str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
        # Конвертация kgm в Nm
        is_kgm = s.str.contains('kgm', na=False)
        torque_val[is_kgm] = torque_val[is_kgm] * 9.8
        
        # RPM (берем последнее число в строке - обычно это и есть max_torque_rpm)
        list_of_numbers = s.str.findall(r'\d+')
        def get_last_num(x):
            if isinstance(x, list) and len(x) > 0:
                try: return float(x[-1].replace(',', ''))
                except: return np.nan
            return np.nan
        
        rpm = list_of_numbers.apply(get_last_num)
        temp['torque'] = torque_val
        temp['max_torque_rpm'] = rpm
    return temp

@st.cache_data
def load_and_clean():
    df_train = pd.read_csv(TRAIN_URL)
    df_test = pd.read_csv(TEST_URL)
    
    # Удаление дубликатов
    cols_check = [c for c in df_train.columns if c != 'selling_price']
    df_train = df_train.drop_duplicates(subset=cols_check).reset_index(drop=True)

    # Очистка строк
    df_train = process_strings(df_train)
    df_test = process_strings(df_test)
    
    # Считаем медианы
    num_features = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
    medians = df_train[num_features].median()
    
    # Заполняем пропуски
    df_train[num_features] = df_train[num_features].fillna(medians)
    df_test[num_features] = df_test[num_features].fillna(medians)
    
    return df_train, df_test, medians

df_train, df_test, medians = load_and_clean()

# --- 2. ПОДГОТОВКА ПРИЗНАКОВ ---

cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']

def get_X_y(df, is_train=True, encoder=None, scaler=None):
    df = df.copy()
    # Приводим названия марок к единому виду
    df['name'] = df['name'].str.split().str[0]
    
    # Если каких-то колонок нет (например, torque_rpm), создаем их
    for col in num_cols:
        if col not in df.columns:
            df[col] = medians.get(col, 0)
    
    y = df['selling_price'] if 'selling_price' in df.columns else None
    
    if is_train:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df[cat_cols])
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
        
    cat_enc = encoder.transform(df[cat_cols])
    num_scaled = scaler.transform(df[num_cols])
        
    X = np.hstack([num_scaled, cat_enc])
    feature_names = num_cols + list(encoder.get_feature_names_out(cat_cols))
    
    return X, y, encoder, scaler, feature_names

X_train, y_train, ohe, std_scaler, feat_names = get_X_y(df_train)

# --- 3. ОБУЧЕНИЕ МОДЕЛИ ---

@st.cache_resource
def train_ridge(X, y):
    # Убеждаемся, что y — это исходные рубли. 
    # Если цена была в тыс., проверьте данные, но обычно в CSV она полная.
    params = {'alpha': np.logspace(-2, 3, 10)}
    grid = GridSearchCV(Ridge(), params, cv=5, scoring='neg_root_mean_squared_error')
    grid.fit(X, y)
    return grid.best_estimator_

model = train_ridge(X_train, y_train)

# --- 4. ИНТЕРФЕЙС STREAMLIT ---

tabs = st.tabs(["📊 Анализ (EDA)", "🔮 Предсказание", "🧮 Веса модели"])

with tabs[0]:
    st.header("Анализ данных")
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
        st.pyplot(fig2)

with tabs[1]:
    mode = st.radio("Способ ввода:", ["Вручную", "Загрузить CSV"])
    
    if mode == "Вручную":
        c1, c2 = st.columns(2)
        with c1:
            name = st.selectbox("Марка", sorted(df_train['name'].unique()))
            year = st.slider("Год", 1990, 2024, 2017)
            km = st.number_input("Пробег (км)", 0, 1000000, 60000)
            fuel = st.selectbox("Топливо", df_train['fuel'].unique())
            seats = st.selectbox("Мест", sorted(df_train['seats'].unique()))
        with c2:
            trans = st.selectbox("КПП", df_train['transmission'].unique())
            sell = st.selectbox("Продавец", df_train['seller_type'].unique())
            owner = st.selectbox("Владелец", df_train['owner'].unique())
            power = st.number_input("Мощность (hp)", 30, 600, 100)
            torque_str = st.text_input("Torque (например: '190Nm@ 2000rpm')", "190Nm@ 2000rpm")
            
        if st.button("Рассчитать стоимость"):
            input_df = pd.DataFrame([{
                'name': name, 'year': year, 'km_driven': km, 'fuel': fuel,
                'seller_type': sell, 'transmission': trans, 'owner': owner,
                'mileage': medians['mileage'], 'engine': medians['engine'],
                'max_power': float(power), 'torque': torque_str, 'seats': seats
            }])
            
            # Сначала чистим строки (включая torque)
            input_proc = process_strings(input_df)
            # Заполняем пропуски в torque_rpm если не распарсилось
            input_proc['max_torque_rpm'] = input_proc['max_torque_rpm'].fillna(medians['max_torque_rpm'])
            
            X_input, _, _, _, _ = get_X_y(input_proc, False, ohe, std_scaler)
            prediction = model.predict(X_input)[0]
            st.metric("Оценочная стоимость", f"{max(0, round(prediction)):,} ₽")

    else:
        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_file:
            test_upload = pd.read_csv(uploaded_file)
            try:
                # ВАЖНО: Очищаем загруженный файл тем же методом
                test_proc = process_strings(test_upload)
                # Заполняем NaN медианами, чтобы scaler не выдал ошибку
                for col in num_features:
                    test_proc[col] = test_proc[col].fillna(medians[col])

                X_up, _, _, _, _ = get_X_y(test_proc, False, ohe, std_scaler)
                preds = model.predict(X_up)
                
                res_df = test_upload.copy()
                res_df['predicted_price'] = preds.round(0)
                st.dataframe(res_df)
            except Exception as e:
                st.error(f"Ошибка в структуре файла: {e}")

with tabs[2]:
    importance = pd.DataFrame({'Feature': feat_names, 'Weight': model.coef_}).sort_values(by='Weight', ascending=False)
    top_bottom = pd.concat([importance.head(10), importance.tail(10)])
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_bottom, x='Weight', y='Feature', palette='RdYlGn', ax=ax3)
    st.pyplot(fig3)
