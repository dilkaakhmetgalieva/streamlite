import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Настройка страницы
st.set_page_config(page_title="auto-predict", layout="wide")
st.title("Предсказание стоимости автомобиля")

# Константы
TRAIN_URL = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
TEST_URL = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'

# Глобальные списки признаков
cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']

def process_strings(df):
    temp = df.copy()

    for col in ['mileage', 'engine', 'max_power']:
        if col in temp.columns:
            temp[col] = temp[col].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    if 'torque' in temp.columns:
        s = temp['torque'].astype(str).str.lower()

        torque_val = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
        
        is_kgm = s.str.contains('kgm', na=False)
        torque_val[is_kgm] = torque_val[is_kgm] * 9.8

        nums = s.str.findall(r'\d+')

        def get_last_num(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    return float(x[-1].replace(',', ''))
                except:
                    return np.nan
            return np.nan

        rpm = nums.apply(get_last_num)

        temp['torque'] = torque_val
        temp['max_torque_rpm'] = rpm

    return temp

@st.cache_data
def load_and_clean():
    df_train = pd.read_csv(TRAIN_URL)
    df_test = pd.read_csv(TEST_URL)

    cols_check = [c for c in df_train.columns if c != 'selling_price']
    df_train = df_train.drop_duplicates(subset=cols_check).reset_index(drop=True)

    df_train = process_strings(df_train)
    df_test = process_strings(df_test)

    med_cols = [c for c in num_cols if c in df_train.columns]
    medians = df_train[med_cols].median()

    for col in num_cols:
        if col not in df_train.columns:
            df_train[col] = medians.get(col, 0)
        if col not in df_test.columns:
            df_test[col] = medians.get(col, 0)

    for col in num_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna(medians.get(col, 0))
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(medians.get(col, 0))

    for col in cat_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna('unknown').astype(str)
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna('unknown').astype(str)

    return df_train, df_test, medians


df_train, df_test, medians = load_and_clean()

def get_X_y(df, is_train=True, encoder=None, scaler=None):
    df = df.copy()

    if 'name' in df.columns:
        df['name'] = df['name'].astype(str).str.split().str[0]
    else:
        df['name'] = 'unknown'

    for col in num_cols:
        if col not in df.columns:
            df[col] = medians.get(col, 0)

    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'unknown'

    for col in num_cols:
        df[col] = df[col].fillna(medians.get(col, 0))

    for col in cat_cols:
        df[col] = df[col].fillna('unknown').astype(str)

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

@st.cache_resource
def train_ridge(X, y):
    params = {'alpha': np.logspace(-2, 3, 10)}
    grid = GridSearchCV(Ridge(), params, cv=10, scoring='neg_root_mean_squared_error')
    grid.fit(X, y)
    return grid.best_estimator_

model = train_ridge(X_train, y_train)

tabs = st.tabs(["Основные графики", "Предсказание стоимости автомобиля", "Веса модели"])

with tabs[0]:
    st.header("Визуализация данных")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение целевой переменной")
        fig1, ax1 = plt.subplots()
        sns.histplot(df_train['selling_price'], kde=True, color='green', ax=ax1)
        ax1.set_title("Распределение цены (selling_price)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Матрица корреляций")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        # Объединяем числовые признаки и целевую переменную для корреляции
        corr_cols = [c for c in num_cols if c in df_train.columns] + ['selling_price']
        corr_matrix = df_train[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        ax2.set_title("Корреляция признаков и цены")
        st.pyplot(fig2)

    st.divider()
    st.subheader("Попарные отношения признаков")
    

    pairplot_cols = ['selling_price', 'year', 'km_driven', 'max_power', 'engine']
    available_cols = [c for c in pairplot_cols if c in df_train.columns]
    
    df_sample = df_train[available_cols].sample(min(500, len(df_train)))
    
    fig4 = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.5})
    st.pyplot(fig4.figure)

with tabs[1]:
    mode = st.radio("Способ ввода:", ["Вручную", "Загрузить CSV"])

    if mode == "Вручную":
        c1, c2 = st.columns(2)

        with c1:
            name = st.selectbox("Марка", sorted(df_train['name'].dropna().unique()))
            year = st.slider("Год", 1990, 2024, 2017)
            km = st.number_input("Пробег (км)", 0, 1000000, 60000)
            fuel = st.selectbox("Топливо", sorted(df_train['fuel'].dropna().unique()))
            seats = st.selectbox("Мест", sorted(df_train['seats'].dropna().unique()))

        with c2:
            trans = st.selectbox("КПП", sorted(df_train['transmission'].dropna().unique()))
            sell = st.selectbox("Продавец", sorted(df_train['seller_type'].dropna().unique()))
            owner = st.selectbox("Владелец", sorted(df_train['owner'].dropna().unique()))
            power = st.number_input("Мощность (hp)", 30, 600, 100)
            torque_str = st.text_input("Torque (например: '190Nm@ 2000rpm')", "190Nm@ 2000rpm")

        if st.button("Рассчитать стоимость"):
            input_df = pd.DataFrame([{
                'name': name,
                'year': year,
                'km_driven': km,
                'fuel': fuel,
                'seller_type': sell,
                'transmission': trans,
                'owner': owner,
                'mileage': medians.get('mileage', 0),
                'engine': medians.get('engine', 0),
                'max_power': float(power),
                'torque': torque_str,
                'seats': seats
            }])

            input_proc = process_strings(input_df)

            if 'max_torque_rpm' not in input_proc.columns:
                input_proc['max_torque_rpm'] = medians.get('max_torque_rpm', 0)

            input_proc['max_torque_rpm'] = input_proc['max_torque_rpm'].fillna(medians.get('max_torque_rpm', 0))

            X_input, _, _, _, _ = get_X_y(input_proc, False, ohe, std_scaler)
            prediction = model.predict(X_input)[0]
            st.metric("Оценочная стоимость", f"{max(0, round(prediction)):,} ₽")

    else:
        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_file:
            test_upload = pd.read_csv(uploaded_file)
            try:
                test_proc = process_strings(test_upload)

                # Создаём недостающие числовые колонки
                for col in num_cols:
                    if col not in test_proc.columns:
                        test_proc[col] = medians.get(col, 0)
                    else:
                        test_proc[col] = test_proc[col].fillna(medians.get(col, 0))

                for col in cat_cols:
                    if col not in test_proc.columns:
                        test_proc[col] = 'unknown'
                    else:
                        test_proc[col] = test_proc[col].fillna('unknown').astype(str)

                X_up, _, _, _, _ = get_X_y(test_proc, False, ohe, std_scaler)
                preds = model.predict(X_up)

                res_df = test_upload.copy()
                res_df['predicted_price'] = np.round(preds, 0)
                st.dataframe(res_df)

            except Exception as e:
                st.error(f"Ошибка в структуре файла: {e}")

with tabs[2]:
    importance = pd.DataFrame({
        'Feature': feat_names,
        'Weight': model.coef_
    }).sort_values(by='Weight', ascending=False)

    top_bottom = pd.concat([importance.head(10), importance.tail(10)])

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_bottom, x='Weight', y='Feature', palette='RdYlGn', ax=ax3)
    ax3.set_title("Веса признаков модели")
    st.pyplot(fig3)
