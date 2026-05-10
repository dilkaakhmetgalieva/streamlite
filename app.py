import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Прогноз стоимости автомобилей")

# --- 1. ФУНКЦИИ ПРЕДОБРАБОТКИ ---
CARS_TRAIN = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
CARS_TEST = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

df_train = pd.read_csv(CARS_TRAIN)
df_test = pd.read_csv(CARS_TEST)

df_train_filtred = df_train.drop(columns=['selling_price'])
cols = df_train_filtred.columns
df_train = df_train.drop_duplicates(subset=cols, keep='first')
df_train = df_train.reset_index(drop=True)

def cleaner(df):
    #обработываем простые числовые столбцы (mileage, engine, max_power)
    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if df[col].dtype == object: #это я добавила, чтобы при случайном перезапуске этой ячейки, всё не сломалось, так как объекты станут float и проверку на object уже не пройдут
            df[col] = (df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True).replace('', np.nan).replace('nan', np.nan).astype(float))

    #обрабатываем torque
    if 'torque' in df.columns and df['torque'].dtype == object:
        s = df['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
        is_kgm = s.str.contains('kgm', na=False)
        extracted_torque = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
        df['max_torque_rpm'] = s.str.findall(r'\d+').str[-1].astype(float)
        df['torque'] = extracted_torque
        df.loc[is_kgm, 'torque'] *= 9.8
        df.loc[s.eq('nan'), ['torque', 'max_torque_rpm']] = np.nan

    return df

df_train = cleaner(df_train)
df_test = cleaner(df_test)

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

df_train['engine'] = df_train['engine'].astype(int)
df_train['seats'] = df_train['seats'].astype(int)

df_test['engine'] = df_test['engine'].astype(int)
df_test['seats'] = df_test['seats'].astype(int)

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


# --- 2. ОБУЧЕНИЕ МОДЕЛИ ---

scaler = StandardScaler()

X_train_scaled_cat = scaler.fit_transform(numeric_train_cat)
X_test_scaled_cat = scaler.transform(numeric_test_cat)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

alphas = np.logspace(-2, 3, 20)
searcher = GridSearchCV(Ridge(), [{
    "alpha": alphas
}],
                        scoring="neg_root_mean_squared_error",
                        cv=10)
searcher.fit(X_train_scaled_cat, y_train)

best_alpha = searcher.best_params_["alpha"]
print("Best alpha = %.4f" % best_alpha)

lr_ridge_best = Ridge(alpha=best_alpha)
lr_ridge_best.fit(X_train_scaled_cat, y_train)

pred_test = lr_ridge_best.predict(X_test_scaled_cat)
mse = mean_squared_error(y_test, pred_test)

pred_train7 = lr_ridge_best.predict(X_train_scaled_cat)
pred_test7 = lr_ridge_best.predict(X_test_scaled_cat)

# --- 3. ИНТЕРФЕЙС STREAMLIT ---
tabs = st.tabs(["EDA", "Предсказание", "Веса модели"])

# Вкладка 1: EDA
with tabs[0]:
    st.header("Анализ данных (EDA)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Распределение цены")
        fig, ax = plt.subplots()
        sns.histplot(df_train['selling_price'], kde=True, ax=ax, color='green')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Мощность vs Цена")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_train, x='max_power', y='selling_price', alpha=0.5, ax=ax)
        st.pyplot(fig)

    with col3:
        st.subheader("Корреляция признаков")
        fig, ax = plt.subplots()
        sns.heatmap(df_train[num_cols + ['selling_price']].corr(), annot=True, cmap='RdYlGn', ax=ax)
        st.pyplot(fig)

# Вкладка 2: ПРЕДСКАЗАНИЕ
with tabs[1]:
    st.header("Применить модель")
    
    mode = st.radio("Выберите способ ввода:", ("Вручную", "Загрузить CSV"))
    
    if mode == "Вручную":
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            name = st.selectbox("Марка", df_train['name'].str.split().str[0].unique())
            year = st.slider("Год выпуска", 1990, 2023, 2015)
            km_driven = st.number_input("Пробег (км)", 0, 500000, 50000)
            fuel = st.selectbox("Топливо", df_train['fuel'].unique())
            seller_type = st.selectbox("Продавец", df_train['seller_type'].unique())
            
        with col_in2:
            transmission = st.selectbox("КПП", df_train['transmission'].unique())
            owner = st.selectbox("Владельцы", df_train['owner'].unique())
            mileage = st.number_input("Расход топлива", 5.0, 40.0, 20.0)
            engine = st.number_input("Объем двигателя", 600, 5000, 1200)
            max_power = st.number_input("Мощность (л.с.)", 30.0, 500.0, 80.0)
            seats = st.selectbox("Мест", sorted(df_train['seats'].unique()))

        # torque и max_torque_rpm возьмем средние для простоты ручного ввода
        input_data = pd.DataFrame([{
            'name': name, 'year': year, 'km_driven': km_driven, 'fuel': fuel,
            'seller_type': seller_type, 'transmission': transmission, 'owner': owner,
            'mileage': mileage, 'engine': engine, 'max_power': max_power,
            'torque': df_train['torque'].median(), 'max_torque_rpm': df_train['max_torque_rpm'].median(),
            'seats': seats
        }])
        
        if st.button("Предсказать цену"):
            # Кодирование и скалирование
            num_part = input_data[num_cols]
            cat_part = pd.DataFrame(ohe.transform(input_data[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
            full_input = pd.concat([num_part, cat_part], axis=1)
            full_input_scaled = scaler.transform(full_input)
            
            prediction = lr_ridge_best.predict(full_input_scaled)[0]
            st.success(f"Ориентировочная стоимость: {round(prediction, 2)} руб.")

    else:
        uploaded_file = st.file_uploader("Загрузите CSV файл (с теми же колонками, что в train)", type="csv")
        if uploaded_file:
            test_batch = pd.read_csv(uploaded_file)
            # Применяем очистку и те же этапы
            test_batch_cleaned = clean_engine_mileage_power(test_batch)
            # Тут в идеале добавить заполнение NaN и OHE...
            st.write("Результаты предсказания для первых 5 строк:")
            # Упрощенно выведем заглушку или можно прогнать цикл обработки
            st.info("Функционал пакетной обработки требует строгого соответствия колонок CSV.")

# Вкладка 3: ВЕСА МОДЕЛИ
with tabs[2]:
    st.header("Коэффициенты модели Ridge")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': model.coef_
    }).sort_values(by='Weight', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    # Покажем топ-20 самых влиятельных признаков
    sns.barplot(data=pd.concat([coef_df.head(10), coef_df.tail(10)]), x='Weight', y='Feature', ax=ax, palette='coolwarm')
    st.pyplot(fig)
    st.dataframe(coef_df)
