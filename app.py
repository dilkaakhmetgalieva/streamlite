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
st.title("🚗 Прогноз стоимости автомобилей")

# --- 1. ФУНКЦИИ ПРЕДОБРАБОТКИ (из твоего кода) ---
def clean_engine_mileage_power(df):
    df = df.copy()
    # Очистка числовых колонок от единиц измерения
    for col in ['mileage', 'engine', 'max_power']:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Обработка torque
    if 'torque' in df.columns and df['torque'].dtype == object:
        s = df['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
        is_kgm = s.str.contains('kgm', na=False)
        extracted_torque = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
        # Извлекаем последнее число как RPM
        df['max_torque_rpm'] = s.str.findall(r'\d+').str[-1].astype(float)
        df['torque'] = extracted_torque
        df.loc[is_kgm, 'torque'] *= 9.8  # перевод в Nm
    return df

@st.cache_data
def load_and_prep_data():
    # Загрузка (замени пути на свои, если файлы локально)
    train_url = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
    test_url = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'
    
    df_train = pd.read_csv(train_url)
    df_test = pd.read_csv(test_url)
    
    # Удаление дубликатов (кроме целевой переменной)
    cols_no_price = df_train.drop(columns=['selling_price']).columns
    df_train = df_train.drop_duplicates(subset=cols_no_price).reset_index(drop=True)
    
    # Очистка
    df_train = clean_engine_mileage_power(df_train)
    df_test = clean_engine_mileage_power(df_test)
    
    # Заполнение пропусков медианой из train
    numeric_cols = ['mileage', 'engine', 'max
