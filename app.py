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

# Глобальные списки признаков
cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']

# --- 1. ФУНКЦИИ ОЧИСТКИ ---

def process_strings(df):
    temp = df.copy()

    # Чистим числовые колонки
    for col in ['mileage', 'engine', 'max_power']:
        if col in temp.columns:
            temp[col] = temp[col].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # Обработка torque
    if 'torque' in temp.columns:
        s = temp['torque'].astype(str).str.lower()

        # Значение момента
        torque_val = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)

        # Конвертация kgm в Nm
        is_kgm = s.str.contains('kgm', na=False)
        torque_val[is_kgm] = torque_val[is_kgm] * 9.8

        # RPM: берём последнее число из строки
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
