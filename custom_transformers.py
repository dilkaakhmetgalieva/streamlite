import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CarDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Обработка названия (берем только марку)
        if 'name' in X.columns:
            X['name'] = X['name'].astype(str).str.split().str[0]

        # Очистка числовых колонок от единиц измерения
        cols_to_clean = ['mileage', 'engine', 'max_power']
        for col in cols_to_clean:
            if col in X.columns:
                X[col] = (
                    X[col].astype(str)
                    .str.replace(r'[^0-9.]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float)
                )

        # Обработка крутящего момента (Torque)
        if 'torque' in X.columns:
            s = X['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
            is_kgm = s.str.contains('kgm', na=False)

            # Извлекаем само значение момента
            extracted_torque = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
            # Извлекаем RPM (берем последнее число в строке)
            extracted_rpm = s.str.findall(r'\d+')
            X['max_torque_rpm'] = extracted_rpm.apply(lambda x: float(x[-1]) if isinstance(x, list) and len(x) > 0 else np.nan)

            X['torque'] = extracted_torque
            # Перевод из kgm в Nm
            X.loc[is_kgm, 'torque'] *= 9.8
            # Очистка пустых
            X.loc[s.eq('nan'), ['torque', 'max_torque_rpm']] = np.nan

        return X
