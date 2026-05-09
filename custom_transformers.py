import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CarDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if 'mileage' in X.columns:
            X['mileage'] = X['mileage'].astype(str).str.extract(r'([\d.]+)')[0]
            X['mileage'] = pd.to_numeric(X['mileage'], errors='coerce')

        if 'engine' in X.columns:
            X['engine'] = X['engine'].astype(str).str.extract(r'([\d.]+)')[0]
            X['engine'] = pd.to_numeric(X['engine'], errors='coerce')

        if 'max_power' in X.columns:
            X['max_power'] = X['max_power'].astype(str).str.extract(r'([\d.]+)')[0]
            X['max_power'] = pd.to_numeric(X['max_power'], errors='coerce')

        if 'torque' in X.columns:
            X['torque'] = X['torque'].astype(str)
            X['max_torque'] = X['torque'].str.extract(r'([\d.]+)')[0]
            X['max_torque_rpm'] = X['torque'].str.extract(r'(\d{3,5})')[0]

            X['max_torque'] = pd.to_numeric(X['max_torque'], errors='coerce')
            X['max_torque_rpm'] = pd.to_numeric(X['max_torque_rpm'], errors='coerce')

            X = X.drop(columns=['torque'], errors='ignore')

        return X
