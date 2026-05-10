import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

def car_data_cleaner(X):
    df = X.copy()
    
    if 'name' in df.columns:
        df['name'] = df['name'].astype(str).apply(lambda x: x.split()[0])
    
    cols_to_clean = ['mileage', 'engine', 'max_power']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r'[^0-9.]', '', regex=True)
                .replace('', np.nan)
                .astype(float)
            )
    
    if 'torque' in df.columns:
        s = df['torque'].astype(str).str.lower().str.replace(',', '', regex=False)
        is_kgm = s.str.contains('kgm', na=False)
        
        df['max_torque_rpm'] = s.str.findall(r'\d+').str[-1].apply(
            lambda x: float(x) if isinstance(x, str) else np.nan
        )
        df['torque'] = s.str.extract(r'(\d+\.?\d*)')[0].astype(float)
        df.loc[is_kgm, 'torque'] *= 9.8
    
    if 'seats' in df.columns:
        df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
        
    return df

# удаление дубликатов ДО пайплайна
df_train = df_train.drop_duplicates(subset=df_train.columns.drop('selling_price'), keep='first').reset_index(drop=True)

X_train = df_train.drop(columns=['selling_price'])
y_train = df_train['selling_price']

cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

full_pipeline = Pipeline([
    ('cleaner', FunctionTransformer(car_data_cleaner, validate=False)),
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=162.3777))
])

full_pipeline.fit(X_train, y_train)

joblib.dump(full_pipeline, 'car_price_model.pkl')
print("Модель сохранена")
