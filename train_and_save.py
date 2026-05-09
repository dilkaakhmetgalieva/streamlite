import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from custom_transformers import CarDataCleaner

df = pd.read_csv("car_data.csv")

target_col = "selling_price"

X = df.drop(columns=[target_col])
y = df[target_col]

cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'max_torque', 'max_torque_rpm']

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]),
            num_cols
        ),
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]),
            cat_cols
        )
    ]
)

main_pipeline = Pipeline([
    ('cleaner', CarDataCleaner()),
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=162.3777))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

main_pipeline.fit(X_train, y_train)

y_pred = main_pipeline.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

joblib.dump(main_pipeline, "car_price_pipeline.pkl")
joblib.dump(df, "df_train.pkl")

print("Модель и данные успешно сохранены.")
