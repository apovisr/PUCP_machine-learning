
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print(f"Error al cargar los datos: {e}")
    exit()

def clean_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Not specified')
            df[col] = df[col].replace(['Nulo', 'Not specified', 'sin especificar'], 'Not specified')
        else:
            df[col] = df[col].fillna(df[col].median())

    if 'Construction_Area' in df.columns:
        df['Construction_Area'] = df['Construction_Area'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        df['Construction_Area'] = df['Construction_Area'].fillna(df['Construction_Area'].median())

    for col in ['Total_Area_m2', 'Construction_Area_m2']:
         if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    cols_to_drop = ['Id', 'Location', 'District', 'Province', 'Advertiser', 'Publication_Date']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True, drop_first=True)
    return df

y = train_df['Price']
test_ids = test_df['Id']
train_features = train_df.drop('Price', axis=1)

train_cleaned = clean_data(train_features)
test_cleaned = clean_data(test_df.copy())

train_cols = train_cleaned.columns
test_cols = test_cleaned.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    test_cleaned[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    train_cleaned[c] = 0
test_cleaned = test_cleaned[train_cleaned.columns]

X = train_cleaned

plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, bins=50)
plt.title('Distribución de Precios')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.savefig('price_distribution.png')
plt.close()
print("Gráfico de distribución de precios guardado como price_distribution.png")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=20, min_samples_leaf=2, min_samples_split=5)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
r2_rf = r2_score(y_val, y_pred_rf)
print(f'RandomForest R² Score: {r2_rf}')

xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                           n_estimators=1000,
                           learning_rate=0.05,
                           max_depth=7,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           random_state=42,
                           n_jobs=-1,
                           early_stopping_rounds=10)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred_xgb = xgb_model.predict(X_val)
r2_xgb = r2_score(y_val, y_pred_xgb)
print(f'XGBoost R² Score: {r2_xgb}')

if r2_xgb > r2_rf:
    print("Seleccionando XGBoost como el modelo final.")
    final_model = xgb_model
    final_model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
else:
    print("Seleccionando RandomForest como el modelo final.")
    final_model = rf_model
    final_model.fit(X, y)

y_pred_final_val = final_model.predict(X_val)
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_final_val, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title('Predicciones del Modelo Final vs. Valores Reales')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.savefig('final_model_predictions.png')
plt.close()
print("Gráfico de predicciones del modelo final guardado como final_model_predictions.png")

test_predictions = final_model.predict(test_cleaned)
submission_df = pd.DataFrame({'Id': test_ids, 'Price': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print("Archivo submission.csv generado exitosamente.")
