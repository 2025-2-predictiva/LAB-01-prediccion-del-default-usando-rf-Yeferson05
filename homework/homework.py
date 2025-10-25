# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".

import pandas as pd
import numpy as np
import gzip
import pickle
import json
import os
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

def read_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        csv_file = next((f for f in file_list if f.endswith('.csv')), None)
        
        if csv_file is None:
            raise FileNotFoundError(f"No se encontró archivo CSV en {zip_path}")
        
        with zip_ref.open(csv_file) as file:
            return pd.read_csv(file)

def process_data(df):
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    df = df[df['EDUCATION'] > 0]
    df = df[df['MARRIAGE'] > 0]
    
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    
    return df


# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

train_zip_path = os.path.join(project_dir, 'files', 'input', 'train_data.csv.zip')
test_zip_path = os.path.join(project_dir, 'files', 'input', 'test_data.csv.zip')

train_data = read_csv_from_zip(train_zip_path)
test_data = read_csv_from_zip(test_zip_path)

train_data = process_data(train_data)
test_data = process_data(test_data)

x_train = train_data.drop('default', axis=1)
y_train = train_data['default']
x_test = test_data.drop('default', axis=1)
y_test = test_data['default']


# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).

categorical_vars = ['SEX', 'EDUCATION', 'MARRIAGE']

preprocessor = ColumnTransformer(
    transformers=[
        ('categoricas', OneHotEncoder(handle_unknown='ignore'), categorical_vars)
    ],
    remainder='passthrough'
)

ml_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

hyperparameters = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, None],
    'model__min_samples_split': [10],
    'model__min_samples_leaf': [2, 4],
    'model__max_features': [25]
}

optimization = GridSearchCV(
    ml_pipeline,
    hyperparameters,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

optimization.fit(x_train, y_train)


# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

models_directory = os.path.join(project_dir, 'files', 'models')
os.makedirs(models_directory, exist_ok=True)
model_file = os.path.join(models_directory, 'model.pkl.gz')

with gzip.open(model_file, 'wb') as f:
    pickle.dump(optimization, f)


# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

def compute_metrics_and_confusion(model, x_data, y_data, dataset_type):
    predictions = model.predict(x_data)
    
    metrics_data = {
        'type': 'metrics',
        'dataset': dataset_type,
        'precision': float(precision_score(y_data, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(y_data, predictions)),
        'recall': float(recall_score(y_data, predictions)),
        'f1_score': float(f1_score(y_data, predictions))
    }
    
    conf_matrix = confusion_matrix(y_data, predictions)
    matrix_data = {
        'type': 'cm_matrix',
        'dataset': dataset_type,
        'true_0': {
            'predicted_0': int(conf_matrix[0, 0]),
            'predicted_1': int(conf_matrix[0, 1])
        },
        'true_1': {
            'predicted_0': int(conf_matrix[1, 0]),
            'predicted_1': int(conf_matrix[1, 1])
        }
    }
    
    return metrics_data, matrix_data


# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

train_metrics, train_confusion = compute_metrics_and_confusion(optimization, x_train, y_train, 'train')
test_metrics, test_confusion = compute_metrics_and_confusion(optimization, x_test, y_test, 'test')

output_directory = os.path.join(project_dir, 'files', 'output')
os.makedirs(output_directory, exist_ok=True)
results_file = os.path.join(output_directory, 'metrics.json')

with open(results_file, 'w') as f:
    f.write(json.dumps(train_metrics) + '\n')
    f.write(json.dumps(test_metrics) + '\n')
    f.write(json.dumps(train_confusion) + '\n')
    f.write(json.dumps(test_confusion) + '\n')
