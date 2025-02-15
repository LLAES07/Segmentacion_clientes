import os
import sys
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomExecption
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')


class InvoiceNoConverter(BaseEstimator, TransformerMixin):
    def __init__(self, column='InvoiceNo'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            X[self.column] = pd.to_numeric(X[self.column])
        except Exception:
            X[self.column] = (X[self.column]
                              .astype(str)
                              .str.replace(r'[a-zA-Z]+', '', regex=True)
                              .astype(int))
        return X


class MakePositiveTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # Lista de columnas a convertir

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=['number']).columns
        for col in self.columns:
            X[col] = X[col].abs()

        X = X.loc[X['price_total']!=0, :]
        
        return X


class IQRRemover(BaseEstimator, TransformerMixin):
    """
    Transformer que elimina filas que son outliers en las columnas especificadas,
    usando el método del rango intercuartílico (IQR).
    """
    def __init__(self, columns):
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self

    def transform(self, X):
        X = X.copy()
        # Aplicamos el filtrado iterativamente para cada columna.
        for col in self.columns:
            lower_bound, upper_bound = self.bounds_[col]
            X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        return X


class RFMCreator(BaseEstimator, TransformerMixin):
    def __init__(self, invoice_col='InvoiceNo', date_col='InvoiceDate', customer_col='CustomerID', price_col='price_total'):
        self.invoice_col = invoice_col
        self.date_col = date_col
        self.customer_col = customer_col
        self.price_col = price_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Convertir a datetime la columna de fecha
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')
        # Recencia: Última compra por cliente
        fecha_maxima = X.groupby(self.customer_col)[self.date_col].max().reset_index()
        fecha_maxima.columns = [self.customer_col, 'FechaReciente']
        X = pd.merge(X, fecha_maxima, on=self.customer_col, how='left')
        X['FechaReciente'] = pd.to_datetime(X['FechaReciente'], errors='coerce')
        X['Recencia'] = (X['FechaReciente'] - X[self.date_col]).dt.days
        X.drop(columns='FechaReciente', inplace=True)
        # Frecuencia: Número de compras por cliente
        frecuencia_cliente = X.groupby(self.customer_col)[self.invoice_col].nunique().reset_index()
        frecuencia_cliente.columns = [self.customer_col, 'Frecuencia']
        X = pd.merge(X, frecuencia_cliente, on=self.customer_col, how='left')
        # Monto: Total gastado por cliente
        monto_cliente = X.groupby(self.customer_col)[self.price_col].sum().reset_index()
        monto_cliente.columns = [self.customer_col, 'Monto']
        X = pd.merge(X, monto_cliente, on=self.customer_col, how='left')
        return X


class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    """Aplica StandardScaler() pero mantiene la estructura de DataFrame."""
    def __init__(self, columns):
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

    def set_output(self, transform):
        # Implementación mínima para soportar set_output.
        self._transform_output = transform
        return self

    def _more_tags(self):
        return {"preserves_dataframe": True}


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def iniciar_transformacion_datos(self, train_path):



        try:
            # Lectura de datos
            train_set = pd.read_csv(train_path)
            logging.info('Lectura de los datos de entrenamiento completada.')

            logging.info('Ejecutando pipeline secuencial de transformaciones.')

            # Pipeline secuencial:
            # 1. Conversión de InvoiceNo
            # 2. Convertir a valores positivos en columnas numéricas
            # 3. Remover outliers por IQR en 'Quantity' y 'price_total'
            # 4. Creación de columnas RFM
            sequential_pipeline = Pipeline([
                ('invoice_conversion', InvoiceNoConverter(column='InvoiceNo')),
                ('outlier_removal', IQRRemover(columns=['Quantity', 'price_total'])),
                ('make_positive', MakePositiveTransformer(columns=['Quantity', 'price_total'])),
                ('rfm_creator', RFMCreator(invoice_col='InvoiceNo', 
                                           date_col='InvoiceDate', 
                                           customer_col='CustomerID', 
                                           price_col='price_total'))
            ])

            data_transformed = sequential_pipeline.fit_transform(train_set)

            data_transformed.to_csv('artifacts/data_cleaned_no_scaled.csv', index=False, header=True)
            logging.info(f'Datos limpios sin escalar y RFM artifacts/data_cleaned.csv. Contiene {data_transformed.shape} ')

            # Definir las columnas numéricas a escalar (incluyendo las nuevas columnas RFM)
            numeric_cols = ['Quantity', 'price_total', 'StockCode', 'Recencia', 'Frecuencia', 'Monto']

            # ColumnTransformer para aplicar StandardScaler solo a las columnas numéricas.
            scaler_transformer = ColumnTransformer(
                transformers=[
                    ('scaler', DataFrameStandardScaler(columns=numeric_cols[2:]), numeric_cols)
                ],
                remainder='passthrough',
                verbose_feature_names_out=False  # Conservar los nombres originales
            )

            # Forzar la salida a DataFrame (requiere scikit-learn >= 1.2)
            scaler_transformer.set_output(transform="pandas")

            


            final_transformed = scaler_transformer.fit_transform(data_transformed)

            # Guardar DataFrame final en CSV
            final_transformed.to_csv('artifacts/data_cleaned_scaled.csv', index=False, header=True)
            logging.info('Datos limpios y transformados artifacts/data_cleaned_scaled.csv.')

            # (Opcional) Eliminar columnas no deseadas para el entrenamiento final
            columns_to_drop = ['Unnamed: 0', 'InvoiceNo', 'InvoiceDate', 'CustomerID']
            train_set_cleaned = final_transformed.drop(columns=columns_to_drop, errors='ignore')
            train_set_cleaned.to_csv('artifacts/train_set_cleaned.csv', index=False)
            logging.info('Train set limpio guardado en artifacts/train_set_cleaned.csv.')

            save_object (
                self.data_transformation_config.preprocessor_obj_file_path, scaler_transformer
            )

            return train_set_cleaned, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomExecption(e, sys)

if __name__ == "__main__":
    train_path = "artifacts/train.csv"
    data_transformer = DataTransformation()
    train_set_cleaned, preprocessor_path = data_transformer.iniciar_transformacion_datos(train_path)
    print(train_set_cleaned.head())
    logging.info(" Transformación de datos finalizada correctamente.")
