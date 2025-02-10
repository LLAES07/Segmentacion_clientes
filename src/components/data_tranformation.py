
import os, sys,re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.exception import CustomExecption
from src.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')


class EliminaOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns


    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)  # Asegura que X siga siendo un DataFrame
        print(X.columns) 

        # en el caso que no se especifique las columnas todas las numericas
        if self.columns is None:
            self.columns = X.select_dtypes(include=['number']).columns  # Si no se especifican, usa todas las numéricas

        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]  # Filtrar los valores dentro del rango

        return X
    


class InvoiceNoConverter(BaseEstimator, TransformerMixin):
    def __init__(self, column='InvoiceNo'):
        self.column = column

    def fit(self, X, y=None):
        return self  # No necesita aprender nada, solo transforma

    def transform(self, X):
        X = X.copy()  # Evita modificar el DataFrame original

        try:
            X[self.column] = pd.to_numeric(X[self.column])
        except:
            X[self.column] = (
                X[self.column]
                .astype(str)
                .str.replace(r'[a-zA-Z]+', '', regex=True)  # Remueve letras
                .astype(int)
            )

        return X
    


class MakePositiveTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # Lista de columnas a convertir

    def fit(self, X, y=None):
        return self  # No aprende nada, solo transforma

    def transform(self, X):
        X = X.copy()  # Evita modificar el DataFrame original

        if self.columns is None:
            self.columns = X.select_dtypes(include=['number']).columns  # Si no se especifican, usa todas las numéricas

        for col in self.columns:
            X[col] = X[col].apply(lambda x: abs(x))  # Convierte los valores a positivos

        return X


class RFMCreator(BaseEstimator, TransformerMixin):
    def __init__(self, invoice_col='InvoiceNo', date_col='InvoiceDate', customer_col='CustomerID', price_col='price_total'):
        self.invoice_col = invoice_col
        self.date_col = date_col
        self.customer_col = customer_col
        self.price_col = price_col

    def fit(self, X, y=None):
        return self  # No necesita aprender nada, solo transforma

    def transform(self, X):
        X = X.copy()

        X[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')
        # Recencia: Última compra por cliente
        fecha_maxima = X.groupby(self.customer_col)[self.date_col].max().reset_index()
        fecha_maxima.columns = [self.customer_col, 'FechaReciente']
        X = pd.merge(X, fecha_maxima, on=self.customer_col, how='left')
         # 🔹 Convertir `FechaReciente` a datetime si no lo es
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
    """Aplica StandardScaler() pero mantiene la estructura de DataFrame"""
    
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


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            invoice_col = ['InvoiceNo']
            numerical_cols = ['Quantity', 'price_total', 'StockCode']
            date_col = ['InvoiceDate']
            customer_col = ['CustomerID']


            num_pipeline = Pipeline([
                ('solo_positivos', MakePositiveTransformer(columns=['Quantity', 'price_total'])),  # Convierte valores negativos a positivos
                ('removiendo_outliers', EliminaOutliers(columns=numerical_cols[:-1])),  # Eliminar outliers
                ('scaler', DataFrameStandardScaler(columns=numerical_cols))  # Normalizar datos
            ])

            # Pipeline para la conversión de InvoiceNo
            invoice_pipeline = Pipeline([
                ('convert_invoice', InvoiceNoConverter(column='InvoiceNo'))
            ])

            rfm_pipeline = Pipeline([
                ('rfm_creator', RFMCreator())
            ])


            # Integrar todo en un ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, numerical_cols),
                ('invoice', invoice_pipeline, invoice_col),
                ('rfm', rfm_pipeline, [customer_col[0], date_col[0], invoice_col[0], numerical_cols[1]])  # InvoiceDate, InvoiceNo, price_total

            ])
            
            logging.info('')
            return preprocessor
        except Exception as e:
            raise CustomExecption(e, sys)
        

    def iniciar_transformacion_datos(self, train_path):

        try:
            train_set = pd.read_csv(train_path)

            logging.info('Lectura de los datos de entrenamiento completada')

            preprocessing_obj = self.get_data_transformer_object()

            logging.info('Procesador iniciado y listo para trabajar')
            train_features = train_set.columns
            train_arr = preprocessing_obj.fit_transform(train_set)
            
            train_set = pd.DataFrame(data=train_arr, columns=train_features)

            train_set.to_csv('artifacts/data_cleaned.csv', index=False, header=True)
            
            logging.info('Datos procesados. Guardado csv limpio.')


            logging.info('Generando train set limpio para el entrenamiento')

            train_set_cleaned = train_set.drop(columns=['Unnamed: 0', 'InvoiceNo', 'InvoiceDate', 'CustomerID'], axis=1)
            logging.info('Completado la transformación y eliminación de columnas para el entrenamiento')

            train_set_cleaned.to_csv('artifacts/train_set_cleaned.csv')

            return (
                train_set_cleaned,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomExecption(e, sys)

if __name__ == "__main__":
    
    train_path = "artifacts/train.csv"

    data_transformer = DataTransformation()

    train_set_cleaned, preprocessor_path = data_transformer.iniciar_transformacion_datos(train_path)

    print(train_set_cleaned.head())

    logging.info("✅ Transformación de datos finalizada correctamente.")