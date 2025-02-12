# Importando las librerias necesarias
import os, sys
from src.exception import CustomExecption
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.data_tranformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_set_path:str = os.path.join('artifacts', 'train.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def inicia_data_ingestion(self):

        logging.info('Iniciando el proceso de lectura de la unyección de componentes')
        try:
            
            df = pd.read_excel('notebook/data/Retail_Invoices.xlsx')
            logging.info('Lectura del archvio exitosa')

            os.makedirs(os.path.dirname(self.ingestion_config.train_set_path), exist_ok=True)
            logging.info('La carpeta ha sido generada o ya está presente')
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Archivo raw ha sido creado')

            df.to_csv(self.ingestion_config.train_set_path, index=False, header=True)
            
            logging.info('Archivo training ha sido generado')

            logging.info('Inyección completada')

            return (
                self.ingestion_config.train_set_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            logging.info('Ha ocurrido un error')
            raise CustomExecption(e, sys)


if __name__=="__main__":

    obj = DataIngestion()
    obj.inicia_data_ingestion()

    train_path = "artifacts/train.csv"
    data_transformer = DataTransformation()
    train_set_cleaned, preprocessor_path = data_transformer.iniciar_transformacion_datos(train_path)
    print(train_set_cleaned.head())
    logging.info("Transformación de datos finalizada correctamente.")