import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, 
                             OPTICS, Birch, SpectralClustering)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.exception import CustomExecption
from src.logger import logging
from src.utils import save_object  # Función para guardar objetos (ya implementada)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self, 
                 apply_pca: bool = False, 
                 n_components: int = 2, 
                 k: int = 3, 
                 eps: float = 0.5, 
                 min_samples: int = 5,
                 optics_min_samples: int = 5, 
                 optics_max_eps: float = None):
        """
        Parámetros:
          - apply_pca: Si se aplica o no PCA antes del clustering.
          - n_components: Número de componentes a retener en el PCA.
          - k: Número de clusters para KMeans, AgglomerativeClustering, SpectralClustering y Birch.
          - eps, min_samples: Parámetros para DBSCAN.
          - optics_min_samples, optics_max_eps: Parámetros para OPTICS.
        """
        self.apply_pca = apply_pca
        self.n_components = n_components
        self.k = k
        self.eps = eps
        self.min_samples = min_samples
        self.optics_min_samples = optics_min_samples
        self.optics_max_eps = optics_max_eps
        self.model_trainer_config = ModelTrainerConfig()

    def iniciar_modelacion(self, train_path: str):
        try:
            # Lectura de datos
            train_set = pd.read_csv(train_path)
            logging.info('Lectura de datos completada.')

            # Seleccionar las columnas de interés para clustering
            data = train_set[['Recencia', 'Frecuencia', 'Monto']]
            
            # Aplicar PCA opcionalmente
            if self.apply_pca:
                logging.info(f'Aplicando PCA con n_components={self.n_components}')
                pca = PCA(n_components=self.n_components, random_state=42)
                data_pca = pca.fit_transform(data)
                data = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(self.n_components)])
            
            # Diccionario para almacenar resultados
            results = {}

            # Función auxiliar para calcular métricas (si hay al menos 2 clusters válidos)
            def compute_metrics(data, labels):
                unique_labels = set(labels) - {-1}
                if len(unique_labels) > 1:
                    sil = silhouette_score(data, labels)
                    ch = calinski_harabasz_score(data, labels)
                    db = davies_bouldin_score(data, labels)
                    return sil, ch, db
                else:
                    return None, None, None

            # 1. KMeans
            kmeans = KMeans(n_clusters=self.k, random_state=42)
            labels_kmeans = kmeans.fit_predict(data)
            sil_k, ch_k, db_k = compute_metrics(data, labels_kmeans)
            results['KMeans'] = {
                "model": kmeans, 
                "silhouette": sil_k, 
                "calinski": ch_k, 
                "davies": db_k, 
                "labels": labels_kmeans
            }

            # 2. Agglomerative Clustering
            agg = AgglomerativeClustering(n_clusters=self.k)
            labels_agg = agg.fit_predict(data)
            sil_a, ch_a, db_a = compute_metrics(data, labels_agg)
            results['Agglomerative'] = {
                "model": agg, 
                "silhouette": sil_a, 
                "calinski": ch_a, 
                "davies": db_a, 
                "labels": labels_agg
            }

            # 3. Birch Clustering
            birch = Birch(n_clusters=self.k)
            labels_birch = birch.fit_predict(data)
            sil_b, ch_b, db_b = compute_metrics(data, labels_birch)
            results['Birch'] = {
                "model": birch, 
                "silhouette": sil_b, 
                "calinski": ch_b, 
                "davies": db_b, 
                "labels": labels_birch
            }

            # 4. DBSCAN
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels_db = dbscan.fit_predict(data)
            sil_db, ch_db, db_db = compute_metrics(data, labels_db)
            results['DBSCAN'] = {
                "model": dbscan, 
                "silhouette": sil_db, 
                "calinski": ch_db, 
                "davies": db_db, 
                "labels": labels_db
            }

            # Mostrar resultados de cada modelo
            logging.info("Resultados de clustering:")
            for model_name, result in results.items():
                if result['silhouette'] is not None:
                    logging.info(
                        f"{model_name}: Silhouette = {result['silhouette']:.3f}, "
                        f"Calinski-Harabasz = {result['calinski']:.3f}, "
                        f"Davies-Bouldin = {result['davies']:.3f}"
                    )
                else:
                    logging.info(f"{model_name}: No se pudieron formar clusters válidos para calcular las métricas.")

            # Determinar el mejor modelo (usando Silhouette Score como criterio principal)
            best_model_name = None
            best_score = -1
            for name, result in results.items():
                if result['silhouette'] is not None and result['silhouette'] > best_score:
                    best_score = result['silhouette']
                    best_model_name = name

            best_model_info = None
            if best_model_name is not None:
                best_model_info = results[best_model_name]
                logging.info(
                    f"El mejor modelo es {best_model_name} con Silhouette = {best_model_info['silhouette']:.3f}, "
                    f"Calinski-Harabasz = {best_model_info['calinski']:.3f}, "
                    f"Davies-Bouldin = {best_model_info['davies']:.3f}"
                )
                save_object(self.model_trainer_config.trained_model_file_path, best_model_info["model"])
                logging.info(f"Modelo guardado en {self.model_trainer_config.trained_model_file_path}")
            else:
                logging.info("No se pudo determinar un mejor modelo debido a la falta de clusters válidos.")

            # Construir un DataFrame con los resultados de cada modelo
            df_results = pd.DataFrame({
                "Modelo": list(results.keys()),
                "Silhouette": [results[m]["silhouette"] for m in results],
                "Calinski-Harabasz": [results[m]["calinski"] for m in results],
                "Davies-Bouldin": [results[m]["davies"] for m in results]
            })
            
            # DataFrame con la información del mejor modelo
            best_model_df = pd.DataFrame({
                "Modelo": [best_model_name],
                "Silhouette": [best_model_info["silhouette"] if best_model_info is not None else None],
                "Calinski-Harabasz": [best_model_info["calinski"] if best_model_info is not None else None],
                "Davies-Bouldin": [best_model_info["davies"] if best_model_info is not None else None]
            })
            
            # Devolver los DataFrames con los resultados
            return {"all_models": df_results, "best_model": best_model_df}

        except Exception as e:
            raise CustomExecption(e, sys)

if __name__ == "__main__":
    train_path = "artifacts/train_set_cleaned.csv"
    # Ejemplo: Probar sin PCA
    trainer_no_pca = ModelTrainer(apply_pca=False, k=3, eps=0.6, min_samples=3, optics_min_samples=5, optics_max_eps=0.5)
    results_no_pca = trainer_no_pca.iniciar_modelacion(train_path)
    
    # Ejemplo: Probar con PCA (reduciendo a 2 componentes)
    trainer_pca = ModelTrainer(apply_pca=True, n_components=2, k=3, eps=0.6, min_samples=3, optics_min_samples=5, optics_max_eps=0.5)
    results_pca = trainer_pca.iniciar_modelacion(train_path)

    print("Resultados sin PCA:")
    print(results_no_pca["all_models"])
    print("Mejor modelo:")
    print(results_no_pca["best_model"])
    
    print("\nResultados con PCA:")
    print(results_pca["all_models"])
    print("Mejor modelo:")
    print(results_pca["best_model"])
