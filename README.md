# Segmentación de Clientes: Optimización de Estrategias de Marketing

## Descripción del Proyecto
En el competitivo mundo del comercio minorista, la personalización de la experiencia del cliente es clave para el éxito. Este proyecto aborda la segmentación de clientes utilizando técnicas avanzadas de análisis de datos y aprendizaje no supervisado. Basándonos en datos históricos de transacciones, identificamos patrones de compra y clasificamos a los clientes en grupos homogéneos que permiten diseñar estrategias de marketing específicas.

## Objetivo
El principal objetivo es **segmentar a los clientes** para identificar diferentes grupos con características similares, optimizando las estrategias de marketing, promociones y programas de fidelidad. A través de esta segmentación, se busca aumentar la satisfacción del cliente y mejorar la rentabilidad del negocio.

## Fuentes de Datos
El conjunto de datos utilizado incluye información transaccional y de comportamiento del cliente, con las siguientes características:
- **InvoiceNo:** Identificador único de cada transacción.
- **InvoiceDate:** Fecha de la transacción.
- **CustomerID:** Identificador único del cliente.
- **Quantity:** Cantidad total de productos comprados.
- **price_total:** Precio total de la transacción.
- **StockCode:** Cantidad de productos únicos en la transacción.

## Metodología
El análisis se estructuró en las siguientes etapas:

1. **Análisis de Datos Exploratorio (EDA):**
   - Limpieza de datos: Identificación y manejo de valores faltantes y atípicos.
   - Visualización de distribuciones y correlaciones entre las variables clave.

2. **Generación de Variables Derivadas:**
   - Cálculo de métricas RFM (Recencia, Frecuencia y Monto) para resumir el comportamiento de compra de los clientes.
   - Pequeña interpretación.

3. **Aplicación de Algoritmos de Clustering:**
   - Comparación de diferentes métodos:
     - **Clustering Jerárquico.**
     - **K-Means.**
     - **DBSCAN** para detección de clusters densos.
   - Determinación del número óptimo de clusters para cada método.
   - Evaluación de los resultados utilizando métricas como el **Silhouette Score**.


 
4. **Reducción de Dimensionalidad:**
   - Uso de **PCA (Análisis de Componentes Principales)** para simplificar las características y optimizar el rendimiento de los algoritmos de clustering.


5. **Análisis de Resultados:**
   - Caracterización detallada de cada segmento.
   - Visualización de clusters y análisis de su contribución al negocio.

## Resultados
- Identificación de **tres segmentos principales** de clientes con patrones de comportamiento bien definidos.
- Cada cluster fue caracterizado y nombrado para facilitar su interpretación y aplicación:
  - **Cluster 0:** Clientes ocacionales.
  - **Cluster 1:** Clientes frecuentes.
  - **Cluster 2:** Clientes VIP.
- Se destacó que el 80% de los ingresos provienen del Cluster 0 debido a su volumen, mientras que los Clusters 1 y 2 representan clientes estratégicos de alto valor.

## Herramientas Utilizadas
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
- **Algoritmos de Clustering**: DBSCAN, K-Means, Clustering Jerárquico.
- **Reducción de Dimensionalidad**: PCA.
- **Visualización**: Gráficos de Pareto, análisis de distribución y separabilidad de clusters.

## Repositorio del Proyecto
El código y los análisis detallados se encuentran disponibles en el siguiente enlace:
[GitHub: Segmentación de Clientes](https://github.com/LLAES07/project_)

## Conclusión
Este proyecto demuestra cómo el uso de técnicas de machine learning no supervisado, combinado con un análisis sólido de datos, puede ayudar a las empresas a tomar decisiones informadas y estratégicas. Los resultados permiten diseñar campañas de marketing personalizadas que maximizan la rentabilidad y la satisfacción del cliente.


## Instalación y Uso
1. **Clona este repositorio:**
   ```bash
   git clone https://github.com/LLAES07/project_
   cd project_```

2. **Crea y activa un entorno virtual (opcional pero recomendado):**
```bash 
python -m venv venv
source venv/bin/activate # En Windows: venv\Scripts\activate
```  

3. **Instala las dependencias: Asegúrate de tener pip actualizado y luego instala todas las dependencias desde el archivo requirements.txt**
```bash 
pip install -r requirements.txt

```




## Contacto
Para cualquier consulta sobre el proyecto, no dudes en contactarme a través de GitHub o correo electrónico.
