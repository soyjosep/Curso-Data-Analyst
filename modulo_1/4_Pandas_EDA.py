import pandas as pd
import matplotlib.pyplot as plt

# Introducción a pandas y Análisis Exploratorio de Datos(EDA)

""" 
Pandas es una biblioteca esencial en Python para el análisis y manipulación de datos. Facilita la lectura,
manipulación y análisis de conjuntos de datos grandes y complejos.

Objetivos:
    - Familiarizarse con Pandas:
        - Comprender las estructuras de datos básicas: Series y DataFrames.
        - Aprender a importar datos desde archivos CSV o fuentes en línea.
    - Realizar un Análisis Exploratorio de Datos(EDA)
        - Inspección inicial del conjunto de datos.
        - Identificación de valores faltantes y tipos de datos.
        - Cálculo de estadísticas descriptivas.
        - Visualización de relaciones entre variables.      
"""

""" 
1. Importar un conjunto de Datos:
    - Descarga el conjunto de datos Iris desde este enlace: iris.csv
    - Si no puedes descargar el archivo, puedes cargarlo directamente desde la URL en tu código.
"""


# URL del conjunto de datos Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Leer el CSV desde la URL
iris_data = pd.read_csv(url, header=None, names=column_names)

# Mostrar las primeras filas
print(iris_data.head())

"""
2. Cargar el conjunto de Datos en un DataFrame de Pandas:
    - Importa las bibliotecas necesarias(pandas y matplotlib).
    - Lee el archivo CSV y asigna el DataFrame a una variable, por ejemplo, df.
"""


# Definir la URL o la ruta del archivo
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Nombres de las columnas del conjunto de datos
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(url, header=None, names=column_names)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Crear un gráfico de dispersión (scatter plot)
df.plot(kind='scatter', x='sepal_length', y='sepal_width', alpha=0.5)
plt.title('Sepal Length vs Sepal Width')
plt.show()

"""
3. Realizar un Análisis exploratorio Básico:
    - Inspección Inicial:
        - Muestra las primeras 5 filas del DataFrame con df.head().
        - Obtén información general del DataFrame con df.info().
"""

# Mostrar las primeras 5 filas del DataFrame
print("Primeras 5 filas del DataFrame:")
print(df.info())

# Obtener información general del DataFrame
print("\nInformación general del DataFrame:")
print(df.info())

"""
    - Estadísticas Descriptivas:
        - Utiliza df.describe() para calcular estadísticas como media, mediana y desviación estándar 
        de cada característica numérica.
    - Visualización:
        - Crea un gráfico de dispersión (scatter plot) de sepal_length vs sepal_width.
        - Distingue las diferentes especies utilizando colores o marcadores distintos.
        - Añade etiquetas y título al gráfico para una mejor comprensión.

Instrucciones adicionales:
    - Análisis: Después de crear el gráfico, analiza si hay patrones o relaciones entre las variables y las especies.
    - Preguntas para Reflexionar:
        - ¿Puedes identificar agrupaciones de especies en el gráfico?
        - ¿Qué te indican las estadísticas descriptivas sobre cada característica?
"""