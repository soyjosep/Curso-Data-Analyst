import numpy as np
import matplotlib.pyplot as plt

# URL del conjunto de datos Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Nombres de las columnas del conjunto de datos
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(url, header=None, names=column_names)

"""
Lección 5: Limpieza y preparación de Datos

Explicación:

La limpieza y preparación de datos es esencial para garantizar que el análisis y los modelos construidos sean precisos y fiables.
Los datos del mundo real a menudo contienen inconsistencias, valores faltantes o errores que pueden afectar los resultados.

Objetivos de la Lección:

- Identificar y manejar valores faltantes.
- Detectar y eliminar datos duplicados
- Transformar y normalizar datos numéricos.
- Codificar datos categóricos para uso en modelos.
"""
""" 
Ejercicio 5:

1. Trabajar con valores faltantes:
    - Introducir valores fsaltantes:
"""
df.loc[10, 'sepal_length'] = np.nan
df.loc[20, 'sepal_width'] = np.nan
df.loc[30, 'sepal_length'] = np.nan
df.loc[40, 'sepal_width'] = np.nan



