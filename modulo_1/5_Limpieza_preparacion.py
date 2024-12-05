import pandas as pd  # Para manipulación de datos
import numpy as np   # Para cálculos numéricos
import matplotlib.pyplot as plt  # Para visualización
from sklearn.preprocessing import StandardScaler

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
Ejercicios:
1. Trabajar con valores faltantes:
"""
# Introducir valores faltantes:
df.loc[10, 'sepal_length'] = np.nan
df.loc[20, 'sepal_width'] = np.nan
df.loc[30, 'sepal_length'] = np.nan
df.loc[40, 'sepal_width'] = np.nan

# Identificar valores faltantes:
print("Valores faltantes por columna:")
print(df.isnull().sum())

""" 
Resultado:
Valores faltantes por columna:
sepal_length    2
sepal_width     2
petal_length    0
petal_width     0
species         0
dtype: int64
"""

""" Manejar valores faltantes: """
# Eliminar filas con valores faltantes
df_dropped = df.dropna()
removed_rows = len(df) - len(df_dropped)
print(f"\nSe eliminaron {removed_rows} filas con valores faltantes. El DataFrame ahora tiene {len(df_dropped)} filas restantes:\n")
print(df_dropped)
""" 
Resultado:
Se eliminaron 4 filas con valores faltantes. El DataFrame ahora tiene 146 filas restantes:

     sepal_length  sepal_width  petal_length  petal_width         species
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[146 rows x 5 columns]
"""

# Imputar valores faltantes con la media de cada columna
df_imputed = df.fillna(df.mean(numeric_only=True))
print("\nDataFrame después de imputar valores faltantes con la media:\n")
print(df_imputed)
""" 
Resultado:
DataFrame después de imputar valores faltantes con la media:

     sepal_length  sepal_width  petal_length  petal_width         species
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[150 rows x 5 columns]
"""

"""2. Manejar Datos duplicados:"""

# Introducir filas duplicadas
df_duplicates = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
print("\nDataFrame con filas duplicadas añadidas:")
print(df_duplicates)

""" 
Resultado:
DataFrame con filas duplicadas añadidas:
     sepal_length  sepal_width  petal_length  petal_width         species
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica
150           5.1          3.5           1.4          0.2     Iris-setosa
151           4.9          3.0           1.4          0.2     Iris-setosa

[152 rows x 5 columns]
"""

# Eliminar duplicados
df_no_duplicates = df_duplicates.drop_duplicates()
print("\nDataFrame después de eliminar duplicados:")
print(df_no_duplicates)
""" 
Resultado:
DataFrame después de eliminar duplicados:
     sepal_length  sepal_width  petal_length  petal_width         species
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[147 rows x 5 columns]
"""
""" Transformación de datos"""

# Escalador estándar
scaler = StandardScaler()

# Selección de características numéricas
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Crear un nuevo DataFrame con datos escalados
df_scaled = df_no_duplicates.copy()
df_scaled[features] = scaler.fit_transform(df_no_duplicates[features])

# Mostrar un resumen de los datos escalados
print("\nDatos escalados (media = 0, desviación estándar = 1):")
print(df_scaled[features].describe())
