import pandas as pd  # Para manipulación de datos
import numpy as np   # Para cálculos numéricos
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
"""
Transformación de datos
    - Normalización:
"""
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

""" 
Resultado:
Datos escalados (media = 0, desviación estándar = 1):
       sepal_length   sepal_width  petal_length   petal_width
count  1.450000e+02  1.450000e+02  1.470000e+02  1.470000e+02
mean   9.800589e-16 -4.165251e-16 -2.416812e-16 -3.383537e-16
std    1.003466e+00  1.003466e+00  1.003419e+00  1.003419e+00
min   -1.896097e+00 -2.409111e+00 -1.585902e+00 -1.468099e+00
25%   -9.280191e-01 -5.742004e-01 -1.243654e+00 -1.203301e+00
50%   -8.095130e-02 -1.154728e-01  3.535005e-01  1.206904e-01
75%    6.451068e-01  5.726185e-01  7.527893e-01  7.826860e-01
max    2.460252e+00  3.095620e+00  1.779532e+00  1.709480e+00
"""

""" Codificación de Variables Categóricas: """
# Convertir las columnas dummy a tipo entero (0/1)
df_final = pd.get_dummies(df_scaled, columns=['species'])
df_final[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']] = df_final[
    ['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']].astype(int)

print("\nDataFrame con variables dummy como enteros:")
print(df_final.head())
""" 
Resultado:
DataFrame con variables dummy como enteros:
   sepal_length  sepal_width  petal_length  petal_width  species_Iris-setosa  species_Iris-versicolor  species_Iris-virginica
0     -0.928019     1.031346     -1.357737      -1.3357                    1                        0                       0
1     -1.170038    -0.115473     -1.357737      -1.3357                    1                        0                       0
2     -1.412058     0.343255     -1.414778      -1.3357                    1                        0                       0
3     -1.533067     0.113891     -1.300696      -1.3357                    1                        0                       0
4     -1.049029     1.260710     -1.357737      -1.3357                    1                        0                       0
"""

""" 
Análisis y Reflexión:

    - Valores faltantes:
        - Eliminar vs Imputar:
            - Ventaja: Fácil de implementar, elimina datos potencialmente problemáticos.
            - Desventaja: La imputación puede introducir sesgos si los valores faltantes no son aleatorios.
        - Imputar valores:
            - Ventaja: Conserva todas las observaciones, mantiene el tamaño del conjunto de datos.
            - Desventaja: La imputación puede introducir sesgos si los valores faltantes no son aleatorios.

    - Normalización de datos:
        - Es importante normalizar los datos antes de aplicar algoritmos que son sensibles a la escala de las variables,
            como el análisis de componentes principales(PCA) o métodos basados en distancias (por ejemplo,KNN, clustering)

    - Codificación de variables categóricas:
        - Los modelos de machine learning requieren variables numéricas.
        - La codificación one-hot evita asignar un orden arbitrario a las categorías y previene introducir reflexiones inexistentes.

Preguntas para reflexionar:

1. ¿Por qué es importante normalizar los datos?
    - Para garantizar que todas las variables contribuyen por igual al análisis.
    - Evita que variables con magnitudes mayores dominen sobre las de magnitudes menores.

2. ¿Cómo afecta la codificación de variables categóricas al análisis y a los modelos predictivos?
    - Permite incluir variables categóricas en modelos que solo aceptan variables numéricas.
    - Una codificación incorrecta puede introducir relaciones falsas o perder información.
"""