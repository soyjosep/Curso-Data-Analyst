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

"""
2. Cargar el conjunto de Datos en un DataFrame de Pandas:
    - Importa las bibliotecas necesarias(pandas y matplotlib).
    - Lee el archivo CSV y asigna el DataFrame a una variable, por ejemplo, df.
"""
# Nombres de las columnas del conjunto de datos
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(url, header=None, names=column_names)

"""
3. Realizar un Análisis exploratorio Básico:
    - Inspección Inicial:
        - Muestra las primeras 5 filas del DataFrame utilizando df.head().
        - Obtén información general del DataFrame con df.info().
"""
# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame")
print(df.head())

# Obtener información general del DataFrame
print("\nInformación general del DataFrame:")
print(df.info())

"""
Resultado:
Primeras filas del DataFrame
   sepal_length  sepal_width  petal_length  petal_width      species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa

Información general del DataFrame:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
"""

"""
    - Estadísticas Descriptivas:
        - Utiliza df.describe() para calcular estadísticas como media, mediana y desviación estándar 
        de cada característica numérica.
"""
# Estadísticas descriptivas con df.describe()
print("Estadísticas Descriptivas de las características numéricas (con describe):")
print(df.describe())
"""
Resultado:
Estadísticas Descriptivas de las características numéricas (con describe):
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
"""

"""
    - Visualización:
        - Crea un gráfico de dispersión (scatter plot) de sepal_length vs sepal_width.
        - Distingue las diferentes especies utilizando colores o marcadores distintos.
        - Añade etiquetas y título al gráfico para una mejor comprensión.
"""

# Crear el gráfico de dispersión con etiquetas y título mejorado
plt.figure(figsize=(10, 8))
species_unique = df['species'].unique()
colors = ['blue', 'green', 'orange']

for specie, color in zip(species_unique, colors):
    subset = df[df['species'] == specie]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], label=specie, c=color, alpha=0.7)

# Título y etiquetas
plt.title('Relación entre Largo y Ancho del Sépalo Diferenciada por Especie', fontsize=16)
plt.xlabel('Largo del Sépalo (cm)', fontsize=14)
plt.ylabel('Ancho del Sépalo (cm)', fontsize=14)
plt.legend(title='Especies', fontsize=12, title_fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

"""
Instrucciones adicionales:
    - Análisis: Después de crear el gráfico, analiza si hay patrones o relaciones entre las variables y las especies.
        Observaciones generales:
            1. Iris-setosa(azul):
                - Tiende a tener valores más altos de sepal_width en comparación con las otras especies.
                - Los valores de sepal_length son consistentemente más bajos (concentrados entre 4.5 y 55cm).
                - En el gráfico, los puntos azules están claramente agrupados en una región específica del espacio, 
                    lo que sugiere que esta especie es fácilmente diferenciable basándose en estas dos variables.

            2. Iris-versicolor(verde):
                - Tiene valores intermedios de sepal_length (entre 5.0cm y 6.5cm).
                - Los valores de sepal_width son más estrechos y oscilan en un rango menor (entre 2.5cm y 3.5cm).
                - Muestra cierta superposición con las otras dos especies, pero sigue mostrando una separación moderada.

            3. Iris virginica (naranja):
                - Tiende a tener los valores más altos de sepal_length, alcazando hasta 7.5cm.
                - Los valores de sepal_width están en un rango similar al de Ìris-versicolor (entre 2.5cm y 3.5cm),
                pero ligeramente más distribuidos hacia valores mayores.

                XXXXXXXXXXXXXXXXXXXXXXXX VAMOS POR AQUI

    - Preguntas para Reflexionar:
        - ¿Puedes identificar agrupaciones de especies en el gráfico?
        - ¿Qué te indican las estadísticas descriptivas sobre cada característica?
"""