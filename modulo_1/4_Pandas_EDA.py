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

Conclusiones sobre las relaciones:

    1. Separación de Iris-setosa:
    - La Iris-setosa está claramente separada de las otras dos especies en función de estas dos variables.
        Esto se debe a que tiene el menor sepal_length y un mayor sepal_width, creando un patrón único.

    2. Superposición entre Iris-versicolor e Iris-virginica en cuanto a sus valores de sepal_length y sepal_width, 
        pero Iris virginica tiende a tener valores más altos en sepal_length.

    3. Posible relación inversa débil entre las variables:
        - En el caso de Iris-setosa, parece haber una leve relación inversa entre sepal_length y sepal_width,
            es decir, a mayor largo menor ancho.
        - Para las otras especies, la relación no es tan evidente y parece haber una mayor dispersión  entre los datos.


     - ¿Qué te indican las estadísticas descriptivas sobre cada característica?
"""

# Estadísticas descriptivas generales
stats_descriptive = df.describe()

# Estadísticas descriptivas por especie
stats_by_species = df.groupby('species').describe()

# Mostrar estadísticas descriptivas generales
print("Estadísticas descriptivas generales:\n")
print(stats_descriptive)

# Mostrar estadísticas descriptivas por especie
print("\nEstadísticas descriptivas por especie:\n")
print(stats_by_species)

# Guardar las estadísticas descriptivas en un archivo CSV (opcional)
stats_descriptive.to_csv('estadisticas_generales.csv')
stats_by_species.to_csv('estadisticas_por_especie.csv')

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

1. sepal_length (Largo del sépalo):
	•	Rango (min - max): 4.3 a 7.9 cm.
	•	Indica que hay una diferencia significativa en el largo del sépalo entre las especies.
	•	Media: 5.84 cm.
	•	Valor promedio, que cae entre los valores típicos de Iris-versicolor y Iris-virginica.
	•	Desviación estándar (std): 0.83 cm.
	•	Muestra una variabilidad moderada en los datos, sugiriendo que las especies difieren lo suficiente
        en esta característica para separarlas.
	•	Distribución:
	•	La mayoría de los valores se encuentran entre el primer (5.1 cm) y tercer cuartil (6.4 cm), 
        lo que sugiere que el largo del sépalo es una característica importante para distinguir especies.

2. sepal_width (Ancho del sépalo):
	•	Rango (min - max): 2.0 a 4.4 cm.
	•	Indica una menor variabilidad comparada con sepal_length.
	•	Media: 3.05 cm.
	•	Cerca del valor intermedio entre las tres especies.
	•	Desviación estándar (std): 0.43 cm.
	•	Baja dispersión, lo que implica que el ancho del sépalo tiene menos variabilidad entre especies.
	•	Distribución:
	•	Iris-setosa tiende a tener valores más altos de sepal_width, 
        mientras que las otras especies tienen valores más pequeños y similares.

3. petal_length (Largo del pétalo):
	•	Rango (min - max): 1.0 a 6.9 cm.
	•	Muestra una gran diferencia entre especies.
	•	Media: 3.76 cm.
	•	Los valores promedio son más representativos de Iris-versicolor e Iris-virginica.
	•	Desviación estándar (std): 1.76 cm.
	•	Alta variabilidad, lo que indica que esta característica es una de las más útiles para diferenciar las especies.
	•	Distribución:
	•	Iris-setosa tiene pétalos significativamente más cortos (alrededor de 1.0 a 1.5 cm), 
        lo que la separa completamente de las otras dos especies.

4. petal_width (Ancho del pétalo):
	•	Rango (min - max): 0.1 a 2.5 cm.
	•	Similar a petal_length, esta característica muestra una amplia variación entre especies.
	•	Media: 1.19 cm.
	•	Valores cercanos a los de Iris-versicolor.
	•	Desviación estándar (std): 0.76 cm.
	•	Alta dispersión, lo que sugiere que esta característica también es útil para diferenciar especies.
	•	Distribución:
	•	Iris-setosa tiene valores consistentemente bajos (cercanos a 0.1 cm), lo que la distingue de las otras especies.

Conclusiones generales:

	1.	Características más diferenciadoras:
	    •	petal_length y petal_width tienen rangos amplios y altas desviaciones estándar, 
            lo que las convierte en las características más útiles para distinguir entre especies.
	    •	Estas dos características separan claramente a Iris-setosa del resto y permiten 
            diferenciar parcialmente entre Iris-versicolor e Iris-virginica.

	2.	Características menos diferenciadoras:
	    •	sepal_length y sepal_width muestran menos variabilidad y superposición entre especies, 
            aunque siguen siendo útiles para identificar Iris-setosa.

"""