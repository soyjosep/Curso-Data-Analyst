import pandas as pd  # Para manipulación de datos
import numpy as np   # Para cálculos numéricos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# URL del conjunto de datos Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Nombres de las columnas del conjunto de datos
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(url, header=None, names=column_names)

# Eliminar filas con valores faltantes
df_dropped = df.dropna()
removed_rows = len(df) - len(df_dropped)
print(f"\nSe eliminaron {removed_rows} filas con valores faltantes. El DataFrame ahora tiene {len(df_dropped)} filas restantes:\n")
print(df_dropped)

# Introducir filas duplicadas
df_duplicates = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
print("\nDataFrame con filas duplicadas añadidas:")
print(df_duplicates)

df_no_duplicates = df_duplicates.drop_duplicates()
print("\nDataFrame después de eliminar duplicados:")
print(df_no_duplicates)

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

""" Codificación de Variables Categóricas: """
# Convertir las columnas dummy a tipo entero (0/1)
df_final = pd.get_dummies(df_scaled, columns=['species'])
df_final[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']] = df_final[
    ['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']].astype(int)

print("\nDataFrame con variables dummy como enteros:")
print(df_final.head())


""" 
Introducción a modelos de Machine Learning

Explicación:

El machine learning permite crear modelos que identifican patrones y realizan predicciones basadas en datos.
En esta lección, aprenderás a aplicar modelos de clasificación básicos utilizando el conjunto de datos Iris.

Objetivos de la lección:
    - Dividir los datos en conjuntos de entrenamiento y prueba.
    - Entrenar un modelo de clasificación.
    - Evaluar el rendimiento del modelo utilizando métricas estándar.

Modelo: Clasificador K-Nearest Neighbors (KNN)
"""

""" 
Ejercicios:
1. Dividir el Conjunto de Datos:
    - Usa train_test_split de sklearn para dividir el conjunto de datos en 80% entrenamiento y 20% de prueba.
"""
# Separar las características (X) y las etiquetas (y)
X = df_final[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df_final[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']]

# Dividir el conjunto de datos en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar el tamaño de los conjuntos
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
""" 
Resultado:
Tamaño del conjunto de entrenamiento: 117 muestras
Tamaño del conjunto de prueba: 30 muestras
"""

"""
2. Entrenar el modelo KNN:
    - Usa KNeighborsClassifier de sklearn para entrenar un modelo con n_neighbors=3.    
"""

# Crear el modelo KNN con n_neighbors=3
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo con el conjunto de entrenamiento
knn.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = knn.predict(X_test)

# Evaluar el modelo
print("\nMatriz de confusión:")
print(confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))

"""
Explicación de los pasos:
	1.	Modelo KNN:
	    •	KNeighborsClassifier(n_neighbors=3):
	    •	Crea un clasificador basado en los 3 vecinos más cercanos.
	2.	Entrenamiento:
	    •	knn.fit(X_train, y_train):
	    •	Entrena el modelo utilizando los datos de entrenamiento (X_train para características y y_train para etiquetas).
	3.	Predicción:
	    •	knn.predict(X_test):
	    •	Predice las clases de las muestras en el conjunto de prueba.
	4.	Evaluación del modelo:
	    •	Matriz de confusión:
	    •	Compara las etiquetas reales con las predichas.
	    •	Reporte de clasificación:
	    •	Proporciona métricas como precisión, recall, F1-score para cada clase.

Resultado:

Matriz de confusión:
[[11  0  0]
 [ 0  9  1]
 [ 0  1  8]]

 Interpretación:
	•	La primera fila corresponde a Iris-setosa:
	•	11 muestras fueron clasificadas correctamente como Iris-setosa.
	•	La segunda fila corresponde a Iris-versicolor:
	•	9 muestras fueron clasificadas correctamente como Iris-versicolor, pero 1 muestra fue mal clasificada como Iris-virginica.
	•	La tercera fila corresponde a Iris-virginica:
	•	8 muestras fueron clasificadas correctamente como Iris-virginica, pero 1 muestra fue mal clasificada como Iris-versicolor.

Reporte de clasificación:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       0.90      0.90      0.90        10
 Iris-virginica       0.89      0.89      0.89         9

      micro avg       0.93      0.93      0.93        30
      macro avg       0.93      0.93      0.93        30
   weighted avg       0.93      0.93      0.93        30
    samples avg       0.93      0.93      0.93        30

Métricas clave:
	1.	Precisión (precision):
	•	Proporción de predicciones correctas sobre el total de predicciones positivas.
	•	Iris-setosa: 1.00 (todas las predicciones para esta clase fueron correctas).
	•	Iris-versicolor: 0.90.
	•	Iris-virginica: 0.89.
	2.	Recall (recall):
	•	Proporción de predicciones correctas sobre el total de muestras reales de la clase.
	•	Iris-setosa: 1.00 (todas las muestras reales fueron clasificadas correctamente).
	•	Iris-versicolor: 0.90.
	•	Iris-virginica: 0.89.
	3.	F1-score:
	•	Media armónica entre precisión y recall, balanceando ambas métricas.
	•	Muy alto para todas las clases, con valores cercanos a 1.00.
	4.	Macro promedio (macro avg):
	•	Promedio simple de precisión, recall y F1-score para todas las clases.
	•	0.93, lo que indica un rendimiento equilibrado.
	5.	Exactitud (accuracy):
	•	Proporción de predicciones correctas sobre el total de predicciones.
	•	93%, lo que demuestra un alto rendimiento general del modelo.

Conclusión:
	•	Iris-setosa: Clasificada perfectamente (sin errores).
	•	Iris-versicolor e Iris-virginica: Presentaron un ligero error de clasificación, pero con métricas de rendimiento aún muy altas.
	•	General: El modelo KNN con n_neighbors=3 está funcionando de manera efectiva con una precisión global del 93%.
"""

""" 
Interpretar los resultados:
    - Analiza qué tan bien el modelo clasifica las especies de Iris y sugiere posibles mejosas

1. Rendimiento General:
	•	Precisión global: 93%.
	•	Esto indica que el modelo clasifica correctamente el 93% de las muestras en el conjunto de prueba.
	•	Buen rendimiento en general:
	•	Las métricas como precisión, recall y F1-score son muy altas (≥ 0.89) para todas las clases, 
        lo que sugiere que el modelo funciona bien.

2. Matriz de Confusión:
    Errores observados:
	1.	Iris-setosa:
	    •	Se clasificó perfectamente con 0 errores.
	    •	Esto es común, ya que las características de Iris-setosa (como los pétalos más cortos) la separan bien de las otras especies.
	2.	Iris-versicolor:
	    •	De las 10 muestras, 1 fue clasificada incorrectamente como Iris-virginica.
	    •	Esto podría deberse a la similitud en las características entre Iris-versicolor y Iris-virginica.
	3.	Iris-virginica:
	    •	De las 9 muestras, 1 fue clasificada incorrectamente como Iris-versicolor.
	    •	Al igual que el caso anterior, esto refleja cierta superposición en las características de estas dos especies.

    Conclusión del Modelo:
	1.	Fortalezas:
	    •	Iris-setosa es clasificada perfectamente debido a sus características bien diferenciadas.
	    •	El modelo generaliza bien para todas las especies, logrando métricas muy altas.
	2.	Debilidades:
	    •	Hay una ligera confusión entre Iris-versicolor y Iris-virginica, 
        lo cual es esperado dado que estas dos especies tienen características similares.

Posibles Mejoras:
	1.	Ajustar el número de vecinos (n_neighbors):
	    •	Actualmente, estamos usando n_neighbors=3. Probar con valores más altos (como 5 o 7) 
            podría reducir la sensibilidad a pequeños grupos de datos.
	2.	Agregar más características:
	    •	Si se tuvieran datos adicionales, como otra característica relevante, 
            podrían ayudar a reducir la confusión entre Iris-versicolor e Iris-virginica.
	3.	Probar con otros modelos:
	    •	Modelos más complejos podrían manejar mejor las similitudes entre las especies. Por ejemplo:
	    •	Árboles de decisión: Separan mejor datos con relaciones no lineales.
	    •	SVM con kernel RBF: Manejan bien datos que no son linealmente separables.
	    •	Redes neuronales: Para un enfoque más avanzado.
	4.	Normalización en lugar de estandarización:
	    •	KNN utiliza distancias, por lo que probar normalización (escala entre 0 y 1) podría mejorar el rendimiento.
	5.	Validación cruzada:
	    •	Usa k-fold cross-validation para evaluar la robustez del modelo en diferentes subconjuntos de datos.
"""
