import pandas as pd  # Para manipulación de datos
import numpy as np   # Para cálculos numéricos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
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

# Separar las características (X) y las etiquetas (y)
X = df_final[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df_final[['species_Iris-setosa', 'species_Iris-versicolor', 'species_Iris-virginica']]

# Dividir el conjunto de datos en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar el tamaño de los conjuntos
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

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
Lección 7: Optimización del Modelo KNN

Explicación:
El rendimiento del modelo KNN puede depender de los valores de sus hiperparámetros. Específicamente,
el número de vecinos (n_neighbors) puede afectar significativamente a la presión y a la capacidad de generalización.

Técnicas de Optimización:
1. Validación Cruzada:
    - Divide los datos en varios subconjuntos.
    - Entrena el modelo en algunos subconjuntos y evalúalo en los restantes.
    - Repite el proceso para reducir la dependencia de un solo conjunto de prueba.
2. Grid Search:
    - Explora diferentes valores de los hiperparámetros para encontrar la mejor combinación.
    - Evalúa el rendimiento de cada combinación usando validación cruzada.
"""

""" 
Ejercicios:

1. Implementar Validación Cruzada:
    - Usa cross_val_score para evaluar el rendimiento del modelo KNN en diferentes valores de n_neighbors.
"""


# Valores de n_neighbors a evaluar
neighbors = range(1, 11)

# Lista para almacenar las precisiones promedio
mean_accuracies = []

for n in neighbors:
    # Crear el modelo KNN con n_neighbors = n
    knn = KNeighborsClassifier(n_neighbors=n)
    
    # Validación cruzada con 5 folds
    scores = cross_val_score(knn, X, y.values.argmax(axis=1), cv=5, scoring='accuracy')
    
    # Guardar la precisión promedio
    mean_accuracies.append(np.mean(scores))
    print(f"n_neighbors = {n}: Precisión promedio = {np.mean(scores):.2f}")

# Graficar los resultados
plt.figure(figsize=(8, 6))
plt.plot(neighbors, mean_accuracies, marker='o', linestyle='-')
plt.title('Precisión promedio vs n_neighbors (Validación Cruzada)')
plt.xlabel('Número de vecinos (n_neighbors)')
plt.ylabel('Precisión promedio')
plt.xticks(neighbors)
plt.grid(alpha=0.5)
plt.show()