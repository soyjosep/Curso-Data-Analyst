import pandas as pd  # Para manipulación de datos
import numpy as np   # Para cálculos numéricos
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from matplotlib.colors import ListedColormap


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

# Mostrar el gráfico sin bloquear la ejecución
plt.show(block=False)

"""
Mejoras Sugeridas
	1.	Optimizar la métrica de distancia:
	    Las métricas de distancia determinan cómo se mide la proximidad entre puntos. Por ejemplo:
	        •	euclidean: Distancia euclidiana (por defecto en KNN).
	        •	manhattan: Distancia Manhattan (también conocida como “taxicab”).
"""

# Grid Search para optimizar hiperparámetros
param_grid = {
    'n_neighbors': range(1, 11),
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2  # Cambia esto para controlar el nivel de detalle
)

print("Iniciando Grid Search...")
grid_search.fit(X, y.values.argmax(axis=1))
print("Grid Search completado.")

if grid_search.best_params_ and grid_search.best_score_:
    print(f"\nMejores hiperparámetros: {grid_search.best_params_}")
    print(f"Mejor puntuación de validación cruzada: {grid_search.best_score_:.2f}")
else:
    print("No se encontraron mejores hiperparámetros.")

""" 	
2.	Visualización de Fronteras de Decisión:
	•	Selecciona dos características principales (por ejemplo, sepal_length y sepal_width) y visualiza cómo el modelo clasifica el espacio de características.
"""
# Seleccionar dos características principales
features = ['sepal_length', 'sepal_width']
target = 'species'

# Convertir las etiquetas a valores numéricos
df['species'] = df['species'].astype('category').cat.codes

# Dividir los datos en características (X) y etiquetas (y)
X = df[features]
y = df[target]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=6, metric='euclidean', weights='uniform')
knn.fit(X_train, y_train)

# Crear una malla de puntos para graficar las fronteras de decisión
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predecir para cada punto de la malla
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar las fronteras de decisión
plt.figure(figsize=(10, 8))
colors = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.contourf(xx, yy, Z, alpha=0.8, cmap=colors)

# Graficar los puntos de datos
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor='k', s=20, cmap=ListedColormap(['red', 'green', 'blue']))
plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'], loc='upper left')

# Configurar títulos y etiquetas
plt.title("Fronteras de Decisión del Modelo KNN")
plt.xlabel("sepal_length (escalado)")
plt.ylabel("sepal_width (escalado)")
plt.grid(alpha=0.5)
plt.savefig("decision_boundaries.png")

""" 
Evaluación Adicional:
	•	Calcula métricas adicionales como la ROC AUC para un análisis más completo.
"""

# Binarizar las etiquetas
y_train_binarized = label_binarize(y_train.values, classes=[0, 1, 2])
y_test_binarized = label_binarize(y_test.values, classes=[0, 1, 2])

print(f"y_train_binarized: \n{y_train_binarized[:5]}")
print(f"y_test_binarized: \n{y_test_binarized[:5]}")

# Predicciones de probabilidad
y_pred_proba = knn.predict_proba(X_test)

# Calcular el ROC AUC para cada clase (macro y micro)
roc_auc_macro = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
roc_auc_micro = roc_auc_score(y_test_binarized, y_pred_proba, average='micro', multi_class='ovr')

print(f"\nROC AUC (Macro Promedio): {roc_auc_macro:.2f}")
print(f"ROC AUC (Micro Promedio): {roc_auc_micro:.2f}")

""" 
Resultado final

Se eliminaron 0 filas con valores faltantes. El DataFrame ahora tiene 150 filas restantes:

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

Datos escalados (media = 0, desviación estándar = 1):
       sepal_length   sepal_width  petal_length   petal_width
count  1.470000e+02  1.470000e+02  1.470000e+02  1.470000e+02
mean  -4.833624e-17  1.691768e-16 -2.416812e-16 -3.383537e-16
std    1.003419e+00  1.003419e+00  1.003419e+00  1.003419e+00
min   -1.883710e+00 -2.424189e+00 -1.585902e+00 -1.468099e+00
25%   -9.155095e-01 -5.873036e-01 -1.243654e+00 -1.203301e+00
50%   -6.833389e-02 -1.280822e-01  3.535005e-01  1.206904e-01
75%    6.578166e-01  5.607500e-01  7.527893e-01  7.826860e-01
max    2.473193e+00  3.086468e+00  1.779532e+00  1.709480e+00

DataFrame con variables dummy como enteros:
   sepal_length  sepal_width  petal_length  petal_width  species_Iris-setosa  species_Iris-versicolor  species_Iris-virginica
0     -0.915509     1.019971     -1.357737      -1.3357                    1                        0                       0
1     -1.157560    -0.128082     -1.357737      -1.3357                    1                        0                       0
2     -1.399610     0.331139     -1.414778      -1.3357                    1                        0                       0
3     -1.520635     0.101529     -1.300696      -1.3357                    1                        0                       0
4     -1.036535     1.249582     -1.357737      -1.3357                    1                        0                       0
Tamaño del conjunto de entrenamiento: 117 muestras
Tamaño del conjunto de prueba: 30 muestras

Matriz de confusión:
[[11  0  0]
 [ 0  9  1]
 [ 0  1  8]]

Reporte de clasificación:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       0.90      0.90      0.90        10
 Iris-virginica       0.89      0.89      0.89         9

      micro avg       0.93      0.93      0.93        30
      macro avg       0.93      0.93      0.93        30
   weighted avg       0.93      0.93      0.93        30
    samples avg       0.93      0.93      0.93        30

n_neighbors = 1: Precisión promedio = 0.95
n_neighbors = 2: Precisión promedio = 0.94
n_neighbors = 3: Precisión promedio = 0.95
n_neighbors = 4: Precisión promedio = 0.95
n_neighbors = 5: Precisión promedio = 0.96
n_neighbors = 6: Precisión promedio = 0.97
n_neighbors = 7: Precisión promedio = 0.96
n_neighbors = 8: Precisión promedio = 0.97
n_neighbors = 9: Precisión promedio = 0.97
n_neighbors = 10: Precisión promedio = 0.95
Iniciando Grid Search...
Fitting 5 folds for each of 40 candidates, totalling 200 fits
[CV] END ...metric=euclidean, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=euclidean, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=euclidean, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END .metric=euclidean, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=euclidean, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=euclidean, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=euclidean, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=euclidean, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=1, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=1, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=2, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=2, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=3, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=3, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=4, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=4, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=5, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=5, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=6, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=6, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=7, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=7, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=8, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=8, weights=distance; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ...metric=manhattan, n_neighbors=9, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=9, weights=distance; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END ..metric=manhattan, n_neighbors=10, weights=uniform; total time=   0.0s
[CV] END .metric=manhattan, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=manhattan, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=manhattan, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=manhattan, n_neighbors=10, weights=distance; total time=   0.0s
[CV] END .metric=manhattan, n_neighbors=10, weights=distance; total time=   0.0s
Grid Search completado.

Mejores hiperparámetros: {'metric': 'euclidean', 'n_neighbors': 6, 'weights': 'uniform'}
Mejor puntuación de validación cruzada: 0.97
Dimensiones de y_train: (120,)
y_train_binarized: 
[[1 0 0]
 [1 0 0]
 [0 1 0]
 [1 0 0]
 [1 0 0]]
y_test_binarized: 
[[0 1 0]
 [1 0 0]
 [0 0 1]
 [0 1 0]
 [0 1 0]]

ROC AUC (Macro Promedio): 0.94
ROC AUC (Micro Promedio): 0.96
"""