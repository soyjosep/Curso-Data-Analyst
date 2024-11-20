import numpy as np
from scipy import stats


# Fundamentos de Estadística para el Análisis de Datos

""" 
Explicación:
    La estadística es una herramienta esencial en el análisis de datos, ya que nos permite describir, 
    resumir e interpretar grandes volúmennes de información. A continuación, repasaremos algunos conceptos fundamentales:

Medidas de Tendencia Central:
    1. Media(Promedio): Suma de todos los valores dividida entre el número total de valores.
    2. Mediana: Valor central cuando los datos están ordenados de menor a mayor.
    3. Moda: Valor o valores que aparecen con mayor frecuencia en el conjunto de datos.

Medias de Dispersión:
    1. Rango: Diferencia entre el valor máximo y el mínimo.
    2. Varianza: Promedio de las diferencias al cuadrados entre cada valor y la media; mide la dispersión de datos.
    3. Desviación Estándar: Raíz cuadrada de la varianza; indica cuánto se alejan los datos de la media en promedio.

Distribución de Frecuencias:
    - Histograma: Gráfico que representa la distribución de los datos en intervalos (bins).
    - Ayuda a visualizar cómo se distribuyen los datos y a identificar patrones como la simetría o la presencia de valores atípicos.
"""

""" 
Ejercicio 1:
    Calcula las medidas de tendencia central y dispersión:
    Tienes el siguiente conjunto de datos que representa las calificaciones de un examen:
    85, 90, 75, 80, 95, 100, 70, 85, 90, 80

        - Medida de Tendencia Central: Calcula la media, la mediana y moda.
        - Medidas de Dispersión: Calcula el rango, varianza y desviación estándar.
"""


# Conjunto de datos (calificaciones del examen)
calificaciones = [85, 90, 75, 80, 95, 100, 70, 85, 90, 80]

# Medidas de tendencia central
media = np.mean(calificaciones)  # Media (promedio)
mediana = np.median(calificaciones)  # Mediana
moda_resultado = stats.mode(calificaciones, keepdims=True)  # Evitar cambios recientes en scipy
moda = moda_resultado.mode[0]  # Acceder al valor de la moda correctamente

# Medidas de dispersión
rango = np.ptp(calificaciones)  # Rango (máximo - mínimo)
varianza = np.var(calificaciones, ddof=1)  # Varianza muestral
desviacion_estandar = np.std(calificaciones, ddof=1)  # Desviación estándar muestral

# Resultados
print(f"Medidas de Tendencia Central:")
print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Moda: {moda}")

print(f"\nMedidas de Dispersión:")
print(f"Rango: {rango}")
print(f"Varianza: {varianza}")
print(f"Desviación Estándar: {desviacion_estandar}")

""" 
Ejercicio 2:
    Interpreta los resultados:
        - ¿Qué indican estas medidas sobre el rendimiento general de los estudiantes en el examen?
        - Si un estudiante adicional obtuvo una calificación de 60, ¿Cómo afectaría esto a la media y a la desviación estandar?
"""
