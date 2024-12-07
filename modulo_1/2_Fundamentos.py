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

print("Medidas de Tendencia Central:")
print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Moda: {moda}")

print("\nMedidas de Dispersión:")
print(f"Rango: {rango}")
print(f"Varianza: {varianza}")
print(f"Desviación Estándar: {desviacion_estandar}")

"""
Resultados: 
Medidas de Tendencia Central:
Media: 85.0
Mediana: 85.0
Moda: 80

Medidas de Dispersión:
Rango: 30
Varianza: 83.33333333333333
Desviación Estándar: 9.128709291752768
"""

""" 
Ejercicio 2:
    Interpreta los resultados:
        - ¿Qué indican estas medidas sobre el rendimiento general de los estudiantes en el examen?
        - Si un estudiante adicional obtuvo una calificación de 60, ¿Cómo afectaría esto a la media y a la desviación estandar?

Interpretación de los resultados originales:
1. Media (85.0): 
    Indica que, en promedio, las calificaciones de los estudiantes están en el rango de "notable". 
    Es un buen rendimiento general.
2. Mediana (85.0):
    Refleja que la mitad de los estudiantes obtuvo calificaciones iguales o mayores a 85. 
    Esto refuerza la idea de que la mayoría obtuvo un buen rendimiento
3. Moda (80):
    El valor más frecuente es 80, lo que sugiere que varias calificaciones están ligeramente por debajo de la media.
4. Desviación estandar (9.13):
    Indica que las calificaciones tienden a desviarse alrededor de 9 puntos de la media. Esto muestra una dispersión moderada 
    entre los resultados, es decir, algunos estudiantes obtuvieron calificaciones más altas o más bajas que el promedio, 
    pero no están demasiado alejadas.

Impacto de agregar una calificación de 60

1. Efecto en la media:
    La media disminuirá porque 60 es significativamente menor que el promedio original de 85. 
    Este nuevo dato reducirá el valor promedio al incluir una calificacion baja.
2. Efecto en la desviación estándar:
    La desviación estándar aumentará porque 60 está más lejos de la media original, 
    incrementando la dispersión del conjunto de datos. 
    Esto refleja una mayor variabilidad entre las calificaciones.
"""

"""Cálculo del impacto"""
# Agregar un nuevo estudiante con calificación de 60
nueva_calificacion = 60
nuevas_calificaciones = calificaciones + [nueva_calificacion]

# Nuevas medidas
nueva_media = np.mean(nuevas_calificaciones)
nueva_desviacion_estandar = np.std(nuevas_calificaciones, ddof=1)

# Resultados
print(f"Nueva Media: {nueva_media}")
print(f"Nueva Desviación Estándar: {nueva_desviacion_estandar}")

"""
Resultados:
Nueva Media: 82.72727272727273
Nueva Desviación Estándar: 11.48120994574099

Interpretación de los nuevos resultados:
    1. Nueva media (82.73):
        La media disminuyó de 85.0 a 82.73 al agregar la calificación de 60. Esto refleja que un solo valor bajo 
        puede afectar significativamente el promedio, especialmente en conjuntos de datos peuqeños como este.
        El rendimiento general sigue siendo bueno, pero está más cercano al rango "notable bajo".

    2. Nueva desviación Estándar (11.48):
        La desviación estándar aumentó de 9.13 a 11.48, indicando que ahora no hay mayor dispersión entre las calificaciones.
        Esto ocurre porque el valor de 60 está mucho más alejado del promedio original, lo que introduce más variabilidad.

Conclusiones:
    1. Efecto de una calificación baja en la media:
        Los promedios son sensibles a valores extremos, como el 60 en este caso. Esto resalta que, en análisis de datos, 
        es importante considerar el impacto de datos atípicos al interpretar la media.

    2. Aumento de la variabilidad:
        Una calificación baja no solo reduce el promedio, sino que también aumenta la dispersión, lo que indica una mayor 
        desigualdad en los resultados.

    3. Rendimiento general:
        Aunque el promedio se redujo, el rendimiento general sigue siendo positivo, 
        ya que el 60 es una excepción y no representa a la mayoría del grupo.

"""