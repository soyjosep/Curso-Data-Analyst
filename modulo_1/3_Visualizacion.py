import matplotlib.pyplot as plt

""" 
La visualización de datos es una parte crucial del análisis, 
ya que permite representar información de manera gráfica, 
facilitando la comprensión de patrones, tendencias y anomalías en los datos. 
Una buena visualización puede convertir datos complejos en insights accionables.

Objetivos de la visualización de datos:

    - Comunicar Información Claramente: 
        Facilitar la interpretación de datos a través de representaciones visuales.
    - Identificar patrones y tendencias:
        Detectar relaciones que no son evidentes en tablas numéricas.
    - Tomar decisiones informadas:
        Apoyar procesos de decisión basados en evidencia visual.

Tipos comunes de gráficos:

    1. Gráficos de barras:
        - Representan datos categóricos.
        - Comparan diferentes categorias o grupos.

    2. Histogramas:
        - Muestran la distribución de numéricos continuos.
        - Ayudan a visualizar la forma de la distribución (simetría, sesgo, kurtosis).

    3. Gráficos de Líneas:
        - Ideales para mostrar tendencias a lo largo del tiempo.
        - Útiles en series temporales.

    4. Gráficos de Dispersión:
        - Representan la relación entre dos variables numéricas.
        - Pueden revelar correlaciones o patrones.

    5. Diagramas de Caja y Bigotes (Boxplots):
        - Resumen distribuciones de datos a través de cuartiles.
        - Identifican valores atípicos
"""

""" 
Ejercicio 1:

a) Crea un Histograma:
    - Utilizando las calificaciones originales del examen 
    (85, 90, 75, 80, 95, 100, 70, 85, 90, 80), crea un histograma 
    que muestre la distribución de las calificaciones.

    - Instrucciones:
        - Define los intervalos (bins) apropiados.
        - Etiqueta correctamente los ejes. 
        - Usa el lenguaje de programación Python con Matplotib.
"""

# Datos de calificaciones originales
calificaciones = [85, 90, 75, 80, 95, 100, 70, 85, 90, 80]

# Crear el histograma con bins para cada valor único
bins = [70, 75, 80, 85, 90, 95, 100, 105]  # Bins ajustados para que cada valor tenga su propio intervalo
plt.hist(calificaciones, bins=bins, edgecolor='black', color='skyblue', align='left')  # Histograma
plt.title('Distribución de Calificaciones del Examen')  # Título
plt.xlabel('Calificaciones')  # Etiqueta del eje X
plt.ylabel('Frecuencia')  # Etiqueta del eje Y
plt.xticks(bins)  # Mostrar los valores de los bins en el eje X

# Mostrar el gráfico
plt.show()


""" 
b) Análisis del Histograma:
    - Describe la forma de la distribución que observas en el histograma:
        La distribución es aproximadamente simétrica, 
        con la mayor concentración de datos en los valores intermedios (80, 85 y 90). 
        Las calificaciones están bastante distribuidas a lo largo del rango, 
        con pocas diferencias significativas en las frecuencias entre intervalos.

    - ¿La distribución es simétrica, tiene sesgo a la derecha o a la izquierda?
         La distribución es simétrica. Las frecuencias a ambos lados de la media son similares, 
         lo que indica un equilibrio en las calificaciones de los estudiantes. 
         No hay un sesgo evidente hacia la derecha (positiva) 
         o hacia la izquierda (negativa).

    - ¿Hay presencia de valores atípicos?
        No se observan valores atípicos en el histograma. Todas las calificaciones (70 a 100) 
        están dentro de un rango razonable y aparecen con una frecuencia esperada. 
        No hay valores que se destaquen como extremadamente altos o bajos en comparación con el resto.
"""

