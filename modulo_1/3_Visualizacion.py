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

# Crear el histograma
plt. hist(calificaciones, bins=5, edgecolor='Black') # Definir 5 intervalos (bins)

# Añadir etiquetas y título
plt.title('Distribución de Calificaciones del examen') # Título del gráfico
plt.xlabel('Intervalos de Calificaciones') # Etiqueta del eje x
plt.ylabel('Frecuencia') # Etiqueta del eje y

# Mostrar el gráfico
plt.show()


""" 
b) Análisis del Histograma:
    - Describe la forma de la distribución que observas en el histograma
    - ¿La distribución es simétrica, tiene sesgo a la derecha o a la izquierda?
    - ¿Hay presencia de valores atípicos?
"""

