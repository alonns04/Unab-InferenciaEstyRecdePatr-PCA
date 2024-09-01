import pandas as pd
import numpy as np

# Cargar el archivo Excel
file_path = 'excel.xlsx'  # Ruta relativa al archivo .py
df = pd.read_excel(file_path)

# Resto la media a cada uno de los datos:
for column in df.columns: # Para cada columna en todas las columnas que existen
    df[column] = df[column] - (sum(df[column]) / len(df[column]))
"""
La columna va a ser igual a sí misma, menos el promedio de la columna
(es decir, la suma de todos los datos dividido la cantidad de datos)
"""
def matriz_covarianza(matriz):

    columnas = matriz.columns
    # Columnas de la matriz
    matriz_cov = np.zeros((len(columnas), len(columnas))) # Creamos una matriz cuadrada con la 
    #                                            misma cantidad de columnas que la matriz de valores
    n = (matriz.iloc[:, 0].shape)[0] # Cantidad de casos "n"

    for i, column1 in enumerate(matriz.columns): # Para cada columna
        for j, column2 in enumerate(matriz.columns): # Para cada columna
            resultado = (((matriz[column1] * matriz[column2])).sum()) / (n-1) # Recorremos todos los valores
            # Los multiplicamos y hacemos una sumatoria con ".sum()". A esto lo dividimos por la cantidad
            # De casos menos 1
            matriz_cov[i, j] = resultado # Agregamos el resultado en orden
    return matriz_cov # Retornamos la matriz de covarianza resultante

matriz_cov = (matriz_covarianza(df)) # Se ejecuta el algorítmo para crear la matriz de covarianza

valores_propios, vectores_propios = np.linalg.eig(matriz_cov)
"""
Con un método de Numpy se consiguen los valores propios y los vectores propios:
Básicamente la matriz de covarianza, llamémosla "A", se multiplica por un 
vector "v" (Que puede interpretarse como una matriz de la cantidad de 
columnas por 1. Es decir que si hay 2 columnas es de 2x1) y se iguala a 
un escalar lambda λ, multiplicado por el mismo vector.
La ecuación sería:

A * v = λ * v

Ecuación característica:

Det( A - λ * I ) = 0                         (I = Identidad),  (Det() = Determinante)

Lo que nos queda saber es qué valores satisfacen la ecuación. Dichos valores de lambda
se reemplazan en la primera ecuación. Aunque primero se la iguala a 0:

(A - λ * I) * v = 0

Utilizamos los valores de lambda que encontramos y despejamos el vector "v", el cual resulta ser 
un vector unitario.

Luego lo normalizamos, dividiendo sus valores por su norma (la norma del vector: sumamos el cuadrado de sus valores
y al total le aplicamos raíz cuadrada), para generar un vector propio unitario.

"""

orden = np.argsort(valores_propios)[::-1] # Ordenar índices de mayor a menor valor propio
valores_propios = valores_propios[orden] # Se acomodan los valores
vectores_propios = vectores_propios[:, orden] # Lo mismo con los vectores

# Calcular la varianza explicada
varianza_total = np.sum(valores_propios) # La varianza total es la sumatoria de los valores propios

varianza_explicada = valores_propios / varianza_total

varianza_acumulada = np.cumsum(varianza_explicada)



# Transformar los datos
def transformar_datos(df, vectores_propios):
    # Convertir el DataFrame a una matriz NumPy
    datos = df.values
    # Multiplicar los datos por los vectores propios
    return np.dot(datos, vectores_propios)

# Transformar los datos usando los vectores propios ordenados
datos_transformados = transformar_datos(df, vectores_propios)
