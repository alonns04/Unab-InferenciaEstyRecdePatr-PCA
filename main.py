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

matriz_cov = (matriz_covarianza(df))

valores_propios, vectores_propios = np.linalg.eig(matriz_cov)
    
orden = np.argsort(valores_propios)[::-1] # Ordenar índices de mayor a menor valor propio
valores_propios = valores_propios[orden]
vectores_propios = vectores_propios[:, orden]

# Calcular la varianza explicada
varianza_total = np.sum(valores_propios)
print(varianza_total)
varianza_explicada = valores_propios / varianza_total
print(varianza_explicada)
varianza_acumulada = np.cumsum(varianza_explicada)

print("Varianza explicada por cada componente:")
for i, var in enumerate(varianza_explicada):
    print(f"Componente {i+1}: {var * 100:.2f}%")

print("\nVarianza acumulada:")
for i, var in enumerate(varianza_acumulada):
    print(f"Componentes {i+1}: {var * 100:.2f}%")


# Transformar los datos
def transformar_datos(df, vectores_propios):
    # Convertir el DataFrame a una matriz NumPy
    datos = df.values
    # Multiplicar los datos por los vectores propios
    return np.dot(datos, vectores_propios)

# Transformar los datos usando los vectores propios ordenados
datos_transformados = transformar_datos(df, vectores_propios)

print("Valores propios ordenados:")
print(valores_propios)
print("Vectores propios ordenados:")
print(vectores_propios)
print("Datos transformados (proyecciones en los componentes principales):")
print(datos_transformados)