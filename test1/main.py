import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar el archivo Excel
file_path = 'diabetes.csv'  # Ruta relativa al archivo .py

df = pd.read_csv(file_path)

df = df.drop(df.columns[-1], axis=1)  # Excluir la última columna


nombres = df.columns.tolist() # Almacenamos los nombres de cada columna

# Resto la media y divido por la desviación estandar a cada uno de los datos:
for column in df.columns: # Para cada columna en todas las columnas que existen
    df[column] = ((df[column] - df[column].mean())) / df[column].std()

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

def excel_matriz_covarianza():
    # Crear el DataFrame sin nombres de columnas
    mc = pd.DataFrame(matriz_cov)
    print("matriz de covarianza", matriz_cov)
    
    # Guardar el DataFrame en un archivo Excel sin encabezado
    output_file = 'COV.xlsx'
    mc.to_excel(output_file, index=False, header=False)


def excel_pca():
    # Crear nombres de las columnas
    columns = [f'PCA{i+1}' for i in range(len(vectores_propios[0]))]
    
    # Crear el DataFrame con los nombres de las columnas
    df = pd.DataFrame(vectores_propios, columns=columns)
    
    # Guardar el DataFrame en un archivo Excel con encabezado
    output_file = 'PCA.xlsx'
    df.to_excel(output_file, index=False, header=True)

def excel_datos_transformados():
    datos = df.values
    datos_transformados = np.dot(datos, vectores_propios)

    df_transformados = pd.DataFrame(datos_transformados)

    output_file = 'datos_transformados.xlsx'
    df_transformados.to_excel(output_file, index=False, header=False)


    # Leer el archivo CSV, omitiendo la primera fila
    csv_data = pd.read_csv('diabetes.csv', skiprows=1, header=None)

    # Extraer la última columna del CSV
    last_column = csv_data.iloc[:, -1]  # Selecciona la última columna

    # Leer el archivo Excel existente sin encabezado
    excel_data = pd.read_excel('datos_transformados.xlsx', header=None)

    # Agregar la última columna del CSV al final del DataFrame del Excel
    excel_data['Diabetes'] = last_column.values

    # Guardar el DataFrame modificado en el archivo Excel, sin encabezado
    excel_data.to_excel('datos_transformados.xlsx', index=False, header=False)
    excel_data.to_excel('datos_transformados.xlsx', index=False)

    return excel_data

# Transformar los datos usando los vectores propios ordenados


def grafico(datos, *columnas):
    # Verificar que las columnas estén entre 1 y 8
    for col in columnas:
        if col < 1 or col > 8:
            raise ValueError("Los parámetros deben estar entre 1 y 8.")
    
    # Convertir columnas para trabajar con índices 0-based
    columnas = [col - 1 for col in columnas]
    
    # Extraer las columnas seleccionadas
    x = datos.iloc[:, columnas[0]]
    y = datos.iloc[:, columnas[1]]
    diabetes = datos['Diabetes']

    # Si hay 2 columnas, gráfico en 2D
    if len(columnas) == 2:
        plt.figure()
        plt.grid()
        for valor in [0, 1]:
            plt.scatter(x[diabetes == valor], y[diabetes == valor], 
                        label=f"Diabetes {valor}", 
                        color='red' if valor == 1 else 'blue')
        plt.xlabel(f'PCA{columnas[0] + 1}')
        plt.ylabel(f'PCA{columnas[1] + 1}')
        plt.legend()
        plt.show()

    # Si hay 3 columnas, gráfico en 3D
    elif len(columnas) == 3:
        z = datos.iloc[:, columnas[2]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for valor in [0, 1]:
            ax.scatter(x[diabetes == valor], y[diabetes == valor], z[diabetes == valor],
                        label=f"Diabetes {valor}", 
                        color='red' if valor == 1 else 'blue')
        ax.set_xlabel(f'PCA{columnas[0] + 1}')
        ax.set_ylabel(f'PCA{columnas[1] + 1}')
        ax.set_zlabel(f'PCA{columnas[2] + 1}')
        plt.legend()
        plt.show()