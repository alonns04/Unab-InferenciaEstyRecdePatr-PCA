import pandas as pd
import numpy as np
import math

# Crear el DataFrame
file_path = 'excel.xlsx'  # Ruta relativa al archivo .py
df = pd.read_excel(file_path)

# Calcular la matriz de covarianza usando Pandas
matriz = df.cov()
print("Matriz de Covarianza usando Pandas:")

I = np.identity(matriz.shape[0])

print(matriz)

matriz[1,1] = "a"

np.linalg.det(matriz)

diagonal_principal = np.diagonal(matriz)


