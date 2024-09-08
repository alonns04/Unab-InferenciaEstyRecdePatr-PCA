import pandas as pd
from tabulate import tabulate

# Cargar el DataFrame
df = pd.read_csv('iris.data', header=None)
nombres_col = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'clase']
df.columns = nombres_col

# Mostrar las primeras filas
df_head = df.head()

# Mostrar el DataFrame como una tabla en la consola
print(tabulate(df, headers='keys', tablefmt='grid'))
