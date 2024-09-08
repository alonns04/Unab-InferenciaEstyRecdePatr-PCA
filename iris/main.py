import pandas as pd
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('iris.data', header=None)

nombres_col = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'clase']

df.columns = nombres_col

X_cols = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'clase']

ss = StandardScaler()

df[X_cols] = ss.fit_transform(df[X_cols])

df.head()