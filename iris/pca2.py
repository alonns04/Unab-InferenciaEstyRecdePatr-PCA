import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('iris.data', header=None)

# Define column names
nombres_col = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo', 'clase']
df.columns = nombres_col

# Print the first few rows
print(df.head())

# Select only the numeric columns for scaling
X_cols = ['longitud_sepalo', 'ancho_sepalo', 'longitud_petalo', 'ancho_petalo']

# Standardize the numeric columns
ss = StandardScaler()
df[X_cols] = ss.fit_transform(df[X_cols])

# Print the scaled data
print(df.head())

pca2 = PCA(n_components = 2, random_state = 42)

pca_2 = pca2.fit_transform(df[X_cols])

print(pca_2[:4])

df_2 = pd.DataFrame({'PCA1': pca_2[:,0], 'PCA2': pca_2[:,1], 'clase': df['clase']})

print(df_2.head())

print(pca2.explained_variance_ratio_)

print(pca2.explained_variance_ratio_.sum())


sns.barplot(x=['PCA1', 'PCA2', ], y = pca2.explained_variance_ratio_)

plt.show()

sns.scatterplot(x='PCA1', y = 'PCA2', hue='clase', data=df_2)

plt.show()
