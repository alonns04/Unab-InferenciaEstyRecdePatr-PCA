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

#pca2 = PCA(n_components = 2, random_state = 42)

#pca_2 = pca2.fit_transform(df[X_cols])

#print(pca_2[:4])

#df_2 = pd.DataFrame({'PCA1': pca_2[:,0], 'PCA2': pca_2[:,1], 'clase': df['clase']})

#print(df_2.head())

#print(pca2.explained_variance_ratio_)

#print(pca2.explained_variance_ratio_.sum())


pca3 = PCA(n_components = 3, random_state = 42)

pca_3 = pca3.fit_transform(df[X_cols])

df_3 = pd.DataFrame({'PCA1': pca_3[:,0], 'PCA2': pca_3[:,1], 'PCA3': pca_3[:,2], 'clase': df['clase']})

print(df_3.head())

print(pca3.explained_variance_ratio_)

print(pca3.explained_variance_ratio_.sum())

sns.barplot(x=['PCA1', 'PCA2', 'PCA3'], y = pca3.explained_variance_ratio_)

#sns.scatterplot(x='PCA1', y = 'PCA2', hue='clase', data=df_2)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = df_3['PCA1']
y = df_3['PCA2']
z = df_3['PCA3']

col = df_3['clase'].map(
{'Iris-setosa':'r',
'Iris-versicolor': 'g',
'Iris-virginica':'b'})

ax.scatter(x,y,z, c=col, marker = 'o')

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')

plt.show()
