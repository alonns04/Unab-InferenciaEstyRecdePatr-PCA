"""
Prueba hecha 100% con chatgpt.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Paso 1: Cargar el archivo CSV
file_path = 'diabetes.csv'  # Asegúrate de que la ruta al archivo sea correcta          file_path = 'excel.xlsx'                Para usar un excel
df = pd.read_csv(file_path) #                                                           df = pd.read_excel(file_path)

# Paso 2: Normalizar los datos (restar la media y dividir por la desviación estándar)
df_normalized = (df - df.mean()) / df.std()

# Paso 3: Aplicar PCA
pca = PCA()
pca.fit(df_normalized)

# Obtener las componentes principales
componentes_principales = pca.transform(df_normalized)

# Convertir las componentes principales a un DataFrame para facilitar su manipulación
df_pca = pd.DataFrame(componentes_principales, columns=[f'PC{i+1}' for i in range(df_normalized.shape[1])])

# Paso 4: Varianza explicada y porcentaje acumulado
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

# Mostrar los resultados
for i, (ve, va) in enumerate(zip(varianza_explicada, varianza_acumulada)):
    print(f'PC{i+1}: Varianza Explicada = {ve:.4f}, Varianza Acumulada = {va:.4f}')

# Paso 5: Visualización de la varianza explicada
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.5, align='center', label='Varianza Explicada Individual')
plt.step(range(1, len(varianza_acumulada) + 1), varianza_acumulada, where='mid', label='Varianza Acumulada')
plt.ylabel('Porcentaje de Varianza Explicada')
plt.xlabel('Componentes Principales')
plt.legend(loc='best')
plt.show()
