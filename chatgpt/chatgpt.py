import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Función para cargar datos y realizar PCA
def realizar_pca(file_path, n_componentes=None):
    # Cargar el archivo CSV
    df = pd.read_csv(file_path)
    
    # Normalizar los datos (restar la media y dividir por la desviación estándar)
    df_normalizado = (df - df.mean()) / df.std()
    
    # Realizar PCA
    pca = PCA(n_components=n_componentes)
    pca_resultado = pca.fit_transform(df_normalizado)
    
    # Obtener la varianza explicada por cada componente
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)
    
    # Crear un DataFrame con los componentes principales
    columnas_pca = [f"PC{i+1}" for i in range(pca_resultado.shape[1])]
    df_pca = pd.DataFrame(pca_resultado, columns=columnas_pca)
    
    # Guardar la varianza explicada en el DataFrame
    df_varianza = pd.DataFrame({
        "Componente": columnas_pca,
        "Varianza Explicada (%)": varianza_explicada * 100,
        "Varianza Acumulada (%)": varianza_acumulada * 100
    })
    
    # Guardar los resultados en un archivo Excel
    with pd.ExcelWriter('resultado_pca.xlsx') as writer:
        df_pca.to_excel(writer, sheet_name='Componentes PCA', index=False)
        df_varianza.to_excel(writer, sheet_name='Varianza Explicada', index=False)
    
    print("PCA completado y resultados guardados en 'resultado_pca.xlsx'.")

# Ruta del archivo CSV
file_path = 'diabetes.csv'  # Cambia esto con la ruta de tu archivo

# Ejecutar el algoritmo PCA
realizar_pca(file_path)
