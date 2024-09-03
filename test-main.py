from main import varianza_explicada, varianza_acumulada, valores_propios, vectores_propios, datos_transformados, matriz_cov, nombres
import pandas as pd

print("Matriz de covarianza \n", matriz_cov)


print("Varianza explicada por cada componente:")
for i, var in enumerate(varianza_explicada):
    print(f"Componente {i+1}: {var * 100:.2f}%")

print("\nVarianza acumulada:")
for i, var in enumerate(varianza_acumulada):
    print(f"Componentes {i+1}: {var * 100:.2f}%")


print("Valores propios ordenados:")
print(valores_propios)
print("Vectores propios ordenados:")
print(vectores_propios)
print("Datos transformados (proyecciones en los componentes principales):")
print(datos_transformados)

df_transformados = pd.DataFrame(datos_transformados)

df_transformados.loc[-1] = nombres  # Añadir los nombres de columnas como una nueva fila
df_transformados.index = df_transformados.index + 1  # Cambiar el índice para que la nueva fila sea la primera
df_transformados.sort_index(inplace=True)  # Ordenar el índice

# Guardar en un archivo Excel
output_file = 'datos_transformados.xlsx'
df_transformados.to_excel(output_file, index=False, header=False)

print(f"Los datos se han guardado en '{output_file}'")