from main import varianza_explicada, varianza_acumulada, valores_propios, vectores_propios, excel_pca, excel_datos_transformados, excel_matriz_covarianza, grafico


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

excel_pca()

datos = excel_datos_transformados()

excel_matriz_covarianza()

grafico(datos, 1, 2, 3)