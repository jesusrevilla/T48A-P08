import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar datos
df = pd.read_csv("naive-bayes-classifier/data/dataset.csv")

# Separar características y etiquetas
X = df["text"]
y = df["label"]

# Dividir en entrenamiento y prueba
# Con 11 datos y test_size=0.2, tomará 3 para prueba y 8 para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- IMPRESIONES AÑADIDAS ---
# Imprimir qué datos se usaron para entrenar y cuáles para probar
print("--- 1. Datos Seleccionados ---")
print("Textos de Entrenamiento:")
print(X_train.to_string(index=False))
print("\nTextos de Prueba:")
print(X_test.to_string(index=False))
print("-" * 30)

#------------------------------
# Crear pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Entrenar
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)

# --- IMPRESIONES AÑADIDAS ---
# Imprimir las predicciones junto al texto real de prueba
print("\n--- 2. Predicciones del Modelo ---")
df_predicciones = pd.DataFrame({'Texto de Prueba': X_test, 'Etiqueta Real': y_test, 'Predicción del Modelo': y_pred})
print(df_predicciones.to_string(index=False))
print("-" * 30)

# Crear un DataFrame para mostrar cómo se usó cada dato y su predicción
df_general = df.copy()
df_general['Uso en Modelo'] = 'Entrenamiento'
df_general.loc[X_test.index, 'Uso en Modelo'] = 'Prueba'
df_general['Predicción'] = '-'
df_general.loc[X_test.index, 'Predicción'] = y_pred

print("\n--- 3. Vista General del Dataset y el Modelo ---")
print(df_general.to_string())
print("-" * 30)

# Explicación final sobre lo que predice
print("\n--- 4. ¿Qué es lo que predice? ---")
print("El modelo fue entrenado para clasificar la opinión de un texto en 'positivo', 'regular' o 'negativo'.")
# Tomamos el primer elemento de prueba como ejemplo
primer_texto_prueba = X_test.iloc[0]
primera_prediccion = y_pred[0]
print(f"Por ejemplo, para el texto: '{primer_texto_prueba}', el modelo predijo que la etiqueta es: '{primera_prediccion}'.")

#------------------------------
# Reporte de métricas
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))


# Para que las pruebas puedan importar
__all__ = ["model", "y_pred", "y_test"]
