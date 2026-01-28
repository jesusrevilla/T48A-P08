import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Cargar datos
df = pd.read_csv("naive-bayes-classifier/data/dataset.csv", quotechar='"')

# Eliminar filas sin etiqueta
df = df.dropna(subset=["label"])

# Separar características y etiquetas
X = df["text"]
y = df["label"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar info de partición
print("===== Resumen de partición =====")
print(f"Total ejemplos: {len(df)}")
print(f"Cantidad en entrenamiento: {len(X_train)}")
print(f"Cantidad en prueba: {len(X_test)}\n")

# Ejemplos
print("----- Ejemplos de ENTRENAMIENTO (index, text, label) -----")
print(pd.DataFrame({"text": X_train, "label": y_train}).reset_index().to_string(index=False), "\n")

print("----- Ejemplos de PRUEBA (index, text, label) -----")
print(pd.DataFrame({"text": X_test, "label": y_test}).reset_index().to_string(index=False), "\n")

# Crear y entrenar el modelo
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Resultados
results_df = pd.DataFrame({"text": X_test.values, "true_label": y_test.values, "predicted_label": y_pred})
print("===== Resultados por ejemplo (prueba) =====")
print(results_df.reset_index().to_string(index=False), "\n")

# Matriz de confusión
labels = sorted(list(set(y_test) | set(y_pred)))
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("===== Matriz de confusión =====")
print(pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels]).to_string(), "\n")

# Classification report
print("===== Classification Report =====")
print(classification_report(y_test, y_pred))

# Para las pruebas
__all__ = ["model", "y_pred", "y_test"]
