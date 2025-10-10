import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
# Cargar datos
df = pd.read_csv("naive-bayes-classifier/data/dataset.csv")

# Separar características y etiquetas
X = df["text"]
y = df["label"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("===== Resumen de partición =====")
print(f"Total ejemplos: {len(df)}")
print(f"Cantidad en entrenamiento: {len(X_train)}")
print(f"Cantidad en prueba: {len(X_test)}")
print()

train_df = pd.DataFrame({"text": X_train, "label": y_train})
print("----- Ejemplos de ENTRENAMIENTO (index, text, label) -----")
print(train_df.reset_index().to_string(index=False))
print()

test_df = pd.DataFrame({"text": X_test, "label": y_test})
print("----- Ejemplos de PRUEBA (index, text, label) -----")
print(test_df.reset_index().to_string(index=False))
print()
# Crear pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Entrenar
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

results_df = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "predicted_label": y_pred
})
print("===== Resultados por ejemplo (prueba) =====")
print(results_df.reset_index().to_string(index=False))
print()

try:
    labels = list(model.named_steps["multinomialnb"].classes_)
except Exception:
    labels = sorted(list(set(y_test) | set(y_pred)))

cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
print("===== Matriz de confusión =====")
print(cm_df.to_string())
print()

print("===== Classification Report =====")
print(classification_report(y_test, y_pred))
print()

# Para que las pruebas puedan importar
__all__ = ["model", "y_pred", "y_test"]

# Para que las pruebas puedan importar
__all__ = ["model", "y_pred", "y_test"]
