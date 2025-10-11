from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# Cargar datos
df = pd.read_csv("naive-bayes-classifier/data/dataset.csv")

# Separar características y etiquetas
# Separar caracteristicas y etiquetas
X = df["text"]
y = df["label"]

# Dividir en entrenamiento y prueba
class_counts = Counter(y)
can_stratify = all(c >= 2 for c in class_counts.values())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear pipeline
# Imprimir que tomo para Train y en TEST
train_df = pd.DataFrame({"text": X_train.values, "label": y_train.values}).reset_index(drop=True)
test_df  = pd.DataFrame({"text": X_test.values,  "label": y_test.values}).reset_index(drop=True)

print("\n RESUMEN")
print(f"Tamaño TOTAL: {len(df)} | TRAIN: {len(train_df)} | TEST: {len(test_df)}")
print("Distribución:", dict(class_counts))

print("\n Train ")
print(train_df.to_string(index=False))

print("\n TEST")
print(test_df.to_string(index=False))

# Crear pipeline 
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Entrenar
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("\n Reporte")
print(classification_report(y_test, y_pred, zero_division=0))

# Mostrar matrices de caracteristicas de Train y TEST
vectorizer = model.named_steps["countvectorizer"]

X_train_mat = vectorizer.transform(X_train) 
X_test_mat  = vectorizer.transform(X_test)  

print("\n Matrices")
print(f"Vocabulario: {len(vectorizer.get_feature_names_out())}")
print(f"X_train_mat shape: {X_train_mat.shape} (filas=train_ejemplos, columnas=caracteristicas)")
print(f"X_test_mat  shape: {X_test_mat.shape}")

# Para que las pruebas puedan importar
__all__ = ["model", "y_pred", "y_test"]
