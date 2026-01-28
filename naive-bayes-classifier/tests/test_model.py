import unittest
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import model, y_pred, y_test



class TestTextClassificationModel(unittest.TestCase):
    def test_model_training(self):
        """Verifica que el modelo se haya creado correctamente."""
        self.assertIsNotNone(model, "El modelo no fue creado correctamente.")

    def test_prediction_output(self):
        """Verifica que el número de predicciones coincida con el número de ejemplos de prueba."""
        self.assertEqual(len(y_pred), len(y_test), "El número de predicciones no coincide con el número de ejemplos de prueba.")

    def test_prediction_labels(self):
        """Verifica que las etiquetas predichas sean válidas ('positivo' o 'negativo')."""
        for label in y_pred:
            self.assertIn(label, ["positivo", "negativo"], "Etiqueta de predicción no válida.")

    def test_dataset_length(self):
        df = pd.read_csv("naive-bayes-classifier/data/dataset.csv")
        self.assertGreaterEqual(len(df), 10, "El archivo dataset.csv debe contener al menos 10 registros.")
