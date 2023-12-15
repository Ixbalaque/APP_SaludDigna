import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


# Carga de datos
data = pd.read_csv('Cardiovascular_Disease_Dataset.csv')

# Eliminar la columna llamada "Identificación del paciente"
data = data.drop("patientid", axis=1)

# Manejar valores faltantes si es necesario (por ejemplo, eliminar filas con NaN)
data = data.dropna()

# Eliminar las filas con serumcholestrol igual a 0
data = data[data['serumcholestrol'] != 0]


"""      Arbol de Decicion        """

# Separar las características (X) de la variable objetivo (y)
X = data.drop('target', axis=1)
y = data['target']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el clasificador de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

joblib.dump(clf, 'modelo_decision_tree.joblib')

# Imprimir métricas de evaluación
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)

# Calcular la precisión global utilizando accuracy_score
Precision_global = accuracy_score(y_test, y_pred)
print("Precisión Global:", Precision_global)

# Seleccionar un registro aleatorio del DataFrame
registro_aleatorio = data.sample(n=1, random_state=133)

# Extraer las características del registro seleccionado
nuevos_datos_ejemplo = registro_aleatorio.drop('target', axis=1)

# Hacer la predicción de probabilidad utilizando el modelo entrenado
probabilidades_prediccion_ejemplo = clf.predict_proba(nuevos_datos_ejemplo)

# La probabilidad de la clase positiva (enfermedad cardiovascular)
probabilidad_enfermedad_cardiovascular_ejemplo = probabilidades_prediccion_ejemplo[0, 1]

# Imprimir el registro seleccionado y la probabilidad
print("Registro seleccionado:")
print(registro_aleatorio)
print(f"La probabilidad de tener enfermedad cardiovascular es: {probabilidad_enfermedad_cardiovascular_ejemplo:.2%}")