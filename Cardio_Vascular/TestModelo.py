import joblib
import os
import joblib

# Especifica la ruta completa al archivo del modelo
modelo_path = 'modelo_decision_tree.joblib'

# Carga el modelo
loaded_model = joblib.load(modelo_path)

if os.path.exists(modelo_path):
    loaded_model = joblib.load(modelo_path)
    # Continúa con el uso del modelo...
else:
    print(f"El archivo {modelo_path} no existe.")
