import joblib
import pandas as pd

def make_prediction(model_path, new_data):
    # 1. Cargar el modelo entrenado
    model = joblib.load(model_path)
    
    # 2. Convertir los datos del paciente a DataFrame
    df_patient = pd.DataFrame([new_data])
    
    # Nota: En un entorno real, aquí aplicaríamos las mismas 
    # transformaciones (One-Hot Encoding) que en el preprocesamiento.
    
    # 3. Realizar la predicción
    prediction = model.predict(df_patient)
    probabilidad = model.predict_proba(df_patient)
    
    return prediction[0], probabilidad[0]

if __name__ == "__main__":
    # Datos de ejemplo (deben coincidir con las columnas tras el preprocesamiento)
    # Por ahora, usaremos una estructura simplificada para probar la carga
    print("Cargando modelo y realizando prueba de inferencia...")
    # (Aquí iría la lógica para procesar un input crudo antes de predecir)
