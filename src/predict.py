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
    
    # Definimos los parámetros
    ruta = 'models/heart_attack_model.pkl'
    paciente_prueba = {
        'Edad': 45,
        'IMC': 26.3,
        'Flag_hipertension': 0,      # 1 para Sí, 0 para No
        'Flag_corazon': 0,           # Asumiendo que esta es otra columna numérica/binaria
        'Promedio_glucosa': 85.5,    # Ajusta según tus columnas reales
        
        # Columnas generadas por get_dummies (Categoría_Valor)
        'Genero_Masculino': 1, 
        'Tipo_trabajo_Nunca_trabajo': 0,
        'Tipo_trabajo_Privado': 1,
        'Tipo_trabajo_Self-employed': 0,
        'Tipo_trabajo_children': 0,
        'Zona_residencia_Urbana': 1,
        'Flag_fumador_antes_fumaba': 0,
        'Flag_fumador_fumador': 1,
        'Estados_civil_Si': 1
    }
    # Llamamos a la función
    resultado, proba = make_prediction(ruta, paciente_prueba)
    
    print(f"Predicción: {resultado}")
    print(f"Probabilidad: {proba}")
