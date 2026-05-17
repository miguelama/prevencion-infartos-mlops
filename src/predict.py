import joblib
import pandas as pd

def make_prediction(model_path, new_data):
    # 1. Cargar el modelo entrenado
    model = joblib.load(model_path)
    
    # 2. Convertir los datos del paciente a DataFrame
    df_patient = pd.DataFrame([new_data])
    df_patient['Indice_Fragilidad'] = df_patient['Edad'] * df_patient['IMC']
    df_patient['Riesgo_Metabolico'] = ((df_patient['Flag_hipertension'] == 1) & (df_patient['Promedio_nivel_glucosa'] > 140)).astype(int)
    df_patient['Es_Adulto_Mayor'] = (df_patient['Edad'] > 60).astype(int)

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
    # Datos de un paciente ficticio siguiendo el formato exacto
    paciente_prueba = {
        'ID': 12345,
        'Edad': 67,
        'Flag_hipertension': 1,
        'Flag_problem_cardiaco': 0,
        'Promedio_nivel_glucosa': 228.6,
        'IMC': 36.6,
        'Genero_Mujer': 0,
        'Genero_Other': 0,
        'Tipo_trabajo_Empresa_privada': 1,
        'Tipo_trabajo_En_gobierno': 0,
        'Tipo_trabajo_Nunca_trabajo': 0,
        'Tipo_trabajo_cuidar_ninos': 0,
        'Zona_residencia_Urbano': 1,
        'Flag_fumador_antes_fumaba': 1,
        'Flag_fumador_fumador': 0,
        'Estados_civil_Si': 1
    }
    # Llamamos a la función
    resultado, proba = make_prediction(ruta, paciente_prueba)
    print(f"Predicción: {resultado}")
    print(f"Probabilidad: {proba}")
