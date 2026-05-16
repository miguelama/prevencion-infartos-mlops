import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_model(input_path, model_output_path):
    # 1. Cargar datos limpios
    df = pd.read_csv(input_path)

    # 1.5 Limpieza de valores imposibles
    # Marcamos los ceros como valores nulos (NaN) para poder calcular la mediana correctamente
    df['IMC'] = df['IMC'].replace(0, np.nan)
    
    # Calculamos la mediana de los valores que sí son válidos
    mediana_imc = df['IMC'].median()
    
    # Rellenamos los huecos con esa mediana
    df['IMC'] = df['IMC'].fillna(mediana_imc)
    
    print(f"✅ Limpieza completada. Valor de IMC corregido con la mediana: {mediana_imc:.2f}")
    
    # 2. Separar características (X) y objetivo (y)
    # Suponiendo que 'Ataque_cardiaco' es la columna a predecir
    X = df.drop('Ataque_cardiaco', axis=1)
    y = df['Ataque_cardiaco']

    # Columnas numéricas que identificamos
    cols_numericas = ['Edad', 'Promedio_nivel_glucosa', 'IMC']
    
    # Creamos el escalador
    scaler = StandardScaler()
    
    # Escalamos SOLO las numéricas en el set de entrenamiento
    X_train[cols_numericas] = scaler.fit_transform(X_train[cols_numericas])
    
    # Aplicamos la MISMA escala al set de prueba (importante: solo transform, no fit)
    X_test[cols_numericas] = scaler.transform(X_test[cols_numericas])

    # Calculamos la correlación de todas las variables con la columna objetivo
    correlaciones = df.corr()['Ataque_cardiaco'].sort_values(ascending=False)
    
    print("🔍 Correlación de las variables con el Ataque Cardíaco:")
    print(correlaciones)

    # 3. División Entrenamiento/Prueba (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Entrenamiento del modelo
    print("Iniciando entrenamiento del bosque aleatorio...")
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluación (Testing)
    # Obtenemos las probabilidades y las guardamos en la variable y_probs
    y_probs = model.predict_proba(X_test)[:, 1] 
    
    umbral = 0.05
    nuevas_predicciones = (y_probs > umbral).astype(int)
    guardar_matriz(y_test, nuevas_predicciones,umbral)
    
    reporte = classification_report(y_test, nuevas_predicciones)
    print("\n📊 Reporte de Evaluación (Umbral {umbral}):")
    print(reporte)

    # Abrir el archivo en modo "w" (write/escribir)
    with open("models/metrics.txt", "w") as f:
        f.write("=== Reporte de Metricas del Modelo ===\n")
        f.write(f"Umbral utilizado: {umbral}\n") # Es buena práctica anotar el umbral
        f.write(reporte)
        
    # 6. Guardar el modelo entrenado
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\n✅ Modelo guardado en: {model_output_path}")

def guardar_matriz(y_real, y_pred, umbral):
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sano', 'Riesgo'], 
                yticklabels=['Sano', 'Riesgo'])
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.title(f'Matriz de Confusión (Umbral {umbral})')
    plt.savefig('models/confusion_matrix.png') # Se guarda en la carpeta de artefactos
    plt.close()



if __name__ == "__main__":
    train_model('data/processed/data_clean.csv', 'models/heart_attack_model.pkl')
  
