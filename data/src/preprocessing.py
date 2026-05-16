import pandas as pd
from sklearn.model_selection import train_test_split

def validate_data(df):
    # Regla 1: No debe haber nulos
    assert df.isnull().sum().sum() == 0, "Error: Se encontraron valores nulos"
    
    # Regla 2: Edades coherentes
    assert df['Edad'].min() >= 0, "Error: Hay edades negativas"
    
    print("✅ Validación exitosa: Los datos son consistentes.")

def preprocess_data(input_path, output_path):
    # 1. Cargar (Asegúrate de que el nombre coincida con tu carpeta data/raw)
    df = pd.read_csv(input_path, sep=';')
    
    # 2. Limpiar nulos
    df['IMC'] = df['IMC'].fillna(df['IMC'].median())
    
    # 3. Transformar categorías
    cols_to_encode = ['Genero', 'Tipo_trabajo', 'Zona_residencia', 'Flag_fumador', 'Estados_civil']
    df_final = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    
    # --- PASO DE VALIDACIÓN ---
    # Llamamos a la función antes de guardar
    validate_data(df_final) 
    
    # 4. Guardar
    df_final.to_csv(output_path, index=False)
    print(f"✅ Datos validados y guardados en: {output_path}")

if __name__ == "__main__":
    preprocess_data('data/raw/Dataset_prevencion_infartos.csv', 'data/processed/data_clean.csv')
