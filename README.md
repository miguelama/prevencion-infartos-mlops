# prevencion-infartos-mlops

Proyecto: Sistema Predictivo de Riesgo de Infartos.
Alumno: Miguel Angel Muñoz Alvarado
Curso: G1-MLOps: Del Modelo al Entorno Productivo

Descripción General: Este proyecto implementa un modelo de aprendizaje automático (Machine Learning) diseñado para identificar pacientes con alto riesgo de sufrir un ataque cardíaco. 
El enfoque principal es mejorar la detección temprana (aumentar el Recall) en un conjunto de datos altamente desequilibrado.

Funcionalidades Principales:
1.	Limpieza de Datos: Identificación y corrección de valores imposibles en la variable IMC mediante la imputación por mediana. 
2.	Ingeniería de Características (Feature Engineering): Creación de nuevas variables médicas para mejorar la capacidad predictiva:
o	Índice de Fragilidad: Interacción entre Edad e IMC. 
o	Riesgo Metabólico: Combinación de hipertensión y niveles altos de glucosa. 
o	Indicador de Adulto Mayor: Segmentación por edad (mayor a 60 años). 
3.	Entrenamiento Balanceado: Uso de un algoritmo de Random Forest con pesos de clase balanceados para compensar la falta de casos de riesgo en los datos. 
4.	Optimización de Umbral: Ajuste de la sensibilidad del modelo (Umbral 0.05) para priorizar la captura de casos de riesgo sobre la precisión general. 
5.	Persistencia del Modelo: Exportación del modelo entrenado y del escalador de datos (StandardScaler) para garantizar inferencias consistentes en el futuro. 
