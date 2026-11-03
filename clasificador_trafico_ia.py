import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generar_dataset_simulado(n_muestras=10000):
    """
    Simula la recolección de datos de red (Nodo SRC / Colector según ITU-T Y.3172).
    Genera métricas de tráfico: duración de conexión, tamaño de paquete, puerto y flujo de bytes.
    """
    np.random.seed(42)
    
    # Tráfico Normal (0)
    duracion_normal = np.random.exponential(scale=10, size=int(n_muestras * 0.8))
    tamano_paquete_normal = np.random.normal(loc=500, scale=100, size=int(n_muestras * 0.8))
    tasa_bytes_normal = tamano_paquete_normal / (duracion_normal + 0.1)
    puertos_normal = np.random.choice([80, 443, 22, 53], size=int(n_muestras * 0.8))
    etiqueta_normal = np.zeros(int(n_muestras * 0.8))
    
    # Tráfico Anómalo / Ataque DDoS (1)
    duracion_anomala = np.random.exponential(scale=0.5, size=int(n_muestras * 0.2))
    tamano_paquete_anomalo = np.random.normal(loc=1400, scale=50, size=int(n_muestras * 0.2))
    tasa_bytes_anomalo = tamano_paquete_anomalo / (duracion_anomala + 0.01)
    puertos_anomalo = np.random.choice([80, 8080, 3389], size=int(n_muestras * 0.2))
    etiqueta_anomala = np.ones(int(n_muestras * 0.2))
    
    # Consolidación del conjunto de datos
    duracion = np.concatenate([duracion_normal, duracion_anomala])
    tamano_paquete = np.concatenate([tamano_paquete_normal, tamano_paquete_anomalo])
    tasa_bytes = np.concatenate([tasa_bytes_normal, tasa_bytes_anomalo])
    puertos = np.concatenate([puertos_normal, puertos_anomalo])
    etiquetas = np.concatenate([etiqueta_normal, etiqueta_anomala])
    
    df = pd.DataFrame({
        'duracion_conexion_sec': duracion,
        'tamano_paquete_bytes': tamano_paquete,
        'tasa_transferencia_Bps': tasa_bytes,
        'puerto_destino': puertos,
        'es_ataque': etiquetas
    })
    
    return df.sample(frac=1).reset_index(drop=True)

def ejecutar_pipeline_ia():
    print("=" * 60)
    print("  CLASIFICADOR DE TRÁFICO DE RED BASADO EN MACHINE LEARNING")
    print("=" * 60)
    
    # 1. Recolección de Datos
    print("\n[1] Generando dataset de tráfico de red simulado (10,000 registros)...")
    datos = generar_dataset_simulado(10000)
    
    X = datos.drop(columns=['es_ataque'])
    y = datos['es_ataque']
    
    # 2. Preprocesamiento (Nodo PP según ITU-T Y.3172)
    print("[2] Preprocesando y escalando datos...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Entrenamiento del Modelo (Nodo M - Random Forest)
    print("[3] Entrenando modelo de Inteligencia Artificial (Random Forest)...")
    inicio_entrenamiento = time.time()
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_scaled, y_train)
    
    tiempo_entrenamiento = time.time() - inicio_entrenamiento
    print(f"    --> Entrenamiento completado en: {tiempo_entrenamiento:.4f} segundos")
    
    # 4. Evaluación de Rendimiento y Predicción
    print("\n[4] Evaluando rendimiento en tráfico de prueba...")
    inicio_prediccion = time.time()
    
    y_pred = modelo.predict(X_test_scaled)
    
    tiempo_prediccion = time.time() - inicio_prediccion
    precision = accuracy_score(y_test, y_pred)
    
    # 5. Generación de Métricas Cuantitativas para el Informe
    print("\n" + "=" * 60)
    print("             RESULTADOS DE LA EVALUACIÓN CUALITATIVA")
    print("=" * 60)
    print(f"Muestras de evaluación (Test Set) : {len(y_test)} paquetes")
    print(f"Precisión Global del Modelo       : {precision * 100:.2f}%")
    print(f"Tiempo de Inferencia / Predicción : {tiempo_prediccion:.4f} segundos")
    print(f"Latencia promedio por paquete     : {(tiempo_prediccion / len(y_test)) * 1000:.6f} ms")
    
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [Tráfico Normal Correcto  : {cm[0][0]}] | [Falsos Positivos: {cm[0][1]}]")
    print(f"  [Falsas Alarmas (Misses)  : {cm[1][0]}] | [Ataques Detectados: {cm[1][1]}]")
    
    print("\nImportancia de Atributos de Red (XAI / Interpretabilidad):")
    importancias = modelo.feature_importances_
    for col, imp in zip(X.columns, importancias):
        print(f"  - {col:<25}: {imp * 100:.2f}% de influencia")
    print("=" * 60)

if __name__ == "__main__":
    ejecutar_pipeline_ia()