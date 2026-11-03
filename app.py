import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Clasificador de Tráfico de Red - IA", layout="wide")

st.title("🛡️ Clasificador de Tráfico de Red basado en IA")
st.markdown("Interfaz interactiva para evaluar el rendimiento del modelo de Machine Learning (ITU-T Y.3172).")

# Controles en la barra lateral
st.sidebar.header("⚙️ Configuración del Dataset")
n_registros = st.sidebar.slider(
    "Cantidad de registros a simular",
    min_value=1000,
    max_value=500000,  # Límite elevado a medio millón
    value=50000,
    step=25000,
    help="Seleccioná cuántos paquetes de red querés generar para el entrenamiento y prueba."
)

@st.cache_data
def cargar_datos_y_modelo(n):
    np.random.seed(42)
    
    duracion = np.concatenate([np.random.exponential(10, int(n*0.8)), np.random.exponential(0.5, int(n*0.2))])
    tamano = np.concatenate([np.random.normal(500, 100, int(n*0.8)), np.random.normal(1400, 50, int(n*0.2))])
    tasa = tamano / (duracion + 0.1)
    puertos = np.concatenate([np.random.choice([80, 443, 22, 53], int(n*0.8)), np.random.choice([80, 8080, 3389], int(n*0.2))])
    etiquetas = np.concatenate([np.zeros(int(n*0.8)), np.ones(int(n*0.2))])
    
    df = pd.DataFrame({
        'duracion_conexion_sec': duracion,
        'tamano_paquete_bytes': tamano,
        'tasa_transferencia_Bps': tasa,
        'puerto_destino': puertos,
        'es_ataque': etiquetas
    }).sample(frac=1).reset_index(drop=True)
    
    X = df.drop(columns=['es_ataque'])
    y = df['es_ataque']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_s, y_train)
    
    return modelo, scaler, X_test, y_test, X_test_s, df

with st.spinner(f"Generando y entrenando con {n_registros} registros..."):
    modelo, scaler, X_test, y_test, X_test_s, df_completo = cargar_datos_y_modelo(n_registros)

if st.button("🚀 Ejecutar Inferencia y Evaluar Métricas"):
    inicio = time.time()
    y_pred = modelo.predict(X_test_s)
    y_prob = modelo.predict_proba(X_test_s)[:, 1] # Probabilidades para la Curva ROC
    tiempo = time.time() - inicio
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Reporte de clasificación detallado
    reporte_dict = classification_report(y_test, y_pred, output_dict=True)
    
    st.success(f"¡Evaluación completada con éxito sobre {len(y_test)} muestras de prueba!")
    
    # Tarjetas de métricas principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Precisión Global", f"{acc*100:.2f}%")
    col2.metric("Tiempo de Inferencia", f"{tiempo:.4f} s")
    col3.metric("Latencia Promedio", f"{(tiempo/len(y_test))*1000:.4f} ms")
    
    # --- NUEVA SECCIÓN: Distribución del Tráfico Generado ---
    st.subheader("📊 Distribución del Tráfico en el Dataset")
    conteo_trafico = df_completo['es_ataque'].value_counts().reset_index()
    conteo_trafico.columns = ['Tipo', 'Cantidad']
    conteo_trafico['Tipo'] = conteo_trafico['Tipo'].map({0.0: 'Tráfico Normal', 1.0: 'Ataque'})
    
    fig_pie = px.pie(conteo_trafico, names='Tipo', values='Cantidad', 
                     title="Proporción de Tráfico Normal vs. Ataques simulados",
                     hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Sección de métricas detalladas por clase
    st.subheader("📋 Métricas Detalladas por Clase (Precision, Recall, F1-Score)")
    df_reporte = pd.DataFrame(reporte_dict).transpose()
    
    df_reporte = df_reporte.rename(index={
        '0.0': 'Tráfico Normal (0)', 
        '1.0': 'Ataque (1)', 
        '0': 'Tráfico Normal (0)', 
        '1': 'Ataque (1)', 
        'accuracy': 'Precisión Global (Accuracy)', 
        'macro avg': 'Promedio Macro', 
        'weighted avg': 'Promedio Ponderado'
    })
    
    st.dataframe(df_reporte.style.format("{:.4f}"), use_container_width=True)
    
    st.subheader("📉 Matriz de Confusión")
    df_cm = pd.DataFrame(cm, index=["Normal (Real)", "Ataque (Real)"], columns=["Predicho Normal", "Predicho Ataque"])
    st.dataframe(df_cm, use_container_width=True)
    
    # --- NUEVA SECCIÓN: Curva ROC y AUC ---
    st.subheader("📈 Curva ROC y Métrica AUC (Área Bajo la Curva)")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    df_roc = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
    fig_roc = px.line(df_roc, x='False Positive Rate', y='True Positive Rate',
                      title=f"Curva ROC (AUC = {roc_auc:.4f})",
                      labels={'False Positive Rate': 'Tasa de Falsos Positivos (FPR)', 'True Positive Rate': 'Tasa de Verdaderos Positivos (TPR)'})
    # Línea diagonal de referencia aleatoria
    fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.subheader("🔍 Importancia de Atributos (XAI)")
    importancias = pd.DataFrame({
        'Atributo': X_test.columns,
        'Importancia (%)': modelo.feature_importances_ * 100
    }).sort_values(by='Importancia (%)', ascending=True)
    
    # Gráfico interactivo con Plotly
    fig = px.bar(importancias, x='Importancia (%)', y='Atributo', orientation='h',
                 title="Peso relativo de cada variable en la decisión del modelo",
                 color='Importancia (%)', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)