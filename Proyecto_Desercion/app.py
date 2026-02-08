# ==============================
# STREAMLIT - RIESGO DE DESERCIÓN
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# CONFIGURACIÓN
# ==============================
st.set_page_config(
    page_title="Riesgo de Deserción Estudiantil",
    page_icon="🎓",
    layout="centered"
)
# ==============================
# STREAMLIT - RIESGO DE DESERCIÓN
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==============================
# CONFIGURACIÓN
# ==============================
st.set_page_config(
    page_title="Riesgo de Deserción Estudiantil",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Predicción de Riesgo de Deserción")
st.write("Modelo de Machine Learning con variables académicas")

# ==============================
# CARGA DE DATOS Y MODELOS
# ==============================
df = pd.read_csv("REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO_LIMPIO.csv")  # Dataset limpio y anonimizado
modelo = joblib.load("modelo_desercion.pkl")
scaler = joblib.load("scaler.pkl")
metricas = joblib.load("metricas.pkl")
matriz = joblib.load("matriz_confusion.pkl")
importancias = pd.read_csv("importancias.csv")

# ==============================
# FUNCIONES
# ==============================
def calcular_riesgo_academico(promedio, asistencia, reprobadas, total):
    promedio_norm = 1 - (promedio / 10)
    reprobadas_norm = reprobadas / total if total > 0 else 0
    asistencia_norm = 1 - (asistencia / 100)

    return (
        0.5 * promedio_norm +
        0.3 * reprobadas_norm +
        0.2 * asistencia_norm
    )


def clasificar_probabilidad(p):
    if p < 0.30:
        return "🟢 BAJO RIESGO", "green"
    elif p < 0.60:
        return "🟡 RIESGO MEDIO", "yellow"
    else:
        return "🔴 ALTO RIESGO", "red"

# ==============================
# ANÁLISIS EXPLORATORIO (EDA)
# ==============================
st.header("📊 Análisis Exploratorio de Datos")

if st.checkbox("Mostrar análisis exploratorio"):
    st.subheader("Distribución de Promedios")
    fig, ax = plt.subplots()
    ax.hist(df['promedio_estudiante'], bins=10, color="skyblue", edgecolor="black")
    ax.set_xlabel("Promedio")
    ax.set_ylabel("Cantidad de estudiantes")
    st.pyplot(fig)

    st.subheader("Distribución de Asistencia")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['asistencia_promedio'], bins=10, color="salmon", edgecolor="black")
    ax2.set_xlabel("Asistencia (%)")
    ax2.set_ylabel("Cantidad de estudiantes")
    st.pyplot(fig2)

    st.subheader("Proporción de Deserción")
    st.bar_chart(df['desercion'].value_counts(normalize=True))

    st.subheader("Estadísticas Descriptivas")
    st.write(df.describe())

# ==============================
# INPUTS DEL ESTUDIANTE
# ==============================
st.header("Datos del estudiante")
promedio = st.slider("Promedio académico", 0.0, 10.0, 7.0, 0.1)
asistencia_input = st.slider("Asistencia promedio (%)", 0, 100, 80)
total_materias = st.number_input("Total de materias cursadas", min_value=1, value=10)
materias_reprobadas = st.number_input("Materias reprobadas", min_value=0, value=2)

# Validación simple
if materias_reprobadas > total_materias:
    st.error("Las materias reprobadas no pueden ser mayores que el total de materias.")

# ==============================
# BOTÓN DE PREDICCIÓN
# ==============================
if st.button("Evaluar riesgo") and materias_reprobadas <= total_materias:
    riesgo_academico = calcular_riesgo_academico(
        promedio,
        asistencia_input,
        materias_reprobadas,
        total_materias
    )

    X = pd.DataFrame([{
        "riesgo_academico": riesgo_academico,
        "promedio_estudiante": promedio,
        "asistencia_promedio": asistencia_input
    }])

    X_scaled = scaler.transform(X)
    prob_desercion = modelo.predict_proba(X_scaled)[0][1]

    nivel, color = clasificar_probabilidad(prob_desercion)

    # ==============================
    # RESULTADOS
    # ==============================
    st.subheader("Resultado")
    st.metric(
        label="Probabilidad de deserción",
        value=f"{prob_desercion:.2%}"
    )
    st.markdown(
        f"<h3 style='color:{color}'>{nivel}</h3>",
        unsafe_allow_html=True
    )

    st.subheader("Explicación del resultado")
    st.write(f"""
    - **Riesgo académico calculado:** `{riesgo_academico:.3f}`
    - **Promedio académico:** `{promedio}`
    - **Asistencia promedio:** `{asistencia_input}%`
    - **Materias reprobadas:** `{materias_reprobadas}`
    """)

# ==============================
# EVALUACIÓN DEL MODELO
# ==============================
st.header("📈 Evaluación del Modelo")
st.subheader("Métricas")
st.table(metricas)

st.subheader("Matriz de Confusión")
st.dataframe(pd.DataFrame(
    matriz,
    columns=["Pred No Deserta", "Pred Deserta"],
    index=["Real No Deserta", "Real Deserta"]
))

# ==============================
# IMPORTANCIA DE VARIABLES
# ==============================
st.header("🧠 Variables más importantes")
fig3, ax3 = plt.subplots()
importancias.plot(kind="barh", x="variable", y="coeficiente", ax=ax3, color="skyblue")
ax3.set_xlabel("Coeficiente")
ax3.set_ylabel("Variable")
st.pyplot(fig3)
