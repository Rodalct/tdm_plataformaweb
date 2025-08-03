import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re

# ——— Configuración de página y CSS ———
st.set_page_config(page_title="Evaluación Salud Mental", layout="centered")
st.markdown("""
<style>
    .stApp { background-color: #F7F3FB; }
    .titulo { font-size:36px; color:#4B0082; text-align:center; }
    .card { background:#ffffffdd; border-radius:12px; padding:20px; margin-bottom:20px;
            box-shadow:0 4px 8px rgba(0,0,0,0.1); }
    input[type="range"] { accent-color:#4B0082; }
    .stRadio label { color:#4B0082; }
    .stSelectbox>div[role="listbox"] { background-color:#f0f0ff; border-radius:8px; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="titulo">🧠 Plataforma de Salud Mental</div>', unsafe_allow_html=True)

# ——— Carga modelo ———
@st.cache_resource
def load_model():
    return joblib.load("ML_Stacking.pkl")
model = load_model()

# ——— Variables esperadas y DATA ———
expected_cols = [
    'id','school_year','age','gender','bmi','who_bmi','phq_score',
    'depression_severity','depressiveness','suicidal','depression_treatment',
    'gad_score','anxiety_severity','anxiousness','anxiety_diagnosis',
    'anxiety_treatment','epworth_score','sleepiness'
]
DEFAULT_ID = 9999

# ——— Crear pestañas ———
tab1, tab2 = st.tabs(["📝 Evaluación", "📘 Información"])

with tab1:
    # Inicializar data
    data = {'id': DEFAULT_ID}

    # 1. Año escolar
    st.markdown("<div class='card'><h4>1. ¿En qué año de universidad estás?</h4></div>", unsafe_allow_html=True)
    data['school_year'] = st.slider("Selecciona tu año de universidad:", 1, 12, 1, key="school_year")

    # 2. Edad
    st.markdown("<div class='card'><h4>2. ¿Cuál es tu edad?</h4></div>", unsafe_allow_html=True)
    data['age'] = st.number_input("Ingresa tu edad (años):", min_value=5, max_value=100, value=18, key="age")

    # 3. Género
    st.markdown("<div class='card'><h4>3. Género</h4></div>", unsafe_allow_html=True)
    gender_map = {0: "Hombre", 1: "Mujer"}
    data['gender'] = st.radio("Selecciona tu género:", list(gender_map.keys()),
                              format_func=lambda x: gender_map[x], index=0, key="gender")

    # 4. BMI numérico
    st.markdown("<div class='card'><h4>4. ¿Cuál es tu BMI?</h4></div>", unsafe_allow_html=True)
    data['bmi'] = st.number_input("Ingresa tu BMI (ej. 22.5):", min_value=10.0, max_value=50.0,
                                  value=22.0, step=0.1, format="%.1f", key="bmi")

    # 5. Categoría de BMI
    st.markdown("<div class='card'><h4>5. ¿Cuál es tu categoría de BMI?</h4></div>", unsafe_allow_html=True)
    bmi_map = {
        0: "Sobrepeso", 1: "Normal", 2: "Obesidad grado I", 3: "No disponible",
        4: "Bajo peso", 5: "Obesidad grado III", 6: "Obesidad grado II"
    }
    data['who_bmi'] = st.selectbox("Selecciona categoría BMI:", list(bmi_map.keys()),
                                   format_func=lambda x: bmi_map[x], index=1, key="who_bmi")

    # 6. Puntaje PHQ-9
    st.markdown("<div class='card'><h4>6. Puntaje PHQ-9</h4></div>", unsafe_allow_html=True)
    data['phq_score'] = st.slider("PHQ-9 (0–27):", 0, 27, 5, key="phq_score")

    # 7. Gravedad de depresión
    st.markdown("<div class='card'><h4>7. ¿Qué tan deprimido te sientes?</h4></div>", unsafe_allow_html=True)
    severity_map = {
        0: "Leve", 1: "Moderadamente grave", 2: "Moderado",
        3: "Ninguno-mínimo", 4: "Severo", 5: "Ninguno"
    }
    data['depression_severity'] = st.radio("Selecciona la gravedad:", list(severity_map.keys()),
                                           format_func=lambda x: severity_map[x],
                                           index=3, key="depression_severity")

    # 8. Autoevaluación depresión
    st.markdown("<div class='card'><h4>8. ¿Sientes que tienes depresión?</h4></div>", unsafe_allow_html=True)
    data['depressiveness'] = st.radio("¿Te sientes deprimido?", [0,1],
                                      format_func=lambda x: "No" if x==0 else "Sí",
                                      index=0, key="depressiveness")

    # 9. Pensamientos suicidas
    st.markdown("<div class='card'><h4>9. ¿Has tenido pensamientos suicidas?</h4></div>", unsafe_allow_html=True)
    data['suicidal'] = st.radio("¿Has pensado en hacerte daño?", [0,1],
                                format_func=lambda x: "No" if x==0 else "Sí",
                                index=0, key="suicidal")

    # 10. Tratamiento depresión
    st.markdown("<div class='card'><h4>10. ¿Has recibido tratamiento para la depresión?</h4></div>", unsafe_allow_html=True)
    data['depression_treatment'] = st.radio("¿Estás en tratamiento para depresión?", [0,1],
                                            format_func=lambda x: "No" if x==0 else "Sí",
                                            index=0, key="depression_treatment")

    # 11. Puntaje GAD-7
    st.markdown("<div class='card'><h4>11. Puntaje GAD-7</h4></div>", unsafe_allow_html=True)
    data['gad_score'] = st.slider("GAD-7 (0–21):", 0, 21, 5, key="gad_score")

    # 12. Severidad ansiedad
    st.markdown("<div class='card'><h4>12. ¿Qué tan severa es tu ansiedad?</h4></div>", unsafe_allow_html=True)
    anx_map = {0: "Ninguna-mínima", 1: "Severa", 2: "Leve", 3: "Moderada"}
    data['anxiety_severity'] = st.selectbox("Selecciona severidad de ansiedad:",
                                            list(anx_map.keys()),
                                            format_func=lambda x: anx_map[x],
                                            index=0, key="anxiety_severity")

    # 13. Autoevaluación ansiedad
    st.markdown("<div class='card'><h4>13. ¿Te consideras ansioso?</h4></div>", unsafe_allow_html=True)
    data['anxiousness'] = st.radio("¿Te sientes ansioso habitualmente?", [0,1],
                                   format_func=lambda x: "No" if x==0 else "Sí",
                                   index=0, key="anxiousness")

    # 14. Diagnóstico ansiedad
    st.markdown("<div class='card'><h4>14. ¿Te han diagnosticado ansiedad?</h4></div>", unsafe_allow_html=True)
    data['anxiety_diagnosis'] = st.radio("¿Tienes diagnóstico médico de ansiedad?", [0,1],
                                         format_func=lambda x: "No" if x==0 else "Sí",
                                         index=0, key="anxiety_diagnosis")

    # 15. Tratamiento ansiedad
    st.markdown("<div class='card'><h4>15. ¿Has recibido tratamiento para la ansiedad?</h4></div>", unsafe_allow_html=True)
    data['anxiety_treatment'] = st.radio("¿Estás en tratamiento para ansiedad?", [0,1],
                                         format_func=lambda x: "No" if x==0 else "Sí",
                                         index=0, key="anxiety_treatment")

    # 16. Puntaje Epworth
    st.markdown("<div class='card'><h4>16. Puntaje Epworth</h4></div>", unsafe_allow_html=True)
    data['epworth_score'] = st.slider("Epworth Sleepiness Scale (0–24):", 0, 24, 8, key="epworth_score")

    # 17. Somnolencia
    st.markdown("<div class='card'><h4>17. ¿Con qué frecuencia te sientes somnoliento?</h4></div>", unsafe_allow_html=True)
    data['sleepiness'] = st.slider("0 = Nunca, 1 = Frecuente", 0, 1, 0, key="sleepiness")

    # Prepara DataFrame
    entrada = pd.DataFrame([data], columns=expected_cols)
    st.markdown("### 📋 Vista previa de tus respuestas")
    st.dataframe(entrada)

    # Botón y predicción
    if st.button("Predecir", key="predict_button"):
        try:
            proba = model.predict_proba(entrada)[0,1]
            pred  = model.predict(entrada)[0]
            respuesta = "Depresión detectada" if pred==1 else "Sin indicios fuertes de depresión"
            st.success(f"🧾 Resultado: **{respuesta}** (Probabilidad: {proba:.1%})")
        except ValueError as e:
            msg = str(e)
            m = re.search(r"columns are missing: \{(.*?)\}", msg)
            if m:
                faltantes = [c.strip(" '") for c in m.group(1).split(",")]
                st.error("❌ ¿Faltan estas columnas imprescindibles?")
                for col in faltantes:
                    st.write(f"- {col}")
            else:
                st.error("❌ Se produjo un error inesperado:")
                st.text(msg)

with tab2:
    st.markdown("## 📖 ¿Qué es la depresión?")
    st.markdown("""
La **depresión** es un trastorno del estado de ánimo que afecta cómo una persona se siente, piensa y maneja las actividades diarias. 
Los síntomas pueden incluir tristeza persistente, pérdida de interés en actividades, cambios en el apetito o el sueño, dificultad para concentrarse, y pensamientos de inutilidad o suicidio.
    """)
    st.markdown("## 🧾 Explicación de las variables utilizadas")
    st.markdown("""
- **school_year**: Año escolar actual del estudiante (1 a 12).  
- **age**: Edad en años.  
- **gender**: Género (0 = Hombre, 1 = Mujer).  
- **bmi**: Índice de masa corporal (peso/estatura²).  
- **who_bmi**: Clasificación de BMI según la OMS (ej. Normal, Sobrepeso, Obesidad).  
- **phq_score**: Puntaje del test PHQ-9, que evalúa síntomas de depresión (0–27).  
- **depression_severity**: Nivel de gravedad de la depresión.  
- **depressiveness**: Autoevaluación de sentirse deprimido (0 = No, 1 = Sí).  
- **suicidal**: Pensamientos suicidas (0 = No, 1 = Sí).  
- **depression_treatment**: Si recibe tratamiento por depresión (0 = No, 1 = Sí).  
- **gad_score**: Puntaje del test GAD-7, que mide ansiedad (0–21).  
- **anxiety_severity**: Grado de severidad de la ansiedad.  
- **anxiousness**: Autoevaluación de sentirse ansioso (0 = No, 1 = Sí).  
- **anxiety_diagnosis**: Diagnóstico clínico de ansiedad (0 = No, 1 = Sí).  
- **anxiety_treatment**: Si recibe tratamiento por ansiedad (0 = No, 1 = Sí).  
- **epworth_score**: Puntaje del test de somnolencia Epworth (0–24).  
- **sleepiness**: Frecuencia de somnolencia diurna (0 = Nunca, 1 = Frecuente).  
    """)
