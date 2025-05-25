import streamlit as st
import pandas as pd

# Función mejorada de análisis de sentimiento
def analizar_sentimiento(texto):
    texto = texto.lower()
    score = 0

    # Frases compuestas
    frases_positivas = {
        "me encanta": 3, "muy bueno": 2, "deja la piel espectacular": 3,
        "me gusta mucho": 2, "muy buena": 2, "es la mejor": 3,
        "lo recomiendo": 2, "muy satisfecho": 2
    }

    frases_negativas = {
        "no me gusta": 3, "no volveré": 3, "no sirve": 3, "es fatal": 3,
        "me arde": 3, "me decepcionó": 3, "no lo recomiendo": 3,
        "no es tan bueno": 2, "no es tan buena": 2, "no fue lo que esperaba": 2,
        "no me convenció": 2, "no cumplió mis expectativas": 3,
        "me pareció regular": 2, "esperaba más": 2, "muy regular": 2
    }

    # Palabras individuales
    palabras_positivas = {
        'magnífico': 3, 'espectacular': 3, 'maravilloso': 3, 'excelente': 3,
        'mejor': 2, 'recomiendo': 2, 'buen': 2, 'genial': 2, 'perfecto': 3,
        'bonita': 2, 'natural': 1, 'sana': 1, 'fácil': 1, 'calidad': 2
    }

    palabras_negativas = {
        'terrible': 3, 'fatal': 3, 'arde': 3, 'problema': 2, 'decepcionante': 3,
        'pasteluda': 2, 'oscuro': 1, 'alcohol': 1, 'duro': 1
    }

    # Evaluar frases compuestas
    for frase, valor in frases_positivas.items():
        if frase in texto:
            score += valor

    for frase, valor in frases_negativas.items():
        if frase in texto:
            score -= valor

    # Evaluar palabras individuales
    for palabra, valor in palabras_positivas.items():
        if palabra in texto:
            score += valor

    for palabra, valor in palabras_negativas.items():
        if palabra in texto:
            score -= valor

    # Clasificación final
    if score >= 2:
        return "Positivo", score
    elif score <= -2:
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

# Interfaz con Streamlit
st.set_page_config(page_title="Análisis de Comentarios", page_icon="💬")
st.title("🧴 Análisis de Comentarios de Productos de Cuidado Facial")

comentarios = []

with st.form("comentario_form"):
    comentario = st.text_area("Escribe tu comentario sobre el producto:")
    submitted = st.form_submit_button("Analizar Comentario")

    if submitted and comentario:
        sentimiento, puntaje = analizar_sentimiento(comentario)
        comentarios.append({
            "Comentario": comentario,
            "Sentimiento": sentimiento,
            "Puntaje": puntaje
        })

# Mostrar comentarios analizados
if comentarios:
    df = pd.DataFrame(comentarios)
    st.subheader("📊 Resultados del Análisis")
    st.dataframe(df, use_container_width=True)

    # Contadores generales
    positivos = df[df["Sentimiento"] == "Positivo"].shape[0]
    negativos = df[df["Sentimiento"] == "Negativo"].shape[0]
    neutrales = df[df["Sentimiento"] == "Neutral"].shape[0]

    st.markdown("### 📌 Resumen")
    st.write(f"🔹 Comentarios positivos: {positivos}")
    st.write(f"🔹 Comentarios negativos: {negativos}")
    st.write(f"🔹 Comentarios neutrales: {neutrales}")
