import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import gc

# Configuración inicial
st.set_page_config(
    page_title="Análisis Completo de Opiniones",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App de análisis de opiniones con múltiples funcionalidades"
    }
)

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Datos y modelo Naive Bayes
comentarios = [
    "Me encanta este producto, es económico",
    "No me gusta, me reseca mucho la piel",
    "Es un producto más, no está mal",
    "Excelente calidad y precio",
    "Horrible, me irrita la piel",
    "No tengo opinión",
    "No lo volvería a comprar",
    "Lo amo, me deja la piel suave",
    "Pésimo, me ardió la cara",
    "Muy bueno, huele delicioso",
    "No me hizo efecto",
    "Es neutral para mí",
    "Fantástico, super recomendado",
    "Decepcionante, esperaba más",
    "No me gustó para nada",
    "Lo recomiendo totalmente",
    "Es aceptable, nada especial",
    "Una maravilla de producto",
    "Es malo, me brotó la piel",
    "Me agrada, pero no es el mejor"
]

etiquetas = [
    "positivo", "negativo", "neutral", "positivo", "negativo", "neutral",
    "negativo", "positivo", "negativo", "positivo", "neutral", "neutral",
    "positivo", "negativo", "negativo", "positivo", "neutral", "positivo",
    "negativo", "neutral"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comentarios)
modelo_nb = MultinomialNB()
modelo_nb.fit(X, etiquetas)

# Función mejorada de análisis de sentimiento
def analizar_sentimiento(texto):
    positivo = {
        'magnífico': 3, 'espectacular': 3, 'maravilloso': 3, 'excelente': 3,
        'mejor': 2, 'recomiendo': 2, 'buen': 2, 'genial': 2, 'perfecto': 3,
        'bonita': 2, 'natural': 1, 'sana': 1, 'fácil': 1, 'calidad': 2,
        '10/10': 3, 'mejor aliado': 2, 'muy bien': 2, 'muy bueno': 2,
        'excelente cobertura': 3, 'no es grasosa': 1, 'diferente': 1
    }

    negativo = {
        'terrible': 3, 'fatal': 3, 'no sirve': 3, 'no me gusta': 2,
        'arde': 3, 'problema': 2, 'decepcionante': 3, 'pasteluda': 2,
        'oscuro': 1, 'alcohol': 1, 'duro': 1, 'no volveré': 3,
        'no sale': 2, 'no gustó': 2, 'no gusta': 2,
        'pensé que sería mejor': 2, 'queda pasteluda': 2, 'irrita': 2, 'problemas': 2
    }

    expresiones_negativas = [
        r'no la volver[é|e] a comprar',
        r'no me gust[oó]',
        r'no me gusta',
        r'es mucho más oscuro',
        r'queda la piel pasteluda',
        r'me arde al aplicarla',
        r'no sirve el envase',
        r'problema con el producto'
    ]

    texto = texto.lower()
    score = 0

    for expr in expresiones_negativas:
        if re.search(expr, texto):
            score -= 3

    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor

    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor

    if score >= 3:
        return "Positivo", score
    elif score <= -2:
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

# Función de resumen
def generar_resumen(texto):
    oraciones = nltk.sent_tokenize(texto)
    if len(oraciones) > 1:
        return oraciones[0] + " [...] " + oraciones[-1]
    return texto

# Función para extraer palabras clave
def palabras_clave(textos, n=10):
    palabras_comunes = {'producto', 'base', 'maquillaje', 'piel', 'buen', 'como', 'que', 'con', 'para'}
    palabras = []

    for texto in textos:
        tokens = [p.lower() for p in nltk.word_tokenize(texto)
                  if p.isalpha() and p not in stopwords.words('spanish')
                  and p.lower() not in palabras_comunes]
        palabras.extend(tokens)

    return Counter(palabras).most_common(n)

# Interfaz
def main():
    st.title("💬 Análisis Completo de Opiniones de Productos Cosméticos")

    tab_nb, tab_avanzado = st.tabs(["Análisis rápido", "Análisis avanzado y exploración"])

    with tab_nb:
        st.header("Análisis rápido")
        comentario_usuario = st.text_area("Escribe tu comentario aquí:")
        if st.button("Analizar Sentimiento"):
            if comentario_usuario.strip() == "":
                st.warning("Por favor escribe un comentario antes de analizar.")
            else:
                comentario_vectorizado = vectorizer.transform([comentario_usuario])
                prediccion = modelo_nb.predict(comentario_vectorizado)[0]
                st.success(f"Sentimiento detectado: **{prediccion.upper()}**")

                proba = modelo_nb.predict_proba(comentario_vectorizado)[0]
                for etiqueta, prob in zip(modelo_nb.classes_, proba):
                    st.write(f"{etiqueta.capitalize()}: {prob:.2f}")

    with tab_avanzado:
        st.header("Análisis avanzado de opiniones y exploración de datos")

        subtab1, subtab2 = st.tabs(["Analizar nuevo comentario", "Explorar opiniones existentes"])

        with subtab1:
            comentario = st.text_area("Escribe tu opinión sobre el producto:", height=150)
            if st.button("Analizar Sentimiento (Avanzado)"):
                if comentario.strip():
                    with st.spinner("Analizando..."):
                        sentimiento, puntaje = analizar_sentimiento(comentario)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Resultado del Análisis")
                            if sentimiento == "Positivo":
                                st.success(f"✅ {sentimiento} (Puntaje: {puntaje})")
                            elif sentimiento == "Negativo":
                                st.error(f"❌ {sentimiento} (Puntaje: {puntaje})")
                            else:
                                st.info(f"➖ {sentimiento} (Puntaje: {puntaje})")

        with subtab2:
            st.subheader("Opiniones existentes")
            opiniones = [
                "Un sérum magnífico, deja la piel espectacular con un acabado natural, el tono está muy bien.",
                "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación al día. 10/10.",
                "El empaque es terrible, no la volveré a comprar porque no sirve el envase.",
                "Me gusta mucho cómo deja mi piel, es buen producto aunque no me gusta su presentación.",
                "No me gustó su cobertura.",
                "La sensación en la piel no me gusta, me arde al aplicarla.",
                "Excelente cobertura y precio.",
                "Muy buen producto, solo que dura poco tiempo.",
                "Es la mejor base si buscas una cobertura muy natural.",
                "Pensé me sentaría mejor el número 8, es buena pero queda la piel pasteluda."
            ]

            for i, op in enumerate(opiniones, 1):
                sentimiento, puntaje = analizar_sentimiento(op)
                st.markdown(f"**Opinión {i}:** {op}")
                st.write(f"➡️ Resultado: **{sentimiento}** (Puntaje: {puntaje})")
                st.divider()

if __name__ == "__main__":
    main()
