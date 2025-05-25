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
import gc

# Descargar recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ----------------------------------
# Datos para modelo Naive Bayes
comentarios_nb = [
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

etiquetas_nb = [
    "positivo", "negativo", "neutral", "positivo", "negativo", "neutral",
    "negativo", "positivo", "negativo", "positivo", "neutral", "neutral",
    "positivo", "negativo", "negativo", "positivo", "neutral", "positivo",
    "negativo", "neutral"
]

# Entrenar modelo Naive Bayes
vectorizer_nb = CountVectorizer()
X_nb = vectorizer_nb.fit_transform(comentarios_nb)
modelo_nb = MultinomialNB()
modelo_nb.fit(X_nb, etiquetas_nb)

# ----------------------------------
# Opiniones para análisis avanzado
opiniones = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural, el tono está muy bien. Si quieres una opción natural de maquillaje esta es la mejor.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación al día. 10/10.",
    "Es la mejor base si buscas una cobertura muy natural. No se nota que traes algo puesto, pero empareja el tono y deja la piel luciendo muy sana y bonita.",
    "Excelente base buen cubrimiento.",
    "Mi piel es sensible y este producto es el mejor aliado del día a día, excelente cubrimiento, rendimiento porque con poco tienes sobre el rostro y te ves tan natural.",
    "Excelente base buen cubrimiento.",
    "El empaque es terrible, no la volveré a comprar porque no sirve el envase, el producto no sale por el aplicador, es fatal.",
    "Sí se siente una piel diferente después de usar el producto.",
    "Me gusta mucho cómo deja mi piel, es buen producto aunque no me gusta su presentación.",
    "Me parece buena, pero pienso que huele mucho a alcohol, no sé si es normal.",
    "Creo que fue el color que no lo supe elegir, no está mal, pero me imaginaba algo más uff.",
    "La base de maquillaje ofrece un acabado mate y aterciopelado que deja la piel lisa y es fácil de aplicar. En general, es una base que destaca por su buen desempeño y calidad.",
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas, pero muy bueno.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces para mejor cobertura pero ya queda la piel pasteluda.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# ----------------------------------
# Funciones análisis avanzado

def analizar_sentimiento(texto):
    positivo = {
        'magnífico': 3, 'espectacular': 3, 'maravilloso': 3, 'excelente': 3,
        'mejor': 2, 'recomiendo': 2, 'buen': 2, 'genial': 2, 'perfecto': 3,
        'bonita': 2, 'natural': 1, 'sana': 1, 'fácil': 1, 'calidad': 2
    }
    negativo = {
        'terrible': 3, 'fatal': 3, 'no sirve': 3, 'no me gusta': 2,
        'arde': 3, 'problema': 2, 'decepcionante': 3, 'pasteluda': 2,
        'oscuro': 1, 'alcohol': 1, 'duro': 1
    }
    texto = texto.lower()
    score = 0
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor
    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor
    if score > 2:
        return "Positivo", score
    elif score < -2:
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

def generar_resumen(texto):
    oraciones = nltk.sent_tokenize(texto)
    if len(oraciones) > 1:
        return f"{oraciones[0]} [...] {oraciones[-1]}"
    return texto

def palabras_clave(textos, n=10):
    palabras_comunes = {'producto', 'base', 'maquillaje', 'piel', 'buen', 'como'}
    palabras = []
    for texto in textos:
        tokens = [p.lower() for p in nltk.word_tokenize(texto)
                  if p.isalpha() and p not in stopwords.words('spanish')
                  and p.lower() not in palabras_comunes]
        palabras.extend(tokens)
    return Counter(palabras).most_common(n)

# ----------------------------------
# Interfaz Streamlit

st.set_page_config(
    page_title="Análisis Completo de Opiniones",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App de análisis de opiniones con todas las funcionalidades"
    }
)

st.title("💄 Análisis de Sentimientos y Opiniones de Productos Cosméticos")

tab1, tab2 = st.tabs(["🔎 Modelo Naive Bayes", "📊 Análisis Avanzado y Visualización"])

with tab1:
    st.header("Análisis con Modelo Naive Bayes")
    comentario_usuario = st.text_area("Escribe tu comentario aquí:")
    if st.button("Analizar Sentimiento (NB)"):
        if comentario_usuario.strip() == "":
            st.warning("Por favor escribe un comentario antes de analizar.")
        else:
            vector_usuario = vectorizer_nb.transform([comentario_usuario])
            prediccion = modelo_nb.predict(vector_usuario)[0]
            proba = modelo_nb.predict_proba(vector_usuario)[0]
            st.success(f"Sentimiento detectado: **{prediccion.upper()}**")
            st.write("Probabilidades:")
            for etiqueta, prob in zip(modelo_nb.classes_, proba):
                st.write(f"{etiqueta.capitalize()}: {prob:.2f}")

with tab2:
    st.header("Análisis Avanzado de Opiniones Existentes y Nuevas")
    opcion = st.radio("Seleccione acción:",
                      ["Analizar Nuevo Comentario", "Explorar Opiniones Existentes"])

    if opcion == "Analizar Nuevo Comentario":
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
                            st.info(f"➖ {sentimiento}")
                    with col2:
                        st.subheader("Resumen Automático")
                        resumen = generar_resumen(comentario)
                        st.text_area(" ", value=resumen, height=100)
            else:
                st.warning("Por favor escribe un comentario para analizar")

    else:  # Explorar Opiniones Existentes
        st.header("Análisis de las 20 Opiniones")
        df = pd.DataFrame({'Opinión': opiniones})
        df['Sentimiento'] = df['Opinión'].apply
