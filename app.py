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
import re  # ✅ ESTA IMPORTACIÓN ES CLAVE PARA QUE FUNCIONE analizar_sentimiento

# Descargar recursos necesarios de nltk
nltk.download('punkt')
nltk.download('stopwords')

# Datos de entrenamiento
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

# Entrenamiento del modelo
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comentarios)
modelo_nb = MultinomialNB()
modelo_nb.fit(X, etiquetas)

# Diccionarios y expresiones negativas para análisis avanzado
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

# Función de análisis de sentimiento mejorada
def analizar_sentimiento(texto):
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

# Streamlit UI
st.title("Análisis de Opiniones")

comentario_usuario = st.text_input("Escribe tu opinión:")

if comentario_usuario:
    sentimiento, puntaje = analizar_sentimiento(comentario_usuario)
    st.write(f"Sentimiento detectado: **{sentimiento}** (Puntaje: {puntaje})")
