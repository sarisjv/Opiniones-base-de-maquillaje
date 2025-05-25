import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from transformers import pipeline
import time

# Configuración inicial
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("Análisis de Opiniones sobre Bases de Maquillaje")

# Descargar recursos de NLTK (solo una vez)
nltk.download('punkt')
nltk.download('stopwords')

# Cargar modelo de análisis de sentimientos (cacheado)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_model = load_sentiment_model()

# Inicializar estado de opiniones si no existe
if 'opiniones' not in st.session_state:
    st.session_state.opiniones = [
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

# Función para limpiar y tokenizar texto
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Záéíóúñ\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

# Análisis de sentimientos
def analyze_sentiment(text):
    try:
        result = sentiment_model(text)[0]
        label = result['label']
        score = result['score']
        
        if '1 star' in label:
            return "Muy Negativo", score
        elif '2 stars' in label:
            return "Negativo", score
        elif '3 stars' in label:
            return "Neutral", score
        elif '4 stars' in label:
            return "Positivo", score
        else:
            return "Muy Positivo", score
    except:
        positive_words = ['magnífico', 'espectacular', 'maravilloso', 'excelente', 'buen', 'mejor', 'bonita', 'lindo', 'natural']
        negative_words = ['terrible', 'fatal', 'arde', 'pasteluda', 'oscuro', 'horas', 'alcohol']
        text = text.lower()
        pos = sum(1 for word in positive_words if word in text)
        neg = sum(1 for word in negative_words if word in text)
        if pos > neg:
            return "Positivo", pos/(pos+neg+1)
        elif neg > pos:
            return "Negativo", neg/(pos+neg+1)
        else:
            return "Neutral", 0.5

def generate_summary(text):
    sentences = text.split('. ')
    if len(sentences) >= 3:
        return '. '.join(sentences[:2]) + '.'
    return text

def analyze_topics(opiniones_seleccionadas, n_topics=3):
    all_text = ' '.join(opiniones_seleccionadas)
    tokens = clean_and_tokenize(all_text)
    word_counts = Counter(tokens)
    common_words = ['producto', 'base', 'maquillaje', 'piel']
    topics = [word for word, count in word_counts.most_common(20) if word not in common_words]
    return topics[:n_topics]

# Convertir opiniones a DataFrame
df = pd.DataFrame({'Opinión': st
