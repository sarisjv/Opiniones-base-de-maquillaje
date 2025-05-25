import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

# Configuración inicial
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("Análisis de Opiniones sobre Bases de Maquillaje")

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Opiniones iniciales
opiniones_iniciales = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación.",
    "Es la mejor base si buscas una cobertura muy natural.",
    "Excelente base buen cubrimiento.",
    "Mi piel es sensible y este producto es el mejor aliado del día a día.",
    "El empaque es terrible, no la volveré a comprar porque no sirve el envase.",
    "Me gusta mucho cómo deja mi piel, es buen producto.",
    "Me parece buena, pero pienso que huele mucho a alcohol.",
    "La base de maquillaje ofrece un acabado mate y aterciopelado.",
    "No me gustó su cobertura."
]

# Inicializar opiniones en session_state
if 'opiniones' not in st.session_state:
    st.session_state.opiniones = opiniones_iniciales.copy()

# Funciones de análisis optimizadas
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\sáéíóúñ]', '', text)
    return text

def analyze_sentiment(text):
    positive_words = {'magnífico', 'espectacular', 'maravilloso', 'excelente', 'buen', 'mejor'}
    negative_words = {'terrible', 'no sirve', 'problema', 'alcohol', 'no me gustó'}
    
    clean_text = clean_text(text)
    tokens = set(word_tokenize(clean_text))
    
    pos = len(tokens & positive_words)
    neg = len(tokens & negative_words)
    
    if pos > neg:
        return "Positivo", round(pos/(pos+neg+1), 2)
    elif neg > pos:
        return "Negativo", round(neg/(pos+neg+1), 2)
    return "Neutral", 0.5

def generate_summary(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return '. '.join(sentences[:2]) + ('...' if len(sentences) > 2 else '')

def analyze_topics(texts, n=3):
    all_text = ' '.join(texts)
    tokens = [word for word in word_tokenize(clean_text(all_text)) 
             if word not in stopwords.words('spanish') and len(word) > 3]
    return [word for word, _ in Counter(tokens).most_common(n)]

# Interfaz de usuario optimizada
def main():
    # Sidebar para nuevas opiniones
    with st.sidebar:
        st.header("➕ Nueva Opinión")
        new_opinion = st.text_area("Escribe tu opinión:")
        if st.button("Analizar y Agregar"):
            if new_opinion:
                sentiment, score = analyze_sentiment(new_opinion)
                st.session_state.opiniones.append(new_opinion)
                
                st.success(f"✅ Opinión agregada | Sentimiento: {sentiment} ({score*100}%)")
                st.subheader("Resumen:")
                st.write(generate_summary(new_opinion))
            else:
                st.warning("Por favor escribe una opinión")

    # Mostrar todas las opiniones
    st.header("📝 Todas las Opiniones")
    df = pd.DataFrame({
        'N°': range(1, len(st.session_state.opiniones)+1),
        'Opinión': st.session_state.opiniones,
        'Sentimiento': [analyze_sentiment(op)[0] for op in st.session_state.opiniones]
    })
    st.dataframe(df, height=400)

    # Análisis interactivo
    st.header("🔍 Análisis Interactivo")
    
    # Selección de opiniones
    selected = st.multiselect(
        "Selecciona opiniones para analizar:",
        options=df['N°'].tolist(),
        default=df['N°'].tolist()[:5]
    )
    
    if st.button("Analizar selección"):
        selected_ops = [st.session_state.opiniones[i-1] for i in selected]
        
        # Análisis de temas
        st.subheader("📌 Temas principales")
        topics = analyze_topics(selected_ops)
        for i, topic in enumerate(topics, 1):
            st.write(f"{i}. {topic.capitalize()}")
        
        # Resumen colectivo
        st.subheader("📄 Resumen colectivo")
        combined_text = ' '.join(selected_ops)
        st.write(generate_summary(combined_text))
        
        # Distribución de sentimientos
        st.subheader("😃 Sentimientos")
        sentiment_dist = df[df['N°'].isin(selected)]['Sentimiento'].value_counts()
        st.bar_chart(sentiment_dist)

    # Análisis individual
    st.header("🔎 Análisis por opinión")
    op_num = st.selectbox("Selecciona una opinión:", df['N°'])
    selected_op = st.session_state.opiniones[op_num-1]
    
    st.write("**Opinión seleccionada:**")
    st.info(selected_op)
    
    sentiment, score = analyze_sentiment(selected_op)
    st.write(f"**Sentimiento:** {sentiment} (Confianza: {score*100:.0f}%)")
    st.write("**Resumen:**", generate_summary(selected_op))

if __name__ == "__main__":
    main()
