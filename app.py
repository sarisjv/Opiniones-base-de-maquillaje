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
    "Un sérum magnífico, deja la piel espectacular con un acabado natural, el tono está muy bien.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación al día. 10/10.",
    "Es la mejor base si buscas una cobertura muy natural. No se nota que traes algo puesto.",
    "Excelente base buen cubrimiento.",
    "Mi piel es sensible y este producto es el mejor aliado del día a día.",
    "El empaque es terrible, no la volveré a comprar porque no sirve el envase.",
    "Me gusta mucho cómo deja mi piel, es buen producto aunque no me gusta su presentación.",
    "Me parece buena, pero pienso que huele mucho a alcohol, no sé si es normal.",
    "La base de maquillaje ofrece un acabado mate y aterciopelado que deja la piel lisa.",
    "No me gustó su cobertura."
]

# Inicializar opiniones en session_state
if 'opiniones' not in st.session_state:
    st.session_state.opiniones = opiniones_iniciales.copy()

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
    positive_words = ['magnífico', 'espectacular', 'maravilloso', 'excelente', 'buen', 'mejor']
    negative_words = ['terrible', 'no sirve', 'problema', 'caro', 'no me gustó']
    
    text = text.lower()
    pos = sum(1 for word in positive_words if word in text)
    neg = sum(1 for word in negative_words if word in text)
    
    if pos > neg:
        return "Positivo", pos/(pos+neg+1)
    elif neg > pos:
        return "Negativo", neg/(pos+neg+1)
    else:
        return "Neutral", 0.5

# Generar resumen
def generate_summary(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) >= 3:
        return '. '.join(sentences[:2]) + '...'
    return text

# Analizar temas principales
def analyze_topics(opiniones_seleccionadas):
    all_text = ' '.join(opiniones_seleccionadas)
    tokens = clean_and_tokenize(all_text)
    word_counts = Counter(tokens)
    common_words = ['producto', 'base', 'maquillaje', 'piel']
    topics = [word for word, count in word_counts.most_common(10) if word not in common_words]
    return topics

# Interfaz de usuario
def main():
    st.sidebar.header("Agregar Nueva Opinión")
    nueva_opinion = st.sidebar.text_area("Escribe tu opinión:")
    if st.sidebar.button("Agregar Opinión"):
        if nueva_opinion:
            st.session_state.opiniones.append(nueva_opinion)
            st.sidebar.success("¡Opinión agregada!")
        else:
            st.sidebar.warning("Por favor escribe una opinión")

    # Mostrar todas las opiniones
    st.header("Todas las Opiniones")
    df = pd.DataFrame({
        'Opinión': st.session_state.opiniones,
        'Número': range(1, len(st.session_state.opiniones)+1)
    })
    
    # Aplicar análisis
    df['Sentimiento'], df['Confianza'] = zip(*df['Opinión'].apply(analyze_sentiment))
    df['Resumen'] = df['Opinión'].apply(generate_summary)
    
    # Mostrar tabla con todas las opiniones
    st.dataframe(df[['Número', 'Opinión', 'Sentimiento']], height=400)
    
    # Sección de análisis
    st.header("Análisis de Opiniones")
    
    # Seleccionar opiniones para análisis
    selected_indices = st.multiselect(
        "Selecciona opiniones para analizar:",
        options=df['Número'].tolist(),
        default=df['Número'].tolist()
    )
    
    selected_opinions = [op for i, op in enumerate(st.session_state.opiniones) if (i+1) in selected_indices]
    
    if st.button("Analizar Opiniones Seleccionadas"):
        if not selected_opinions:
            st.warning("Selecciona al menos una opinión")
        else:
            # Análisis de texto
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Palabras más frecuentes")
                all_text = ' '.join(selected_opinions)
                tokens = clean_and_tokenize(all_text)
                word_counts = Counter(tokens)
                top_words = word_counts.most_common(10)
                top_words_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
                st.bar_chart(top_words_df.set_index('Palabra'))
            
            with col2:
                st.subheader("Distribución de sentimientos")
                selected_df = df[df['Número'].isin(selected_indices)]
                sentiment_counts = selected_df['Sentimiento'].value_counts()
                st.bar_chart(sentiment_counts)
            
            # Temas principales
            st.subheader("Temas principales detectados")
            topics = analyze_topics(selected_opinions)
            for i, topic in enumerate(topics, 1):
                st.write(f"{i}. {topic.capitalize()}")
    
    # Consultar opinión específica
    st.header("Consultar Opinión Específica")
    opinion_num = st.selectbox(
        "Selecciona una opinión:",
        options=df['Número'].tolist()
    )
    
    selected_opinion = st.session_state.opiniones[opinion_num-1]
    sentiment, confidence = analyze_sentiment(selected_opinion)
    
    st.write(f"**Opinión #{opinion_num}:**")
    st.info(selected_opinion)
    
    st.write(f"**Sentimiento:** {sentiment} (Confianza: {confidence*100:.1f}%)")
    st.write("**Resumen:**", generate_summary(selected_opinion))

if __name__ == "__main__":
    main()
