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

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar modelo de análisis de sentimientos (lo cacheamos para mejor performance)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_model = load_sentiment_model()

# Opiniones integradas en el código
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

# Función para limpiar y tokenizar texto
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Záéíóúñ\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

# Análisis de sentimientos mejorado con el modelo
def analyze_sentiment(text):
    try:
        result = sentiment_model(text)[0]
        label = result['label']
        score = result['score']
        
        # Convertir etiqueta a formato más amigable
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
        # Fallback si el modelo falla
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

# Función para generar resumen (simplificado)
def generate_summary(text):
    sentences = text.split('. ')
    if len(sentences) >= 3:
        return '. '.join(sentences[:2]) + '.'
    return text

# Función para analizar temas principales
def analyze_topics(opiniones_seleccionadas, n_topics=3):
    all_text = ' '.join(opiniones_seleccionadas)
    tokens = clean_and_tokenize(all_text)
    word_counts = Counter(tokens)
    
    # Excluir palabras demasiado comunes
    common_words = ['producto', 'base', 'maquillaje', 'piel']
    topics = [word for word, count in word_counts.most_common(20) if word not in common_words]
    
    return topics[:n_topics]

# Interfaz de usuario
def main():
    st.sidebar.title("Opciones")
    
    # Convertir a DataFrame
    df = pd.DataFrame({'Opinión': opiniones})
    df['Sentimiento'] = df['Opinión'].apply(lambda x: analyze_sentiment(x)[0])
    df['Puntaje'] = df['Opinión'].apply(lambda x: analyze_sentiment(x)[1])
    
    # Sección para nuevo comentario
    st.header("Añadir nuevo comentario")
    new_comment = st.text_area("Escribe tu opinión sobre el producto:")
    
    if st.button("Analizar comentario"):
        if new_comment:
            with st.spinner("Analizando sentimiento..."):
                sentiment, score = analyze_sentiment(new_comment)
                summary = generate_summary(new_comment)
                
                st.success("Análisis completado!")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentimiento detectado")
                    st.write(f"**Tipo:** {sentiment}")
                    st.write(f"**Puntaje de confianza:** {score:.2f}")
                    
                    # Mostrar emoji según sentimiento
                    if "Positivo" in sentiment:
                        st.write("😊")
                    elif "Negativo" in sentiment:
                        st.write("😞")
                    else:
                        st.write("😐")
                
                with col2:
                    st.subheader("Resumen automático")
                    st.write(summary)
                
                # Agregar a la lista de opiniones (en memoria)
                opiniones.append(new_comment)
                st.experimental_rerun()
        else:
            st.warning("Por favor escribe un comentario antes de analizar")
    
    st.header("Análisis de opiniones existentes")
    
    # Mostrar todas las opiniones con filtros
    st.subheader("Explorar opiniones")
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        filter_sentiment = st.multiselect(
            "Filtrar por sentimiento",
            options=df['Sentimiento'].unique(),
            default=df['Sentimiento'].unique()
        )
    
    with col2:
        min_score = st.slider(
            "Puntaje mínimo de confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    # Aplicar filtros
    filtered_df = df[
        (df['Sentimiento'].isin(filter_sentiment)) & 
        (df['Puntaje'] >= min_score)
    ]
    
    # Mostrar opiniones filtradas
    st.dataframe(filtered_df[['Opinión', 'Sentimiento', 'Puntaje']].sort_values('Puntaje', ascending=False))
    
    # Selección de opiniones para análisis
    st.subheader("Analizar opiniones seleccionadas")
    selected_indices = st.multiselect(
        "Selecciona opiniones para analizar (máx 5)",
        options=range(len(opiniones)),
        format_func=lambda x: f"Opinión {x+1}: {opiniones[x][:50]}..."
    )
    
    if selected_indices:
        selected_comments = [opiniones[i] for i in selected_indices if i < len(opiniones)]
        
        if len(selected_comments) > 5:
            st.warning("Has seleccionado más de 5 opiniones. Mostrando solo las primeras 5.")
            selected_comments = selected_comments[:5]
        
        st.write("**Opiniones seleccionadas:**")
        for i, comment in enumerate(selected_comments, 1):
            st.write(f"{i}. {comment}")
        
        # Opciones de análisis
        analysis_option = st.radio(
            "¿Qué análisis deseas realizar?",
            options=["Resumen conjunto", "Temas principales", "Análisis de sentimiento agregado"]
        )
        
        if st.button("Realizar análisis"):
            with st.spinner("Procesando..."):
                time.sleep(1)  # Simular procesamiento
                
                if analysis_option == "Resumen conjunto":
                    combined_text = ' '.join(selected_comments)
                    summary = generate_summary(combined_text)
                    st.subheader("Resumen conjunto")
                    st.write(summary)
                
                elif analysis_option == "Temas principales":
                    topics = analyze_topics(selected_comments)
                    st.subheader("Temas principales encontrados")
                    for i, topic in enumerate(topics, 1):
                        st.write(f"{i}. {topic.capitalize()}")
                    
                    # Mostrar nube de palabras para los seleccionados
                    st.write("**Nube de palabras de las opiniones seleccionadas**")
                    selected_text = ' '.join(selected_comments)
                    tokens = clean_and_tokenize(selected_text)
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                
                elif analysis_option == "Análisis de sentimiento agregado":
                    sentiments = [analyze_sentiment(comment)[0] for comment in selected_comments]
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    st.subheader("Distribución de sentimientos")
                    st.bar_chart(sentiment_counts)
                    
                    avg_score = sum(analyze_sentiment(comment)[1] for comment in selected_comments) / len(selected_comments)
                    st.write(f"**Puntaje promedio:** {avg_score:.2f}")
    
    # Análisis general de todas las opiniones
    st.header("Análisis general")
    col1, col2 = st.columns(2)
    
    with col1:
        # Nube de palabras
        st.write("**Nube de palabras**")
        all_text = ' '.join(opiniones)
        tokens = clean_and_tokenize(all_text)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    with col2:
        # Palabras más frecuentes
        st.write("**Palabras más frecuentes**")
        word_counts = Counter(tokens)
        top_words = word_counts.most_common(10)
        top_words_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
        st.bar_chart(top_words_df.set_index('Palabra'))
    
    # Distribución de sentimientos
    st.write("**Distribución de sentimientos**")
    sentiment_counts = df['Sentimiento'].value_counts()
    st.bar_chart(sentiment_counts)

if __name__ == "__main__":
    main()
