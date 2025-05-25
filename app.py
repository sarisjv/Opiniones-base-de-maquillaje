import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import pipeline
import torch

# Configuración inicial
st.set_page_config(page_title="Análisis de Comentarios", layout="wide")
st.title("📊 Análisis de Comentarios")

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Cargar modelos de análisis de sentimiento y resumen (usaremos CPU para Render)
@st.cache_resource
def load_models():
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                                    device="cpu")
        
        summarizer = pipeline("summarization", 
                             model="facebook/bart-large-cnn",
                             device="cpu")
        return sentiment_analyzer, summarizer
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

sentiment_analyzer, summarizer = load_models()

# Datos de las 20 opiniones
opiniones = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural, el tono está muy bien.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación al día. 10/10.",
    "Es la mejor base si buscas una cobertura muy natural. No se nota que traes algo puesto.",
    "Excelente base buen cubrimiento.",
    "Mi piel es sensible y este producto es el mejor aliado del día a día, excelente cubrimiento.",
    "Excelente base buen cubrimiento.",
    "El empaque es terrible, no la volveré a comprar porque no sirve el envase, el producto no sale por el aplicador, es fatal.",
    "Sí se siente una piel diferente después de usar el producto.",
    "Me gusta mucho cómo deja mi piel, es buen producto aunque no me gusta su presentación.",
    "Me parece buena, pero pienso que huele mucho a alcohol, no sé si es normal.",
    "Creo que fue el color que no lo supe elegir, no está mal, pero me imaginaba algo más.",
    "La base de maquillaje ofrece un acabado mate y aterciopelado que deja la piel lisa.",
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Función para análisis de sentimiento mejorado
def analizar_sentimiento(texto):
    if not sentiment_analyzer:
        return "Modelo no disponible", 0
    
    try:
        resultado = sentiment_analyzer(texto)[0]
        etiqueta = resultado['label']
        puntuacion = resultado['score']
        
        # Convertir etiqueta numérica a texto
        if '1 star' in etiqueta or '2 stars' in etiqueta:
            return "Negativo", puntuacion
        elif '3 stars' in etiqueta:
            return "Neutral", puntuacion
        else:
            return "Positivo", puntuacion
    except:
        return "Error en análisis", 0

# Función para generar resumen
def generar_resumen(texto):
    if not summarizer:
        return "Modelo de resumen no disponible"
    
    try:
        resumen = summarizer(texto, max_length=130, min_length=30, do_sample=False)
        return resumen[0]['summary_text']
    except Exception as e:
        return f"No se pudo generar resumen: {str(e)}"

# Función para extraer temas principales
def extraer_temas(textos, n_palabras=5):
    todas_palabras = ' '.join(textos).lower()
    palabras = [p for p in nltk.word_tokenize(todas_palabras) 
               if p.isalpha() and p not in stopwords.words('spanish') and len(p) > 2]
    
    # Filtrar palabras comunes no útiles
    palabras_comunes = ['producto', 'base', 'maquillaje', 'piel', 'buen', 'bueno']
    palabras = [p for p in palabras if p not in palabras_comunes]
    
    frecuentes = Counter(palabras).most_common(n_palabras)
    return [palabra[0] for palabra in frecuentes]

# Interfaz principal
def main():
    st.sidebar.title("Opciones")
    opcion = st.sidebar.radio("Seleccione una opción:", 
                             ["Analizar nuevo comentario", "Explorar comentarios existentes"])
    
    if opcion == "Analizar nuevo comentario":
        st.header("📝 Analizar nuevo comentario")
        
        comentario = st.text_area("Escribe tu comentario aquí:", height=150)
        
        if st.button("Analizar"):
            if comentario.strip():
                with st.spinner("Analizando..."):
                    # Análisis de sentimiento
                    sentimiento, puntuacion = analizar_sentimiento(comentario)
                    
                    # Generar resumen
                    resumen = generar_resumen(comentario)
                    
                    # Mostrar resultados
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Análisis de Sentimiento")
                        st.write(f"**Tipo:** {sentimiento}")
                        st.write(f"**Confianza:** {puntuacion:.2f}")
                        
                        # Visualización simple
                        if sentimiento == "Positivo":
                            st.success("✅ Comentario positivo")
                        elif sentimiento == "Negativo":
                            st.error("❌ Comentario negativo")
                        else:
                            st.info("➖ Comentario neutral")
                    
                    with col2:
                        st.subheader("Resumen automático")
                        st.write(resumen)
            else:
                st.warning("Por favor escribe un comentario para analizar")
    
    else:  # Explorar comentarios existentes
        st.header("📂 Explorar comentarios existentes")
        
        df = pd.DataFrame({'Opinión': opiniones})
        df['Análisis'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[0])
        df['Puntuación'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[1])
        
        opcion_existente = st.radio("Qué deseas hacer con los comentarios existentes?",
                                  ["Ver tabla completa", 
                                   "Obtener resumen general",
                                   "Analizar temas principales"])
        
        if opcion_existente == "Ver tabla completa":
            st.subheader("Tabla de Comentarios")
            st.dataframe(df)
            
        elif opcion_existente == "Obtener resumen general":
            with st.spinner("Generando resumen de todos los comentarios..."):
                todos_comentarios = " ".join(opiniones)
                resumen_general = generar_resumen(todos_comentarios)
                
                st.subheader("Resumen General")
                st.write(resumen_general)
                
                # Estadísticas
                st.subheader("Estadísticas")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total comentarios", len(opiniones))
                with col2:
                    positivos = df[df['Análisis'] == "Positivo"].shape[0]
                    st.metric("Comentarios positivos", positivos)
                with col3:
                    negativos = df[df['Análisis'] == "Negativo"].shape[0]
                    st.metric("Comentarios negativos", negativos)
                
        else:  # Analizar temas principales
            st.subheader("Temas principales en los comentarios")
            
            # Opción para filtrar por tipo de comentario
            filtro = st.selectbox("Filtrar por tipo de comentario:", 
                                ["Todos", "Positivos", "Negativos", "Neutrales"])
            
            if filtro == "Todos":
                textos = opiniones
            else:
                textos = df[df['Análisis'] == filtro[:-1]]['Opinión'].tolist()
            
            temas = extraer_temas(textos)
            
            st.write("**Palabras clave más frecuentes:**")
            for i, tema in enumerate(temas, 1):
                st.write(f"{i}. {tema.capitalize()}")
            
            # Wordcloud
            st.subheader("Nube de palabras")
            todas_palabras = ' '.join(textos)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(todas_palabras)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
