import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

# Configuración inicial
st.set_page_config(page_title="Analizador Interactivo", layout="wide")
st.title("💬 Analizador de Opiniones Interactivo")

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Datos de las 20 opiniones
opiniones = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural.",
    "Este producto es maravilloso, minimiza imperfecciones.",
    # ... (todas tus 20 opiniones aquí)
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Funciones de análisis mejoradas
def analizar_sentimiento(texto):
    """Analiza el sentimiento con un sistema de puntuación mejorado"""
    texto = texto.lower()
    
    # Diccionario de palabras clave con pesos
    palabras_clave = {
        'positivo': {'magnífico':2, 'espectacular':2, 'maravilloso':2, 'excelente':2, 'recomiendo':1, 'buen':1},
        'negativo': {'terrible':2, 'fatal':2, 'arde':2, 'problema':1, 'decepcionante':2, 'no me gusta':2},
        'neutral': {'normal':1, 'regular':1, 'aceptable':1, 'satisfactorio':1}
    }
    
    # Calcular puntuaciones
    puntuaciones = {'positivo':0, 'negativo':0, 'neutral':0}
    
    for categoria, palabras in palabras_clave.items():
        for palabra, peso in palabras.items():
            if palabra in texto:
                puntuaciones[categoria] += peso
    
    # Determinar resultado
    max_cat = max(puntuaciones, key=puntuaciones.get)
    return max_cat.capitalize(), puntuaciones[max_cat]

def generar_resumen(texto):
    """Genera un resumen básico del texto"""
    oraciones = nltk.sent_tokenize(texto)
    if len(oraciones) >= 2:
        return oraciones[0] + " [...] " + oraciones[-1]
    return texto

def palabras_frecuentes(textos, n=10):
    """Extrae las palabras más frecuentes"""
    todas_palabras = []
    for texto in textos:
        palabras = [p.lower() for p in nltk.word_tokenize(texto) 
                   if p.isalpha() and p not in stopwords.words('spanish')]
        todas_palabras.extend(palabras)
    return Counter(todas_palabras).most_common(n)

# Interfaz de usuario con pestañas
def main():
    tab1, tab2 = st.tabs(["📝 Analizar Nuevo Comentario", "📊 Explorar Opiniones Existentes"])
    
    with tab1:
        st.header("Analiza un Comentario Nuevo")
        nuevo_comentario = st.text_area("Escribe tu opinión aquí:", height=150)
        
        if st.button("Analizar Sentimiento"):
            if nuevo_comentario.strip():
                with st.spinner("Analizando..."):
                    # Análisis de sentimiento
                    sentimiento, puntuacion = analizar_sentimiento(nuevo_comentario)
                    st.success(f"Sentimiento: **{sentimiento}** (Puntuación: {puntuacion})")
                    
                    # Resumen automático
                    resumen = generar_resumen(nuevo_comentario)
                    st.text_area("Resumen:", value=resumen, height=100)
            else:
                st.warning("Por favor escribe un comentario para analizar")
    
    with tab2:
        st.header("Explora las 20 Opiniones")
        opcion = st.radio("Qué análisis deseas ver:",
                         ["🔍 Temas principales", 
                          "📌 Resumen general",
                          "📈 Distribución de sentimientos"])
        
        if opcion == "🔍 Temas principales":
            st.subheader("Palabras más mencionadas")
            palabras = palabras_frecuentes(opiniones)
            
            # Gráfico de barras
            df_palabras = pd.DataFrame(palabras, columns=['Palabra', 'Frecuencia'])
            st.bar_chart(df_palabras.set_index('Palabra'))
            
            # Nube de palabras
            st.subheader("Nube de palabras")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(palabras))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            
        elif opcion == "📌 Resumen general":
            st.subheader("Resumen de las opiniones")
            texto_largo = " ".join(opiniones)
            resumen = generar_resumen(texto_largo)
            st.write(resumen)
            
            # Estadísticas
            st.write("\n**Estadísticas:**")
            palabras = palabras_frecuentes(opiniones, 5)
            st.write("Palabras más usadas:")
            for palabra, freq in palabras:
                st.write(f"- {palabra} ({freq} veces)")
            
        elif opcion == "📈 Distribución de sentimientos":
            st.subheader("Análisis de Sentimientos")
            resultados = [analizar_sentimiento(o)[0] for o in opiniones]
            distribucion = pd.Series(resultados).value_counts()
            
            # Gráfico
            st.bar_chart(distribucion)
            
            # Ejemplos
            st.write("**Ejemplos por categoría:**")
            df_opiniones = pd.DataFrame({'Opinión': opiniones, 'Sentimiento': resultados})
            
            for categoria in ["Positivo", "Neutral", "Negativo"]:
                ejemplos = df_opiniones[df_opiniones['Sentimiento'] == categoria]['Opinión'].head(2)
                if not ejemplos.empty:
                    st.write(f"**{categoria}:**")
                    for ejemplo in ejemplos:
                        st.write(f"- {ejemplo[:100]}...")

if __name__ == "__main__":
    main()
