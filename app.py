import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from collections import Counter

# Configuración básica
st.set_page_config(
    page_title="Analizador de Opiniones",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App optimizada para Render Free Tier"
    }
)

# Descarga mínima de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Datos de ejemplo (20 opiniones)
opiniones = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación.",
    # ... (agrega tus 20 opiniones aquí)
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Funciones optimizadas sin modelos pesados
def analizar_sentimiento(texto):
    """Análisis de sentimiento básico sin modelos externos"""
    positivo = ['excelente', 'magnífico', 'maravilloso', 'recomiendo', 'buen', 'genial']
    negativo = ['terrible', 'decepcionante', 'no me gusta', 'arde', 'problema']
    
    texto = texto.lower()
    puntaje = sum(1 for palabra in positivo if palabra in texto) - \
              sum(1 for palabra in negativo if palabra in texto)
    
    if puntaje > 0:
        return "Positivo", puntaje
    elif puntaje < 0:
        return "Negativo", abs(puntaje)
    else:
        return "Neutral", 0

def generar_resumen(texto):
    """Resumen básico tomando las primeras oraciones"""
    oraciones = nltk.sent_tokenize(texto)
    return " ".join(oraciones[:2]) + "..." if len(oraciones) > 2 else texto

def palabras_clave(textos, n=10):
    """Extrae palabras clave frecuentes"""
    palabras = []
    for texto in textos:
        palabras.extend([p for p in nltk.word_tokenize(texto.lower()) 
                        if p.isalpha() and p not in stopwords.words('spanish')])
    return Counter(palabras).most_common(n)

# Interfaz de usuario
def main():
    st.title("📌 Análisis de Opiniones Optimizado")
    
    # Pestañas principales
    tab1, tab2 = st.tabs(["➕ Analizar Nuevo Comentario", "📊 Explorar Opiniones"])
    
    with tab1:
        st.header("Analizar Comentario Nuevo")
        comentario = st.text_area("Escribe tu opinión (máx. 250 caracteres):", 
                                max_chars=250,
                                height=150)
        
        if st.button("Analizar"):
            with st.spinner("Procesando..."):
                # Análisis de sentimiento
                sentimiento, puntaje = analizar_sentimiento(comentario)
                st.metric("Sentimiento", f"{sentimiento} (Puntaje: {puntaje})")
                
                # Resumen
                st.text_area("Resumen:", 
                            value=generar_resumen(comentario),
                            height=100)
    
    with tab2:
        st.header("Explorar 20 Opiniones de Ejemplo")
        opcion = st.radio("Seleccione análisis:",
                         ["🔍 Ver temas principales", 
                          "📄 Resumen general",
                          "📈 Distribución de sentimientos"])
        
        if opcion == "🔍 Ver temas principales":
            st.subheader("Palabras más frecuentes")
            palabras = palabras_clave(opiniones)
            
            # Gráfico de barras
            df_palabras = pd.DataFrame(palabras, columns=['Palabra', 'Frecuencia'])
            st.bar_chart(df_palabras.set_index('Palabra'))
            
            # Nube de palabras
            wc = WordCloud(width=600, height=300, background_color='white').generate_from_frequencies(dict(palabras))
            plt.figure(figsize=(10, 5))
            plt.imshow(wc)
            plt.axis('off')
            st.pyplot(plt, clear_figure=True)
            
        elif opcion == "📄 Resumen general":
            st.subheader("Resumen de las opiniones")
            texto_largo = " ".join([o[:100] for o in opiniones])  # Limitar longitud
            resumen = generar_resumen(texto_largo)
            st.write(resumen)
            
        elif opcion == "📈 Distribución de sentimientos":
            st.subheader("Análisis de Sentimientos")
            resultados = [analizar_sentimiento(o)[0] for o in opiniones]
            distribucion = pd.Series(resultados).value_counts()
            st.bar_chart(distribucion)
            
            # Ejemplos
            st.write("**Ejemplos por categoría:**")
            for cat in ["Positivo", "Neutral", "Negativo"]:
                ejemplos = [o for o in opiniones if analizar_sentimiento(o)[0] == cat][:2]
                if ejemplos:
                    st.write(f"**{cat}:**")
                    for e in ejemplos:
                        st.write(f"- {e[:80]}...")

if __name__ == "__main__":
    from nltk.corpus import stopwords
    main()
