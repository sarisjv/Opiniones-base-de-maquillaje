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
import gc  # Para liberar memoria

# Configuración inicial
st.set_page_config(
    page_title="Análisis de Comentarios Optimizado",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App optimizada para Render Free Tier"
    }
)

# Descargar recursos de NLTK (solo lo necesario)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Datos de las 20 opiniones
opiniones = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural.",
    # ... (agrega tus 20 opiniones aquí)
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Análisis de sentimiento ligero (sin modelos pesados)
def analizar_sentimiento(texto):
    """Analiza el sentimiento usando un sistema de reglas simple pero efectivo"""
    positivo = ['magnífico', 'espectacular', 'maravilloso', 'excelente', 'recomiendo', 'buen', 'genial']
    negativo = ['terrible', 'fatal', 'no me gusta', 'arde', 'problema', 'decepcionante']
    
    texto = texto.lower()
    puntaje = sum(1 for p in positivo if p in texto) - sum(1 for n in negativo if n in texto)
    
    if puntaje > 0:
        return "Positivo", puntaje
    elif puntaje < 0:
        return "Negativo", abs(puntaje)
    else:
        return "Neutral", 0

# Resumen básico sin modelos grandes
def generar_resumen(texto):
    """Genera un resumen tomando las oraciones más importantes"""
    oraciones = nltk.sent_tokenize(texto)
    if len(oraciones) > 1:
        return oraciones[0] + " [...] " + oraciones[-1]
    return texto

# Interfaz optimizada
def main():
    st.title("📊 Análisis de Opiniones Optimizado")
    
    # Pestañas para organización
    tab1, tab2 = st.tabs(["➕ Analizar Nuevo", "📚 Opiniones Existentes"])
    
    with tab1:
        st.header("Analizar Nuevo Comentario")
        comentario = st.text_area("Escribe tu opinión (máx. 200 caracteres):", 
                                max_chars=200,
                                height=150)
        
        if st.button("Analizar"):
            if comentario.strip():
                with st.spinner("Procesando..."):
                    # Análisis de sentimiento
                    sentimiento, puntaje = analizar_sentimiento(comentario)
                    st.metric("Sentimiento", f"{sentimiento} (Puntaje: {puntaje})")
                    
                    # Resumen
                    st.text_area("Resumen:", 
                                value=generar_resumen(comentario),
                                height=100)
            else:
                st.warning("Por favor escribe un comentario")
    
    with tab2:
        st.header("Explorar 20 Opiniones")
        opcion = st.radio("Seleccione análisis:",
                         ["📋 Tabla completa", 
                          "🔍 Temas principales",
                          "📈 Distribución"])
        
        df = pd.DataFrame({'Opinión': opiniones})
        df['Sentimiento'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[0])
        
        if opcion == "📋 Tabla completa":
            st.dataframe(df)
            
        elif opcion == "🔍 Temas principales":
            # Palabras más frecuentes
            palabras = [p.lower() for o in opiniones 
                       for p in nltk.word_tokenize(o) 
                       if p.isalpha() and p not in stopwords.words('spanish')]
            
            st.subheader("Palabras más frecuentes")
            frecuentes = Counter(palabras).most_common(10)
            st.bar_chart(pd.DataFrame(frecuentes, columns=['Palabra', 'Frecuencia']).set_index('Palabra'))
            
            # Nube de palabras
            st.subheader("Nube de palabras")
            wc = WordCloud(width=600, height=300, background_color='white').generate(" ".join(palabras))
            plt.figure(figsize=(10, 5))
            plt.imshow(wc)
            plt.axis('off')
            st.pyplot(plt, clear_figure=True)
            plt.close()  # Liberar memoria
            
        elif opcion == "📈 Distribución":
            st.subheader("Distribución de Sentimientos")
            distribucion = df['Sentimiento'].value_counts()
            st.bar_chart(distribucion)
            
            # Ejemplos
            st.write("**Ejemplos por categoría:**")
            for cat in ["Positivo", "Neutral", "Negativo"]:
                ejemplos = df[df['Sentimiento'] == cat]['Opinión'].head(2)
                if not ejemplos.empty:
                    st.write(f"**{cat}:**")
                    for e in ejemplos:
                        st.write(f"- {e[:80]}...")
        
        # Liberar memoria explícitamente
        gc.collect()

if __name__ == "__main__":
    main()
