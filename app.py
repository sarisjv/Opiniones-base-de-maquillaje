import streamlit as st
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configuración básica
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("📊 Análisis de Opiniones sobre Bases de Maquillaje")

# Descargar datos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Datos de ejemplo
opiniones = [
    "Base excelente, cubre bien sin ser pesada",
    "No me gustó el tono, muy oscuro para mi piel",
    "Textura ligera pero buena cobertura",
    "El empaque no es práctico, se derrama",
    "Queda natural y dura todo el día"
]

# Inicializar en session state
if 'opiniones' not in st.session_state:
    st.session_state.opiniones = opiniones.copy()

# Funciones de análisis
def limpiar_texto(texto):
    texto = texto.lower()
    return re.sub(r'[^\w\sáéíóúñ]', '', texto)

def analizar_sentimiento(texto):
    positivas = ['excelente', 'buen', 'buena', 'perfecto', 'natural']
    negativas = ['no me gustó', 'problema', 'derrama', 'oscuro']
    texto = limpiar_texto(texto)
    
    if any(palabra in texto for palabra in positivas):
        return "Positivo"
    elif any(palabra in texto for palabra in negativas):
        return "Negativo"
    return "Neutral"

# Interfaz principal
def main():
    # Sidebar para nuevas opiniones
    with st.sidebar:
        st.header("➕ Nueva Opinión")
        nueva = st.text_area("Escribe tu opinión:")
        if st.button("Agregar"):
            if nueva:
                st.session_state.opiniones.append(nueva)
                st.success("¡Opinión agregada!")
            else:
                st.warning("Escribe una opinión primero")

    # Mostrar todas las opiniones
    st.header("📝 Todas las Opiniones")
    df = pd.DataFrame({
        "N°": range(1, len(st.session_state.opiniones)+1),
        "Opinión": st.session_state.opiniones,
        "Sentimiento": [analizar_sentimiento(op) for op in st.session_state.opiniones]
    })
    st.dataframe(df, height=400, use_container_width=True)

    # Análisis
    st.header("🔍 Análisis")
    
    # Palabras frecuentes
    todas_opiniones = " ".join(st.session_state.opiniones)
    palabras = [word for word in word_tokenize(limpiar_texto(todas_opiniones)) 
               if word not in stopwords.words('spanish') and len(word) > 3]
    contador = Counter(palabras)
    
    st.subheader("📈 Palabras más usadas")
    st.bar_chart(pd.DataFrame(contador.most_common(10), x=0, y=1)

    # Distribución de sentimientos
    st.subheader("😃😐😠 Sentimientos")
    dist = df['Sentimiento'].value_counts()
    st.bar_chart(dist)

if __name__ == "__main__":
    main()
