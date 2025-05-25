import streamlit as st
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Configuración mínima
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("💄 Análisis de Opiniones")

# Descarga solo lo esencial de NLTK
nltk.download('punkt', quiet=True)

# Datos de ejemplo optimizados
opiniones = [
    "Base excelente, cubre bien sin ser pesada",
    "No me gustó el tono, muy oscuro para mi piel",
    "Textura ligera pero buena cobertura",
    "El empaque no es práctico",
    "Queda natural y dura todo el día"
]

# Función de análisis ligera
def analizar(texto):
    positivo = len(re.findall(r'excelente|buen[a]?|natural|perfect', texto.lower()))
    negativo = len(re.findall(r'no me gustó|problema|oscuro|no es', texto.lower()))
    return "👍 Positivo" if positivo > negativo else "👎 Negativo" if negativo > positivo else "😐 Neutral"

# Interfaz simplificada
def main():
    # Sidebar para nuevas opiniones
    with st.sidebar:
        nueva = st.text_area("✍️ Añade tu opinión:")
        if st.button("Analizar"):
            if nueva:
                resultado = analizar(nueva)
                st.success(f"Resultado: {resultado}")
                opiniones.append(nueva)
            else:
                st.warning("Escribe una opinión primero")

    # Mostrar análisis
    st.header("📊 Resumen de Opiniones")
    df = pd.DataFrame({
        "Opinión": opiniones,
        "Análisis": [analizar(op) for op in opiniones]
    })
    st.dataframe(df, height=300)

    # Palabras frecuentes (sin NLTK)
    palabras = re.findall(r'\b\w{4,}\b', ' '.join(opiniones).lower())
    comunes = Counter(palabras).most_common(5)
    st.subheader("🔠 Palabras más usadas")
    st.write(", ".join([f"{palabra} ({count})" for palabra, count in comunes]))

if __name__ == "__main__":
    main()
