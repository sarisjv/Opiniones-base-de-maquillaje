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
st.set_page_config(
    page_title="Análisis de Sentimientos Mejorado",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App con análisis de sentimientos mejorado"
    }
)

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Todas las opiniones
opiniones = [
    # ... (tus 20 opiniones aquí)
]

# Función MEJORADA de análisis de sentimiento
def analizar_sentimiento(texto):
    """Analiza el sentimiento con un sistema de reglas mejorado"""
    # Palabras clave con pesos mejorados
    positivo = {
        'magnífico': 3, 'espectacular': 3, 'maravilloso': 3, 'excelente': 3,
        'mejor': 2, 'recomiendo': 2, 'buen': 2, 'genial': 2, 'perfecto': 3,
        'bonita': 2, 'natural': 1, 'sana': 1, 'fácil': 1, 'calidad': 2,
        '10/10': 3, 'mejor aliado': 2, 'muy bien': 2, 'muy bueno': 2,
        'excelente cobertura': 3, 'no es grasosa': 1
    }
    
    negativo = {
        'terrible': 3, 'fatal': 3, 'no sirve': 3, 'no me gusta': 2,
        'arde': 3, 'problema': 2, 'decepcionante': 3, 'pasteluda': 2,
        'oscuro': 1, 'alcohol': 1, 'duro': 1, 'no volveré': 3,
        'no sale': 2, 'fatal': 3, 'no gustó': 2, 'no gusta': 2,
        'pensé que sería mejor': 2, 'queda pasteluda': 2, 'arde': 3
    }
    
    # Expresiones negativas completas
    expresiones_negativas = [
        r'no la volveré a comprar',
        r'no me gustó',
        r'no me gusta',
        r'es mucho más oscuro',
        r'queda la piel pasteluda',
        r'me arde al aplicarla'
    ]
    
    texto = texto.lower()
    score = 0
    
    # Detectar expresiones negativas completas (más confiables)
    for expr in expresiones_negativas:
        if re.search(expr, texto):
            score -= 3  # Fuerte indicador negativo
    
    # Puntuación positiva
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor
    
    # Puntuación negativa
    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor
    
    # Determinar resultado con umbrales ajustados
    if score >= 3:  # Umbral más alto para positivo
        return "Positivo", score
    elif score <= -2:  # Umbral más bajo para negativo
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

# Resto del código igual que antes...
def generar_resumen(texto):
    # ... (igual que antes)

def palabras_clave(textos, n=10):
    # ... (igual que antes)

def main():
    st.title("💬 Analizador de Opiniones Mejorado")
    
    # Pestañas principales
    tab1, tab2 = st.tabs(["➕ Analizar Nuevo Comentario", "📊 Opiniones Existentes"])
    
    with tab1:
        st.header("Analizar Comentario Nuevo")
        comentario = st.text_area("Escribe tu comentario:", height=150)
        
        if st.button("Analizar"):
            if comentario.strip():
                with st.spinner("Analizando..."):
                    sentimiento, puntaje = analizar_sentimiento(comentario)
                    
                    # Mostrar resultado con colores
                    if sentimiento == "Positivo":
                        st.success(f"🔍 Resultado: {sentimiento} (Puntaje: {puntaje})")
                    elif sentimiento == "Negativo":
                        st.error(f"🔍 Resultado: {sentimiento} (Puntaje: {puntaje})")
                    else:
                        st.info(f"🔍 Resultado: {sentimiento} (Puntaje: {puntaje})")
                    
                    # Mostrar razones del análisis (DEBUG - opcional)
                    st.write("**Palabras clave detectadas:**")
                    palabras = nltk.word_tokenize(comentario.lower())
                    st.write(", ".join(set(palabras) & (
                        set(positivo.keys()) | 
                        set(negativo.keys()) |
                        set(" ".join(expresiones_negativas).split())
                    ))
                    
                    # Resumen
                    st.text_area("Resumen:", value=generar_resumen(comentario), height=100)
            else:
                st.warning("Por favor escribe un comentario")
    
    with tab2:
        # ... (igual que antes)

if __name__ == "__main__":
    main()
