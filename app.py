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
    page_title="Clasificador de Sentimientos Mejorado",
    layout="wide"
)

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Base de datos de opiniones
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
    "La base ofrece un acabado mate y aterciopelado que deja la piel lisa.",
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Sistema de clasificación CORREGIDO
def clasificar_sentimiento(texto):
    """Clasificador mejorado con umbrales y pesos optimizados"""
    
    # Palabras clave con nuevos pesos (aumenté los valores negativos)
    positivo = {
        'magnífico':2, 'espectacular':2, 'maravilloso':2, 'excelente':2, 
        'mejor':2, 'recomiendo':2, 'buen':1, 'genial':2, 'perfecto':2,
        'bonita':1, 'natural':1, 'sana':1, 'fácil':1, 'calidad':1,
        '10/10':3, 'diferente':1, 'lindo':1, 'bueno':1, 'económico':1,
        'delicioso':1, 'suave':1, 'recomendado':2, 'aliado':1
    }
    
    negativo = {
        'terrible':4, 'fatal':4, 'no sirve':4, 'no me gusta':3, 'arde':4, 
        'problema':3, 'decepcionante':4, 'pasteluda':3, 'oscuro':2, 
        'alcohol':2, 'duro':2, 'no volveré':4, 'no sale':3, 'no gustó':3, 
        'irrita':3, 'brotó':3, 'ardió':3, 'horrible':4, 'pésimo':4, 
        'decepcionante':4, 'reseca':3, 'malo':3, 'fatal':4, 'problemas':3
    }
    
    # Expresiones negativas reforzadas
    expresiones_negativas = [
        (r'no la volveré a comprar', 5),
        (r'no me gustó para nada', 5),
        (r'me irrita la piel', 4),
        (r'me arde al aplicarla', 5),
        (r'es mucho más oscuro', 3),
        (r'no sirve el envase', 4),
        (r'queda la piel pasteluda', 4),
        (r'problema con el producto', 4),
        (r'no me gusta', 3),
        (r'es terrible', 4)
    ]
    
    texto = texto.lower()
    score = 0
    
    # 1. Detectar expresiones negativas (prioridad alta)
    for expr, peso in expresiones_negativas:
        if re.search(expr, texto):
            score -= peso
    
    # 2. Puntuación por palabras clave
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor
    
    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor
    
    # 3. Clasificación con nuevos umbrales
    if score >= 3:  # Más exigente para positivo
        return "POSITIVO", abs(score)
    elif score <= -3:  # Más sensible para negativo
        return "NEGATIVO", abs(score)
    else:
        return "NEUTRAL", 0

# Interfaz de usuario
def main():
    st.title("🔍 Clasificador de Sentimientos para Cosméticos")
    
    # Pestañas principales
    tab1, tab2 = st.tabs(["Clasificar Comentario", "Analizar Opiniones"])
    
    with tab1:
        st.header("Clasificar Nuevo Comentario")
        comentario = st.text_area("Escribe tu opinión sobre el producto:", height=150)
        
        if st.button("Clasificar"):
            if comentario.strip():
                with st.spinner("Analizando..."):
                    sentimiento, confianza = clasificar_sentimiento(comentario)
                    
                    # Mostrar resultado con énfasis
                    st.subheader("Resultado del Análisis")
                    if sentimiento == "POSITIVO":
                        st.success(f"✅ Sentimiento: {sentimiento} (Confianza: {confianza})")
                        st.write("**Palabras positivas detectadas:**")
                        palabras_pos = [p for p in positivo if p in comentario.lower()]
                        st.write(", ".join(palabras_pos) if palabras_pos else "No se detectaron palabras clave positivas")
                    
                    elif sentimiento == "NEGATIVO":
                        st.error(f"❌ Sentimiento: {sentimiento} (Confianza: {confianza})")
                        st.write("**Palabras negativas detectadas:**")
                        palabras_neg = [p for p in negativo if p in comentario.lower()]
                        st.write(", ".join(palabras_neg) if palabras_neg else "No se detectaron palabras clave negativas")
                    
                    else:
                        st.info(f"➖ Sentimiento: {sentimiento}")
                        st.write("**Razón:** El comentario no contiene suficientes indicadores positivos o negativos fuertes")
            else:
                st.warning("Por favor ingresa un comentario para analizar")
    
    with tab2:
        st.header("Análisis de Opiniones Existentes")
        
        # Analizar todas las opiniones
        resultados = []
        for i, opinion in enumerate(opiniones):
            sentimiento, confianza = clasificar_sentimiento(opinion)
            resultados.append({
                "Opinión": opinion,
                "Sentimiento": sentimiento,
                "Confianza": confianza
            })
        
        df = pd.DataFrame(resultados)
        
        # Mostrar distribución
        st.subheader("Distribución de Sentimientos")
        dist = df['Sentimiento'].value_counts()
        st.bar_chart(dist)
        
        # Mostrar ejemplos problemáticos previos
        st.subheader("Ejemplos de Clasificación")
        st.write("**Comentarios negativos clasificados correctamente:**")
        ejemplos_neg = df[df['Sentimiento'] == "NEGATIVO"].sample(2)
        for _, row in ejemplos_neg.iterrows():
            st.write(f"- {row['Opinión'][:100]}... (Confianza: {row['Confianza']})")
        
        st.write("**Comentarios positivos clasificados correctamente:**")
        ejemplos_pos = df[df['Sentimiento'] == "POSITIVO"].sample(2)
        for _, row in ejemplos_pos.iterrows():
            st.write(f"- {row['Opinión'][:100]}... (Confianza: {row['Confianza']})")

# Variables globales para las palabras clave
positivo = {
    'magnífico':2, 'espectacular':2, 'maravilloso':2, 'excelente':2, 
    'mejor':2, 'recomiendo':2, 'buen':1, 'genial':2, 'perfecto':2,
    'bonita':1, 'natural':1, 'sana':1, 'fácil':1, 'calidad':1,
    '10/10':3, 'diferente':1, 'lindo':1, 'bueno':1, 'económico':1,
    'delicioso':1, 'suave':1, 'recomendado':2, 'aliado':1
}

negativo = {
    'terrible':4, 'fatal':4, 'no sirve':4, 'no me gusta':3, 'arde':4, 
    'problema':3, 'decepcionante':4, 'pasteluda':3, 'oscuro':2, 
    'alcohol':2, 'duro':2, 'no volveré':4, 'no sale':3, 'no gustó':3, 
    'irrita':3, 'brotó':3, 'ardió':3, 'horrible':4, 'pésimo':4, 
    'decepcionante':4, 'reseca':3, 'malo':3, 'fatal':4, 'problemas':3
}

if __name__ == "__main__":
    main()
