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
import re

# Configuración inicial
st.set_page_config(
    page_title="Análisis de Sentimientos Cosméticos",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "Clasificador de sentimientos para comentarios de productos cosméticos"
    }
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

# Sistema de clasificación mejorado
def clasificar_sentimiento(texto):
    """Clasifica el comentario en Positivo, Negativo o Neutral con mayor precisión"""
    
    # Diccionario mejorado de palabras clave
    positivo = {
        'magnífico':3, 'espectacular':3, 'maravilloso':3, 'excelente':3, 'mejor':2,
        'recomiendo':2, 'buen':2, 'genial':2, 'perfecto':3, 'bonita':2, 'natural':1,
        'sana':1, 'fácil':1, 'calidad':2, '10/10':3, 'diferente':1, 'lindo':2,
        'bueno':2, 'económico':1, 'delicioso':1, 'suave':1, 'recomendado':2
    }
    
    negativo = {
        'terrible':3, 'fatal':3, 'no sirve':3, 'no me gusta':2, 'arde':3, 'problema':2,
        'decepcionante':3, 'pasteluda':2, 'oscuro':1, 'alcohol':1, 'duro':1, 'no volveré':3,
        'no sale':2, 'no gustó':2, 'irrita':2, 'brotó':2, 'ardió':2, 'horrible':3,
        'pésimo':3, 'decepcionante':3, 'reseca':2, 'malo':2
    }
    
    # Expresiones completas (más precisas)
    expresiones_positivas = [
        r'es la mejor', r'lo recomiendo', r'excelente calidad', r'me encanta',
        r'me deja la piel suave', r'super recomendado', r'funciona maravillosamente'
    ]
    
    expresiones_negativas = [
        r'no la volveré a comprar', r'no me gustó para nada', r'me irrita la piel',
        r'me arde al aplicarla', r'es mucho más oscuro', r'no sirve el envase',
        r'queda la piel pasteluda', r'problema con el producto'
    ]
    
    texto = texto.lower()
    score = 0
    
    # Verificar expresiones completas primero (son más confiables)
    for expr in expresiones_positivas:
        if re.search(expr, texto):
            score += 4
    
    for expr in expresiones_negativas:
        if re.search(expr, texto):
            score -= 4
    
    # Puntuación por palabras clave
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor
    
    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor
    
    # Clasificación final con umbrales ajustados
    if score >= 4:
        return "POSITIVO", score
    elif score <= -3:
        return "NEGATIVO", abs(score)
    else:
        return "NEUTRAL", 0

# Funciones auxiliares
def generar_resumen(texto):
    oraciones = nltk.sent_tokenize(texto)
    if len(oraciones) > 1:
        return f"{oraciones[0]} [...] {oraciones[-1]}"
    return texto

def palabras_clave(textos, n=10):
    stop_words = set(stopwords.words('spanish'))
    palabras_comunes = {'producto', 'base', 'maquillaje', 'piel', 'como', 'que'}
    palabras = [p.lower() for texto in textos 
               for p in nltk.word_tokenize(texto) 
               if p.isalpha() and p not in stop_words and p.lower() not in palabras_comunes]
    return Counter(palabras).most_common(n)

# Interfaz de usuario
def main():
    st.title("💄 Analizador de Comentarios Cosméticos")
    
    tab1, tab2 = st.tabs(["Clasificar Nuevo Comentario", "Analizar Opiniones Existentes"])
    
    with tab1:
        st.header("Clasificar Sentimiento de Nuevo Comentario")
        comentario = st.text_area("Escribe tu comentario sobre el producto:", height=150)
        
        if st.button("Clasificar Sentimiento"):
            if comentario.strip():
                with st.spinner("Analizando..."):
                    sentimiento, puntaje = clasificar_sentimiento(comentario)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Resultado")
                        if sentimiento == "POSITIVO":
                            st.success(f"✅ {sentimiento} (Confianza: {puntaje})")
                        elif sentimiento == "NEGATIVO":
                            st.error(f"❌ {sentimiento} (Confianza: {puntaje})")
                        else:
                            st.info(f"➖ {sentimiento}")
                    
                    with col2:
                        st.subheader("Resumen")
                        st.text_area(" ", value=generar_resumen(comentario), height=100)
            else:
                st.warning("Por favor ingresa un comentario")
    
    with tab2:
        st.header("Análisis de las 20 Opiniones de Ejemplo")
        
        # Crear DataFrame con análisis
        df = pd.DataFrame({'OPINIÓN': opiniones})
        df[['SENTIMIENTO', 'CONFIANZA']] = df['OPINIÓN'].apply(
            lambda x: pd.Series(clasificar_sentimiento(x))
        )
        
        opcion = st.radio("Seleccione el análisis:",
                         ["Ver todas las opiniones", 
                          "Distribución de sentimientos", 
                          "Temas principales"])
        
        if opcion == "Ver todas las opiniones":
            st.dataframe(df)
        
        elif opcion == "Distribución de sentimientos":
            st.subheader("Distribución de Sentimientos")
            distribucion = df['SENTIMIENTO'].value_counts()
            st.bar_chart(distribucion)
            
            st.write("**Ejemplos por categoría:**")
            for cat in ["POSITIVO", "NEUTRAL", "NEGATIVO"]:
                ejemplos = df[df['SENTIMIENTO'] == cat]['OPINIÓN'].head(2)
                if not ejemplos.empty:
                    st.write(f"**{cat.capitalize()}:**")
                    for e in ejemplos:
                        st.write(f"- {e[:100]}...")
        
        elif opcion == "Temas principales":
            st.subheader("Palabras Clave Más Frecuentes")
            palabras = palabras_clave(opiniones)
            st.bar_chart(pd.DataFrame(palabras, columns=['Palabra', 'Frecuencia']).set_index('Palabra'))
            
            st.subheader("Nube de Palabras")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(opiniones))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

if __name__ == "__main__":
    main()
