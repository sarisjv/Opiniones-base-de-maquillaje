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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import gc

# Configuración inicial
st.set_page_config(
    page_title="Análisis Completo de Opiniones",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App de análisis de opiniones con múltiples funcionalidades"
    }
)

# Descargar recursos de NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# -------------------------------
# Datos y modelo Naive Bayes
comentarios = [
    "Me encanta este producto, es económico",
    "No me gusta, me reseca mucho la piel",
    "Es un producto más, no está mal",
    "Excelente calidad y precio",
    "Horrible, me irrita la piel",
    "No tengo opinión",
    "No lo volvería a comprar",
    "Lo amo, me deja la piel suave",
    "Pésimo, me ardió la cara",
    "Muy bueno, huele delicioso",
    "No me hizo efecto",
    "Es neutral para mí",
    "Fantástico, super recomendado",
    "Decepcionante, esperaba más",
    "No me gustó para nada",
    "Lo recomiendo totalmente",
    "Es aceptable, nada especial",
    "Una maravilla de producto",
    "Es malo, me brotó la piel",
    "Me agrada, pero no es el mejor"
]

etiquetas = [
    "positivo", "negativo", "neutral", "positivo", "negativo", "neutral",
    "negativo", "positivo", "negativo", "positivo", "neutral", "neutral",
    "positivo", "negativo", "negativo", "positivo", "neutral", "positivo",
    "negativo", "neutral"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comentarios)
modelo_nb = MultinomialNB()
modelo_nb.fit(X, etiquetas)

# -------------------------------
# Opiniones para análisis avanzado
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
    "La base de maquillaje ofrece an acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas, pero muy bueno.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces para mejor cobertura pero ya queda la piel pasteluda.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Función mejorada de análisis de sentimiento
def analizar_sentimiento(texto):
    """Analiza el sentimiento con un sistema de reglas mejorado"""
    # Palabras clave con pesos mejorados
    positivo = {
        'magnífico': 3, 'espectacular': 3, 'maravilloso': 3, 'excelente': 3,
        'mejor': 2, 'recomiendo': 2, 'buen': 2, 'genial': 2, 'perfecto': 3,
        'bonita': 2, 'natural': 1, 'sana': 1, 'fácil': 1, 'calidad': 2,
        '10/10': 3, 'mejor aliado': 2, 'muy bien': 2, 'muy bueno': 2,
        'excelente cobertura': 3, 'no es grasosa': 1, 'diferente': 1,
        'delicioso': 2, 'fantástico': 3, 'super recomendado': 3, 'totalmente': 2,
        'maravilla': 3, 'agrada': 2, 'amo': 3, 'suave': 2, 'minimiza': 2,
        'imperfecciones': 1, 'acabado': 1, 'lindo': 2, 'aterciopelado': 2,
        'lisa': 1, 'desempeño': 1, 'rendimiento': 1, 'aliado': 2
    }
    
    negativo = {
        'terrible': 3, 'fatal': 3, 'no sirve': 3, 'no me gusta': 2,
        'arde': 3, 'problema': 2, 'decepcionante': 3, 'pasteluda': 2,
        'oscuro': 1, 'alcohol': 1, 'duro': 1, 'no volveré': 3,
        'no sale': 2, 'fatal': 3, 'no gustó': 2, 'no gusta': 2,
        'pensé que sería mejor': 2, 'queda pasteluda': 2, 'arde': 3,
        'no la volveré': 3, 'irrita': 2, 'problemas': 2, 'horrible': 3,
        'reseca': 2, 'pésimo': 3, 'ardió': 3, 'brotó': 2, 'decepcionante': 3,
        'esperaba más': 2, 'nada especial': 1, 'malo': 2, 'huele a alcohol': 2,
        'no lo compraría': 3, 'no lo recomiendo': 3, 'no es para nada': 2,
        'no me hizo efecto': 2, 'no está mal': 1, 'no es el mejor': 2,
        'sensación no me gusta': 3, 'arruinó': 3, 'dañó': 3, 'irritación': 3
    }
    
    # Expresiones negativas completas
    expresiones_negativas = [
        r'no la volveré a comprar',
        r'no me gustó',
        r'no me gusta',
        r'es mucho más oscuro',
        r'queda la piel pasteluda',
        r'me arde al aplicarla',
        r'no sirve el envase',
        r'problema con el producto',
        r'no lo volvería a comprar',
        r'pésimo, me ardió la cara',
        r'horrible, me irrita la piel',
        r'decepcionante, esperaba más',
        r'no me gustó para nada',
        r'es malo, me brotó la piel',
        r'la sensación en la piel no me gusta'
    ]
    
    # Expresiones positivas completas
    expresiones_positivas = [
        r'me encanta este producto',
        r'excelente calidad y precio',
        r'lo amo, me deja la piel suave',
        r'muy bueno, huele delicioso',
        r'lo recomiendo totalmente',
        r'una maravilla de producto',
        r'fantástico, super recomendado',
        r'minimiza imperfecciones con una sola aplicación',
        r'mejor aliado del día a día',
        r'excelente cobertura y precio'
    ]
    
    texto = texto.lower()
    score = 0
    
    # Detectar expresiones positivas completas
    for expr in expresiones_positivas:
        if re.search(expr, texto):
            score += 4
    
    # Detectar expresiones negativas completas
    for expr in expresiones_negativas:
        if re.search(expr, texto):
            score -= 4
    
    # Puntuación positiva
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor
    
    # Puntuación negativa
    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor
    
    # Determinar resultado con umbrales ajustados
    if score >= 2:
        return "Positivo", score
    elif score <= -2:
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

# Función para generar resumen
def generar_resumen(texto):
    """Genera un resumen básico del texto"""
    oraciones = nltk.sent_tokenize(texto)
    if len(oraciones) > 1:
        return oraciones[0] + " [...] " + oraciones[-1]
    return texto

# Función para extraer palabras clave
def palabras_clave(textos, n=10):
    """Extrae las palabras más frecuentes excluyendo stopwords"""
    palabras_comunes = {'producto', 'base', 'maquillaje', 'piel', 'buen', 'como', 'que', 'con', 'para'}
    palabras = []
    
    for texto in textos:
        tokens = [p.lower() for p in nltk.word_tokenize(texto) 
                 if p.isalpha() and p not in stopwords.words('spanish') 
                 and p.lower() not in palabras_comunes]
        palabras.extend(tokens)
    
    return Counter(palabras).most_common(n)

def main():
    st.title("💬 Análisis Completo de Opiniones de Productos Cosméticos")

    tab_nb, tab_avanzado = st.tabs(["Análisis rápido", "Análisis avanzado y exploración"])

    # Pestaña 1: Naive Bayes clásico
    with tab_nb:
        st.header("Análisis rápido")
        comentario_usuario = st.text_area("Escribe tu comentario aquí:")
        if st.button("Analizar Sentimiento"):
            if comentario_usuario.strip() == "":
                st.warning("Por favor escribe un comentario antes de analizar.")
            else:
                try:
                    comentario_vectorizado = vectorizer.transform([comentario_usuario])
                    prediccion = modelo_nb.predict(comentario_vectorizado)[0]
                    st.success(f"Sentimiento detectado: **{prediccion.upper()}**")

                    proba = modelo_nb.predict_proba(comentario_vectorizado)[0]
                    for etiqueta, prob in zip(modelo_nb.classes_, proba):
                        st.write(f"{etiqueta.capitalize()}: {prob:.2f}")
                except Exception as e:
                    st.error(f"Error al analizar: {str(e)}")

    # Pestaña 2: análisis avanzado
    with tab_avanzado:
        st.header("Análisis avanzado de opiniones y exploración de datos")

        subtab1, subtab2 = st.tabs(["Analizar nuevo comentario", "Explorar opiniones existentes"])

        with subtab1:
            comentario = st.text_area("Escribe tu opinión sobre el producto:", height=150, key="opinion_input")
            if st.button("Analizar Sentimiento (Avanzado)", key="analyze_btn"):
                if comentario.strip():
                    with st.spinner("Analizando..."):
                        try:
                            sentimiento, puntaje = analizar_sentimiento(comentario)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Resultado del Análisis")
                                if sentimiento == "Positivo":
                                    st.success(f"✅ {sentimiento} (Puntaje: {puntaje})")
                                elif sentimiento == "Negativo":
                                    st.error(f"❌ {sentimiento} (Puntaje: {puntaje})")
                                else:
                                    st.info(f"➖ {sentimiento}")

                            with col2:
                                st.subheader("Resumen Automático")
                                resumen = generar_resumen(comentario)
                                st.text_area("Resumen", value=resumen, height=100, key="resumen_area")
                        except Exception as e:
                            st.error(f"Error en el análisis: {str(e)}")
                else:
                    st.warning("Por favor escribe un comentario para analizar")

        with subtab2:
            st.header("Análisis de las 20 Opiniones")

            opcion = st.radio("Seleccione el tipo de análisis:",
                             ["Ver todas las opiniones", "Temas principales", "Distribución de sentimientos"],
                             key="analysis_option")

            df = pd.DataFrame({'Opinión': opiniones})
            df['Sentimiento'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[0])
            df['Puntaje'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[1])

            if opcion == "Ver todas las opiniones":
                st.subheader("Tabla Completa de Opiniones")
                st.dataframe(df)

            elif opcion == "Temas principales":
                st.subheader("Palabras Clave Más Frecuentes")
                palabras = palabras_clave(opiniones)
                st.bar_chart(pd.DataFrame(palabras, columns=['Palabra', 'Frecuencia']).set_index('Palabra'))
                
                st.subheader("Nube de Palabras")
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(opiniones))
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                    plt.close()
                except Exception as e:
                    st.error(f"Error al generar nube de palabras: {str(e)}")

            elif opcion == "Distribución de sentimientos":
                st.subheader("Distribución de Sentimientos")
                distribucion = df['Sentimiento'].value_counts()
                st.bar_chart(distribucion)
                
                st.write("**Ejemplos por categoría:**")
                for categoria in ["Positivo", "Neutral", "Negativo"]:
                    ejemplos = df[df['Sentimiento'] == categoria]['Opinión'].head(2)
                    if not ejemplos.empty:
                        st.write(f"**{categoria}:**")
                        for e in ejemplos:
                            st.write(f"- {e[:100]}...")

if __name__ == "__main__":
    main()
