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
import gc

# Configuración inicial
st.set_page_config(
    page_title="Análisis Completo de Opiniones",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/tu-repo',
        'About': "App de análisis de opiniones con todas las funcionalidades"
    }
)

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Opiniones iniciales
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
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
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
    positivo = {
        'magnífico': 3, 'espectacular': 3, 'maravilloso': 3, 'excelente': 3,
        'mejor': 2, 'recomiendo': 2, 'buen': 2, 'genial': 2, 'perfecto': 3,
        'bonita': 2, 'natural': 1, 'sana': 1, 'fácil': 1, 'calidad': 2
    }

    negativo = {
        'terrible': 3, 'fatal': 3, 'no sirve': 3, 'no me gusta': 3,
        'arde': 3, 'problema': 2, 'decepcionante': 3, 'pasteluda': 2,
        'oscuro': 1, 'alcohol': 1, 'duro': 1, 'no volveré': 3
    }

    texto = texto.lower()
    score = 0

    # Primero, evaluar frases negativas completas
    for frase, valor in negativo.items():
        if frase in texto:
            score -= valor

    # Luego, evaluar palabras individuales positivas (si no estaban ya en una frase negativa)
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor

    # Clasificación final
    if score > 2:
        return "Positivo", score
    elif score < -2:
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

# Función para generar resumen
def generar_resumen(texto):
    oraciones = nltk.sent_tokenize(texto)
    if len(oraciones) > 1:
        return f"{oraciones[0]} [...] {oraciones[-1]}"
    return texto

# Función para extraer palabras clave
def palabras_clave(textos, n=10):
    palabras_comunes = {'producto', 'base', 'maquillaje', 'piel', 'buen', 'como'}
    palabras = []

    for texto in textos:
        tokens = [p.lower() for p in nltk.word_tokenize(texto)
                  if p.isalpha() and p not in stopwords.words('spanish')
                  and p.lower() not in palabras_comunes]
        palabras.extend(tokens)

    return Counter(palabras).most_common(n)

# Interfaz principal
def main():
    st.title("💬 Análisis Completo de 20 Opiniones")

    tab1, tab2 = st.tabs(["➕ Analizar Nuevo Comentario", "📊 Explorar Opiniones Existentes"])

    with tab1:
        st.header("Analizar Comentario Nuevo")
        comentario = st.text_area("Escribe tu opinión sobre el producto:", height=150)

        if st.button("Analizar Sentimiento"):
            if comentario.strip():
                with st.spinner("Analizando..."):
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
                        st.text_area(" ", value=resumen, height=100)
            else:
                st.warning("Por favor escribe un comentario para analizar")

    with tab2:
        st.header("Análisis de las 20 Opiniones")
        opcion = st.radio("Seleccione el tipo de análisis:",
                          ["🔍 Ver todas las opiniones",
                           "📊 Temas principales",
                           "📈 Distribución de sentimientos"])

        df = pd.DataFrame({'Opinión': opiniones})
        df['Sentimiento'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[0])
        df['Puntaje'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[1])

        if opcion == "🔍 Ver todas las opiniones":
            st.subheader("Tabla Completa de Opiniones")
            st.dataframe(df)

        elif opcion == "📊 Temas principales":
            st.subheader("Análisis de Temas Principales")

            filtro = st.selectbox("Filtrar por:",
                                  ["Todos los comentarios",
                                   "Solo positivos",
                                   "Solo negativos",
                                   "Solo neutrales"])

            if filtro == "Todos los comentarios":
                textos = opiniones
            else:
                tipo = filtro.split()[-1][:-1]
                textos = df[df['Sentimiento'] == tipo.capitalize()]['Opinión'].tolist()

            st.write("**Palabras clave más relevantes:**")
            palabras = palabras_clave(textos)
            for i, (palabra, freq) in enumerate(palabras, 1):
                st.write(f"{i}. {palabra.capitalize()} (aparece {freq} veces)")

            st.subheader("Nube de Palabras")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(textos))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            plt.close()

        elif opcion == "📈 Distribución de sentimientos":
            st.subheader("Distribución de Sentimientos")

            distribucion = df['Sentimiento'].value_counts()
            st.bar_chart(distribucion)

            st.write("**Resumen estadístico:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total opiniones", len(df))
            col2.metric("Opiniones positivas", distribucion.get("Positivo", 0))
            col3.metric("Opiniones negativas", distribucion.get("Negativo", 0))

            st.subheader("Ejemplos Representativos")
            for categoria in ["Positivo", "Neutral", "Negativo"]:
                ejemplos = df[df['Sentimiento'] == categoria]['Opinión'].head(2)
                if not ejemplos.empty:
                    st.write(f"**{categoria}:**")
                    for ejemplo in ejemplos:
                        st.write(f"- {ejemplo[:100]}...")

        gc.collect()

if __name__ == "__main__":
    main()
