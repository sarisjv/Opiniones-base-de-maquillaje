import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Configuración inicial
st.set_page_config(page_title="Análisis Completo de Opiniones", layout="wide")
st.title("📊 Análisis de 20 Opiniones de Clientes")

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Datos de las 20 opiniones
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
    "La base de maquillaje ofrece un acabado mate y aterciopelado que deja la piel lisa.",
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

# Función mejorada de análisis de sentimiento
def analizar_sentimiento(texto):
    texto = texto.lower()
    
    # Palabras clave para cada categoría
    positivo = ['magnífico', 'espectacular', 'maravilloso', 'excelente', 'mejor', 'buen', 'recomiendo', 'genial', 'perfecto']
    negativo = ['terrible', 'fatal', 'no sirve', 'no me gusta', 'arde', 'problema', 'decepcionante']
    neutral = ['normal', 'regular', 'aceptable', 'satisfactorio']
    
    # Contar ocurrencias
    pos = sum(texto.count(p) for p in positivo)
    neg = sum(texto.count(n) for n in negativo)
    neu = sum(texto.count(n) for n in neutral)
    
    # Determinar resultado
    if pos > neg and pos > neu:
        return "Positivo", pos
    elif neg > pos and neg > neu:
        return "Negativo", neg
    else:
        return "Neutral", neu

# Interfaz mejorada
def main():
    st.header("Análisis Completo")
    
    # Convertir a DataFrame
    df = pd.DataFrame({'Opinión': opiniones})
    
    # Aplicar análisis a todas las opiniones
    df['Análisis'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[0])
    df['Puntaje'] = df['Opinión'].apply(lambda x: analizar_sentimiento(x)[1])
    
    # Mostrar todas las opiniones con su análisis
    st.subheader("Tabla Completa de Opiniones")
    st.dataframe(df)
    
    # Gráficos de análisis
    st.subheader("Visualizaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de sentimientos
        st.write("**Distribución de Sentimientos**")
        distribucion = df['Análisis'].value_counts()
        st.bar_chart(distribucion)
    
    with col2:
        # Palabras más frecuentes
        st.write("**Palabras Clave**")
        todas_palabras = ' '.join(opiniones).lower()
        palabras = [p for p in nltk.word_tokenize(todas_palabras) 
                   if p.isalpha() and p not in stopwords.words('spanish')]
        frecuentes = Counter(palabras).most_common(10)
        st.write(pd.DataFrame(frecuentes, columns=['Palabra', 'Frecuencia']))
    
    # Mostrar ejemplos de cada categoría
    st.subheader("Ejemplos por Categoría")
    
    for categoria in ["Positivo", "Neutral", "Negativo"]:
        st.write(f"**{categoria}:**")
        ejemplos = df[df['Análisis'] == categoria]['Opinión'].head(3)
        for i, ejemplo in enumerate(ejemplos, 1):
            st.write(f"{i}. {ejemplo[:100]}...")

if __name__ == "__main__":
    main()    
