import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Título de la aplicación
st.title("Análisis de Sentimientos de Productos Cosméticos")
st.write("Escribe tu opinión sobre el producto y clasificaré el sentimiento como positivo, negativo o neutral.")

# Datos de entrenamiento ampliados
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

# Vectorización
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comentarios)

# Entrenamiento del modelo
modelo = MultinomialNB()
modelo.fit(X, etiquetas)

# Entrada del usuario
comentario_usuario = st.text_area("Escribe tu comentario aquí:")

if st.button("Analizar Sentimiento"):
    if comentario_usuario.strip() == "":
        st.warning("Por favor escribe un comentario antes de analizar.")
    else:
        comentario_vectorizado = vectorizer.transform([comentario_usuario])
        prediccion = modelo.predict(comentario_vectorizado)[0]
        st.success(f"Sentimiento detectado: **{prediccion.upper()}**")

        # Mostrar probabilidades (opcional)
        proba = modelo.predict_proba(comentario_vectorizado)[0]
        for etiqueta, prob in zip(modelo.classes_, proba):
            st.write(f"{etiqueta.capitalize()}: {prob:.2f}")

