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

# [Resto de las configuraciones iniciales y descargas permanecen igual...]

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
            score += 4  # Fuerte indicador positivo
    
    # Detectar expresiones negativas completas
    for expr in expresiones_negativas:
        if re.search(expr, texto):
            score -= 4  # Fuerte indicador negativo
    
    # Puntuación positiva
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor
    
    # Puntuación negativa
    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor
    
    # Determinar resultado con umbrales ajustados
    if score >= 2:  # Umbral más bajo para positivo
        return "Positivo", score
    elif score <= -2:  # Umbral más bajo para negativo
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

# [El resto del código permanece igual...]
