from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
import re  # Necesario para detectar expresiones negativas

app = Flask(__name__)

# Palabras positivas y negativas con sus valores
positivo = {
    "bueno": 1,
    "excelente": 2,
    "maravilloso": 2,
    "agradable": 1,
    "efectivo": 2,
    "recomiendo": 1,
    "perfecto": 2,
    "me encanta": 2,
    "me gustó": 1,
    "funciona": 1,
    "cumple": 1
}

negativo = {
    "malo": 1,
    "horrible": 2,
    "terrible": 2,
    "irritación": 2,
    "ardor": 2,
    "ineficaz": 2,
    "decepcionante": 2,
    "me irritó": 2,
    "no sirve": 2,
    "pasteludo": 1,
    "manchó": 1,
    "granos": 1,
    "oscuro": 1
}

# Nuevas expresiones negativas completas
expresiones_negativas = [
    r'no la volver[é|e] a comprar',
    r'no me gust[oó]',
    r'no me gusta',
    r'no sirve',
    r'es mucho más oscuro',
    r'queda la piel pasteluda',
    r'me ardi[oó] la piel',
    r'me irrit[oó]',
    r'horrible',
    r'decepcionante',
    r'no funciona',
    r'me salieron granos',
    r'no me hizo efecto'
]

comentarios = []

def analizar_sentimiento(texto):
    texto = texto.lower()
    score = 0

    # Detectar expresiones negativas completas
    for expr in expresiones_negativas:
        if re.search(expr, texto):
            score -= 3

    # Palabras positivas
    for palabra, valor in positivo.items():
        if palabra in texto:
            score += valor

    # Palabras negativas
    for palabra, valor in negativo.items():
        if palabra in texto:
            score -= valor

    if score >= 3:
        return "Positivo", score
    elif score <= -2:
        return "Negativo", abs(score)
    else:
        return "Neutral", 0

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        comentario = request.form["comentario"]
        sentimiento, puntuacion = analizar_sentimiento(comentario)
        comentarios.append({"comentario": comentario, "sentimiento": sentimiento})
        resultado = f"Sentimiento del comentario: {sentimiento} (Puntuación: {puntuacion})"
    return render_template("index.html", resultado=resultado, comentarios=comentarios)

@app.route("/grafica")
def grafica():
    if not comentarios:
        return "No hay comentarios aún."

    df = pd.DataFrame(comentarios)
    conteo = df["sentimiento"].value_counts()

    plt.figure(figsize=(6, 4))
    colores = {"Positivo": "green", "Negativo": "red", "Neutral": "gray"}
    conteo.plot(kind="bar", color=[colores.get(sent, "blue") for sent in conteo.index])
    plt.title("Distribución de Sentimientos")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)

    ruta_imagen = "static/grafica.png"
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()

    return render_template("grafica.html", imagen=ruta_imagen)

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
