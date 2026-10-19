import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

MODEL_PATH = "models/modelo_final.keras"
IMG_SIZE = 128
THRESHOLD = 0.380  # el umbral de piso de recall que ya validamos en test

# Se ejecuta una sola vez, al arrancar el servidor
model = load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.array(image)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def predict(image: Image.Image) -> dict:
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    probability_carcinoma = float(prediction[0][0])
    probability_normal = 1 - probability_carcinoma

    if probability_carcinoma > THRESHOLD:
        predicted_class = "carcinoma"
    else:
        predicted_class = "normal"

    return {
        "predicted_class": predicted_class,
        "probability_carcinoma": round(probability_carcinoma, 4),
        "probability_normal": round(probability_normal, 4),
    }