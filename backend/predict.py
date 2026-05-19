import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_model.h5")

model = None

class_names = [
    "acne",
    "eczema",
    "healthy"
]


def get_model():
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
            raise FileNotFoundError(
                "Trained model not found. Add training images, run train.py, "
                "and make sure model/skin_disease_model.h5 is created."
            )

        model = load_model(MODEL_PATH)

    return model


def predict_disease(img_path):
    loaded_model = get_model()

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = loaded_model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = round(
        100 * np.max(prediction),
        2
    )

    return predicted_class, confidence
