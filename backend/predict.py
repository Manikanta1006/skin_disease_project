import json
import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
MIN_CONFIDENCE = 60.0

model = None
class_names = None


def get_class_names():
    global class_names

    if class_names is None:
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(
                "Class names not found. Run train.py again so "
                "model/class_names.json is created."
            )

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)

    return class_names


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
    labels = get_class_names()

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = loaded_model.predict(img_array)

    predicted_index = np.argmax(prediction)

    if predicted_index >= len(labels):
        raise ValueError(
            "Model output does not match class_names.json. "
            "Run train.py again to regenerate both files."
        )

    predicted_class = labels[predicted_index]

    confidence = round(
        100 * np.max(prediction),
        2
    )

    if confidence < MIN_CONFIDENCE:
        predicted_class = "uncertain"

    return predicted_class, confidence
