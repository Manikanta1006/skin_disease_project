import os
from uuid import uuid4

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from predict import predict_disease

app = Flask(__name__)

CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid4().hex}_{filename}"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    file.save(filepath)

    try:
        disease, confidence = predict_disease(filepath)
    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500

    return jsonify({
        "disease": disease,
        "confidence": f"{confidence}%"
    })


if __name__ == "__main__":
    app.run(debug=True)
