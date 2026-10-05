
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import base64

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "fruit_model.keras")
CLASS_PATH = os.path.join("models", "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)


def predict_image(image):

    resized = cv2.resize(image, IMG_SIZE)
    x = np.expand_dims(resized, axis=0)

    predictions = model.predict(x, verbose=0)[0]

    indices = np.argsort(predictions)[::-1][:3]

    results = []

    for i in indices:
        name = class_names[i].replace(" fruit", "").title()
        confidence = float(predictions[i]) * 100

        results.append({
            "name": name,
            "confidence": round(confidence, 2)
        })

    return results


@app.route("/")
def home():
    return render_template(
        "index.html",
        classes=class_names
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    data = np.frombuffer(
        file.read(),
        np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = predict_image(image)

    return jsonify({
        "prediction": results[0]["name"],
        "confidence": results[0]["confidence"],
        "top3": results
    })


@app.route("/enhance", methods=["POST"])
def enhance():

    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]

    method = request.form.get(
        "method",
        "Original"
    )

    data = np.frombuffer(
        file.read(),
        np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if method == "Average Blur":
        image = cv2.blur(image, (7, 7))

    elif method == "Gaussian Blur":
        image = cv2.GaussianBlur(
            image,
            (7, 7),
            0
        )

    elif method == "Edge Detection":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        image = cv2.Canny(
            gray,
            100,
            200
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    elif method == "Sharpening":

        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        image = cv2.filter2D(
            image,
            -1,
            kernel
        )

    elif method == "Resize / Zoom":

        h, w = image.shape[:2]

        image = cv2.resize(
            image,
            (int(w * 1.4), int(h * 1.4))
        )

    _, buffer = cv2.imencode(
        ".jpg",
        image
    )

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    return jsonify({
        "image": "data:image/jpeg;base64," + encoded
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import json
import os
import base64

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "fruit_model.keras")
CLASS_PATH = os.path.join("models", "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)


def predict_image(image):

    resized = cv2.resize(image, IMG_SIZE)
    x = np.expand_dims(resized, axis=0)

    predictions = model.predict(x, verbose=0)[0]

    indices = np.argsort(predictions)[::-1][:3]

    results = []

    for i in indices:
        name = class_names[i].replace(" fruit", "").title()
        confidence = float(predictions[i]) * 100

        results.append({
            "name": name,
            "confidence": round(confidence, 2)
        })

    return results


@app.route("/")
def home():
    return render_template(
        "index.html",
        classes=class_names
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    data = np.frombuffer(
        file.read(),
        np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = predict_image(image)

    return jsonify({
        "prediction": results[0]["name"],
        "confidence": results[0]["confidence"],
        "top3": results
    })


@app.route("/enhance", methods=["POST"])
def enhance():

    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]

    method = request.form.get(
        "method",
        "Original"
    )

    data = np.frombuffer(
        file.read(),
        np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({"error": "Invalid image"}), 400


    # ==========================================
    # 1. Original Image
    # ==========================================

    if method == "Original":

        processed_image = image


    # ==========================================
    # 2. Average Blur
    # ==========================================

    elif method == "Average Blur":

        processed_image = cv2.blur(
            image,
            (7, 7)
        )


    # ==========================================
    # 3. Gaussian Blur
    # ==========================================

    elif method == "Gaussian Blur":

        processed_image = cv2.GaussianBlur(
            image,
            (7, 7),
            0
        )


    # ==========================================
    # 4. Histogram
    # ==========================================

    elif method == "Histogram":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Histogram Equalization
        equalized = cv2.equalizeHist(gray)

        # Convert back to BGR for JPEG output
        processed_image = cv2.cvtColor(
            equalized,
            cv2.COLOR_GRAY2BGR
        )


    # ==========================================
    # 5. Edge Detection
    # ==========================================

    elif method == "Edge Detection":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        processed_image = cv2.cvtColor(
            edges,
            cv2.COLOR_GRAY2BGR
        )


    # ==========================================
    # 6. Sharpening - Advanced Filter
    # ==========================================

    elif method == "Sharpening":

        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        processed_image = cv2.filter2D(
            image,
            -1,
            kernel
        )


    # ==========================================
    # 7. Resize / Zoom
    # ==========================================

    elif method == "Resize / Zoom":

        h, w = image.shape[:2]

        processed_image = cv2.resize(
            image,
            (
                int(w * 1.4),
                int(h * 1.4)
            ),
            interpolation=cv2.INTER_CUBIC
        )


    # ==========================================
    # Unknown Method
    # ==========================================

    else:

        return jsonify({
            "error": "Unknown enhancement method"
        }), 400


    # ==========================================
    # Convert Processed Image to Base64
    # ==========================================

    _, buffer = cv2.imencode(
        ".jpg",
        processed_image
    )

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")


    return jsonify({
        "image": "data:image/jpeg;base64," + encoded
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )