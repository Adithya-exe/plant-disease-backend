from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import os
import json
import uuid
import time

# Firebase
import firebase_admin
from firebase_admin import credentials, auth


# ===== BASIC CONFIG =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Force CPU only

tf.get_logger().setLevel("ERROR")

# Keep TensorFlow lightweight for Render free tier
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 128

app = Flask(__name__)


# ===== CORS =====
CORS(app)


# ===== FILE LIMIT =====
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB


# ===== FIREBASE INIT =====
firebase_key = os.environ.get("FIREBASE_KEY")

if not firebase_key:
    raise ValueError("FIREBASE_KEY environment variable not set")

firebase_config = json.loads(firebase_key)
cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)

print("Firebase initialized")


# ===== FOLDERS =====
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ===== CUSTOM LAYERS =====
@keras.saving.register_keras_serializable(package="Custom")
def reduce_mean_spatial(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)


@keras.saving.register_keras_serializable(package="Custom")
def reduce_max_spatial(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)


# ===== LOAD CLASSES =====
with open(os.path.join(BASE_DIR, "new_class_names.json")) as f:
    classes = json.load(f)

print(f"Loaded {len(classes)} classes")

# ===== LOAD REMEDIES =====
with open(os.path.join(BASE_DIR, "plant_remedies.json")) as f:
    raw_remedies = json.load(f)

# Convert list → dictionary
remedies = {}

for item in raw_remedies:
    key = item["name"]

    # Convert to match prediction format
    formatted_key = key.replace("___", " - ").replace("_", " ")

    remedies[formatted_key] = item["remedy"]

print(f"Loaded {len(remedies)} remedies")



# ===== LOAD MODEL =====
model = None

try:
    print("Loading model...")

    model = keras.models.load_model(
        os.path.join(BASE_DIR, "model/plant_disease_new_dset.keras"),
        compile=False,
        custom_objects={
            "reduce_mean_spatial": reduce_mean_spatial,
            "reduce_max_spatial": reduce_max_spatial,
            "Custom>reduce_mean_spatial": reduce_mean_spatial,
            "Custom>reduce_max_spatial": reduce_max_spatial,
        }
    )

    print("Model loaded successfully!")

    # Warmup prediction
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model.predict_on_batch(dummy)

    print("Warmup prediction complete")

except Exception as e:
    print("Model loading failed:", str(e))


# ===== PREPROCESS =====
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ===== TOKEN VERIFY =====
def verify_token(req):
    auth_header = req.headers.get("Authorization")

    if not auth_header:
        return None

    try:
        parts = auth_header.split(" ")

        if len(parts) != 2:
            return None

        token = parts[1]
        decoded = auth.verify_id_token(token)

        return decoded

    except Exception as e:
        print("Auth error:", str(e))
        return None


# ===== ROUTES =====
@app.route("/")
def home():
    return "Plant Disease API Running 🚀"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "modelLoaded": model is not None
    })


@app.route("/warmup", methods=["GET"])
def warmup():
    return jsonify({"ok": True})


# ===== PREDICT =====
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    path = None

    try:
        print("\n1. Request received")

        # Auth
        user = verify_token(request)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401

        print("2. Auth verified")

        # Model
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        # File
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save
        filename = str(uuid.uuid4()) + "_" + file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        print("3. File saved")

        # Preprocess
        img = preprocess(path)
        print("4. Preprocessing complete")

        # Predict
        pred_start = time.time()
        pred = model.predict_on_batch(img)
        print(f"Predict time: {time.time() - pred_start:.2f} sec")
        print("5. Prediction complete")

        # Results
        idx = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100

        predicted_class_raw = classes[idx]
        predicted_class = predicted_class_raw.replace(
            "___", " - "
        ).replace("_", " ")

        remedy = remedies.get(predicted_class, "No remedy information available.")

        print(
            f"Prediction: {predicted_class} "
            f"({confidence:.2f}%)"
        )

        print(f"Total time: {time.time() - start:.2f} sec")

        # Message
        if confidence < 40:
            message = f"Likely: {predicted_class}"
        elif confidence < 70:
            message = f"{predicted_class} (moderate confidence)"
        else:
            message = f"{predicted_class} ({confidence:.2f}% confidence)"

        return jsonify({
         "prediction": predicted_class,
         "confidence": confidence,
         "message": message,
            "remedy": remedy,
            "user": user.get("email")
        })


    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Server starting on port", port)
    app.run(host="0.0.0.0", port=port)
