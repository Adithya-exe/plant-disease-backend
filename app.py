from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import os
import json
import uuid

# 🔐 Firebase
import firebase_admin
from firebase_admin import credentials, auth

# ===== BASIC CONFIG =====
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Keep TensorFlow lightweight on small instances (Render free tier).
# This reduces CPU parallelism (and peak memory) during inference.
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# 🔐 CORS
# Allow Firebase Hosting + Authorization header (Bearer token) for /predict.
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://pdd-finalyear.web.app",
                "https://pdd-finalyear.firebaseapp.com",
            ],
        }
    },
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
)

# 🔐 File upload limit
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

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

except Exception as e:
    print("Model loading failed:", e)

IMG_SIZE = 128

# ===== PREPROCESS =====
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
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

@app.route("/")
def home():
    return "Plant Disease API Running 🚀"
# ===== HEALTH CHECK =====
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "modelLoaded": model is not None
    })

# ===== PREDICT =====
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔐 AUTH
        user = verify_token(request)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401

        if model is None:
            return jsonify({"error": "Model not loaded"}), 503

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # ✅ FILE TYPE CHECK (FIXED POSITION)
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = str(uuid.uuid4()) + "_" + file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        print(f"\nUser: {user.get('email', 'unknown')}")
        print(f"Received file: {file.filename}")

        img = preprocess(path)

        pred = model.predict(img, verbose=0)
        idx = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100

        predicted_class_raw = classes[idx]

        print(f"Prediction: {predicted_class_raw} ({confidence:.2f}%)")

        try:
            os.remove(path)
        except:
            pass

        predicted_class = predicted_class_raw.replace("___", " - ").replace("_", " ")

        if confidence < 40:
            message = f"The plant is likely {predicted_class}"
        elif confidence < 70:
            message = f"Prediction: {predicted_class} (confidence is moderate)"
        else:
            message = f"It is {confidence:.2f}% confirmed that the plant has {predicted_class}"

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "message": message,
            "user": user.get("email")
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# ===== RUN =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Server starting on port", port)
    app.run(host="0.0.0.0", port=port)