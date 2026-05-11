from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import os
import json
import uuid
import time
import urllib.request

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
@tf.keras.utils.register_keras_serializable(package="Custom")
def reduce_mean_spatial(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)


@tf.keras.utils.register_keras_serializable(package="Custom")
def reduce_max_spatial(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable(package="Custom")
class CBAMLayer(layers.Layer):

    def __init__(self, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]

        self.shared_1 = layers.Dense(
            channels // self.reduction,
            activation='relu'
        )

        self.shared_2 = layers.Dense(channels)

        self.spatial_conv = layers.Conv2D(
            1,
            7,
            padding='same',
            activation='sigmoid'
        )

    def call(self, x):

        avg_p = tf.reduce_mean(
            x,
            axis=(1, 2),
            keepdims=True
        )

        max_p = tf.reduce_max(
            x,
            axis=(1, 2),
            keepdims=True
        )

        ca = tf.nn.sigmoid(
            self.shared_2(self.shared_1(avg_p)) +
            self.shared_2(self.shared_1(max_p))
        )

        x = x * ca

        avg_s = tf.reduce_mean(
            x,
            axis=-1,
            keepdims=True
        )

        max_s = tf.reduce_max(
            x,
            axis=-1,
            keepdims=True
        )

        sa = self.spatial_conv(
            tf.concat([avg_s, max_s], axis=-1)
        )

        return x * sa

    def get_config(self):

        config = super().get_config()

        config.update({
            "reduction": self.reduction
        })

        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.pos_emb = layers.Embedding(
            input_dim=num_patches,
            output_dim=embed_dim
        )

    def call(self, patch):

        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1
        )

        return patch + self.pos_emb(positions)

    def get_config(self):

        config = super().get_config()

        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim
        })

        return config


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


def _is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            return f.read(32).startswith("version https://git-lfs.github.com")
    except OSError:
        return False


def _ensure_model_file(model_path: str) -> None:
    """Railway and similar hosts often ship Git LFS pointer files only; optionally fetch real weights."""
    download_url = (os.environ.get("MODEL_DOWNLOAD_URL") or "").strip()
    exists = os.path.isfile(model_path)
    size = None
    if exists:
        try:
            size = os.path.getsize(model_path)
        except OSError:
            size = None

    is_pointer = _is_git_lfs_pointer(model_path) if exists else False
    # Heuristic: real `.keras` is typically tens/hundreds of MB. LFS pointer is tiny.
    suspiciously_small = (size is not None) and (size < 1024 * 1024)

    print(
        "Model file check:",
        {"exists": exists, "size_bytes": size, "is_lfs_pointer": is_pointer, "too_small": suspiciously_small},
    )

    if exists and not is_pointer and not suspiciously_small:
        return
    if not download_url:
        print("MODEL_DOWNLOAD_URL is not set; skipping model download.")
        if _is_git_lfs_pointer(model_path):
            print(
                "Model file is a Git LFS pointer (real weights missing). "
                "Add a Railpack build `git lfs pull` (see railpack.json) or set MODEL_DOWNLOAD_URL."
            )
        return
    print("Downloading model from MODEL_DOWNLOAD_URL …")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp = model_path + ".part"
    try:
        with urllib.request.urlopen(download_url, timeout=900) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, model_path)
        try:
            print("Model download finished. size_bytes=", os.path.getsize(model_path))
        except OSError:
            print("Model download finished.")
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ===== LOAD MODEL =====
model = None

MODEL_PATH = os.environ.get(
    "MODEL_KERAS_PATH",
    os.path.join(BASE_DIR, "model", "plant_disease_final.keras"),
)

try:
    print("Startup commit:", os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "unknown")
    print("MODEL_PATH:", MODEL_PATH)
    print("MODEL_DOWNLOAD_URL set:", bool((os.environ.get("MODEL_DOWNLOAD_URL") or "").strip()))
    print("Loading model...")
    _ensure_model_file(MODEL_PATH)

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
    "CBAMLayer": CBAMLayer,
    "PatchEncoder": PatchEncoder,
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
