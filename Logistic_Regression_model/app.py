import io, os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import joblib
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# load artifacts
ART_DIR = "models"
clf     = joblib.load(os.path.join(ART_DIR, "emb_classifier.pkl"))
scaler  = joblib.load(os.path.join(ART_DIR, "emb_scaler.pkl"))
le      = joblib.load(os.path.join(ART_DIR, "label_encoder.pkl"))  # maps id <-> name

device   = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn    = MTCNN(image_size=160, margin=20, post_process=True, device=device)
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

app = Flask(__name__)

# Utils
def image_bytes_to_embedding(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    face = mtcnn(img)  # (3,160,160)
    if face is None:
        raise ValueError("No face detected. Try another image.")
    with torch.no_grad():
        emb = embedder(face.unsqueeze(0).to(device)).cpu().numpy()[0]  # (512,)
    return emb

def topk_from_classifier(est, Xs, k=5):
    """Return (indices, scores) for top-k predictions."""
    if hasattr(est, "predict_proba"):
        scores = est.predict_proba(Xs)[0]  # (C,)
    else:
        margins = est.decision_function(Xs)[0]  # (C,)
        # softmax to turn margins into pseudo-probabilities
        m = margins.max()
        scores = np.exp(margins - m)
        scores = scores / scores.sum()
    k = min(k, len(scores))
    idx = np.argpartition(-scores, range(k))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

# ---------- routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file", None)
    if f is None or f.filename == "":
        return jsonify({"error": "no file uploaded"}), 400
    try:
        emb = image_bytes_to_embedding(f.read()).reshape(1, -1)
        Xs  = scaler.transform(emb)

        idx, sc = topk_from_classifier(clf, Xs, k=5)
        names = le.inverse_transform(idx)  # indices -> class names

        out = [{"label": str(n), "score": float(s)} for n, s in zip(names, sc)]
        return jsonify({"predictions": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
    print("Running")