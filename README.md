# Celebrity Face Recognition with Flask

This project implements a celebrity face recognition web app.  
The dataset is from Hugging Face and it contains roughly 16k pictures from around 1000 celebrities.
I started with **hand-crafted features (wavelet transform) + an MLP model**, but results were poor.  
I then tried **pretrained face embeddings** (InceptionResnetV1 from `facenet-pytorch`) and trained a **Logistic Regression classifier** on top — which produced much stronger results.  

---

## 📌 Project Overview

### Phase 1 — Wavelet + MLP
- Extracted features by resizing faces to 32×32 and combining raw pixels with Haar wavelet coefficients (4096 features).
- Trained a Multi-Layer Perceptron (MLP) with PyTorch.
- Validation accuracy plateaued around ~30% Top-1, ~50% Top-5 across ~1000 celebrity classes.
- Conclusion: feature engineering + MLP was not sufficient for robust recognition.

### Phase 2 — Pretrained Face Embeddings
- Switched to `facenet-pytorch` pipeline:
  - **MTCNN** for face detection/alignment.
  - **InceptionResnetV1** pretrained on VGGFace2 to extract **512-D embeddings**.
- Trained a **Logistic Regression classifier** (multinomial softmax with `saga` solver).
- Achieved significantly higher validation/test accuracy compared to MLP.
- Benefits: embeddings capture robust facial features → simpler classifier works well.

### Phase 3 — Flask Web App
- Built a Flask backend with routes:
  - `/` → upload an image in the browser.
  - `/predict` → runs preprocessing → embedding → Logistic Regression → returns top-5 predictions (celebrity name + confidence).
- Frontend: simple HTML/JS with file upload and result display.

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/celebrity-face-recognition.git
cd celebrity-face-recognition
