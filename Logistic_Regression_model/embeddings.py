import os, glob, numpy as np, torch, joblib
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

DATA_DIR = "C:/Users/baodu/Dropbox/ML_project_IC/celebrity_faces_all"   # <-- change me

# 1) List images and labels
paths, labels = [], []
for person in sorted(os.listdir(DATA_DIR)):
    pdir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(pdir): continue
    imgs = glob.glob(os.path.join(pdir, "*"))
    for fp in imgs:
        if fp.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
            paths.append(fp); labels.append(person)

paths = np.array(paths)
labels = np.array(labels)
print("total images:", len(paths), "total classes:", len(np.unique(labels)))

# 2) Encode labels 0..K-1
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)
print("encoded classes:", num_classes)

# 3) Face detector (MTCNN) + embedder (InceptionResnetV1 pretrained on vggface2)
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 4) Extract 512-D embeddings
embs = []
keep_idx = []
for i, fp in enumerate(tqdm(paths, desc="Embedding")):
    try:
        img = Image.open(fp).convert('RGB')
        face = mtcnn(img)   # (3, 160, 160) tensor or None
        if face is None:
            continue
        with torch.no_grad():
            emb = embedder(face.unsqueeze(0).to(device)).cpu().numpy()[0]  # (512,)
        embs.append(emb); keep_idx.append(i)
    except Exception:
        continue

embs = np.array(embs, dtype=np.float32)
y = y[np.array(keep_idx)]
print("kept embeddings:", embs.shape, "labels:", y.shape)

# 5) Save artifacts for training/serving
os.makedirs("models", exist_ok=True)
np.savez_compressed("models/face_embeds.npz", X=embs, y=y)
joblib.dump(le, "models/label_encoder.pkl")
print("Saved: models/face_embeds.npz and models/label_encoder.pkl")
