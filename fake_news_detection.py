# -*- coding: utf-8 -*-
"""fake_news_detection.ipynb

from google.colab import drive
drive.mount('/content/drive')

import os
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image


import torch
from torchvision import transforms
from transformers import ViTModel
from torchvision.models import resnet50,vgg16
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 1) CONFIG: set dataset path
# -----------------------------
# Change this to where your FakeNewsNet root folder is (Drive or Colab workspace)
FAKENEWSNET_ROOT = '/content/drive/MyDrive/dataset11' # <- edit if needed
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'
print('Device:', DEVICE)

# 2) Load CSVs
# -----------------------------
# Example: assume you have these files:
#   FakeNewsNet/fake1.csv
#   FakeNewsNet/fake2.csv
#   FakeNewsNet/real1.csv
#   FakeNewsNet/real2.csv
# Change names below according to your actual file names

fake_files = ['politifact_fake.csv']
real_files = ['politifact_real.csv']

def load_and_label(files, label):
    dfs = []
    for f in files:
        path = os.path.join(FAKENEWSNET_ROOT, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            dfs.append(df)
        else:
            print(f"File not found: {path}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_fake = load_and_label(fake_files, 1)
df_real = load_and_label(real_files, 0)

df = pd.concat([df_fake, df_real], ignore_index=True)

print("Combined dataset shape:", df.shape)
print("Real:", (df['label']==0).sum(), "Fake:", (df['label']==1).sum())

# -----------------------------
# 3) Combine title + url as text
# -----------------------------
df['text'] = df['title'].astype(str) + " " + df['news_url'].astype(str)

# Optionally, if you want to use tweet_ids for text context too:
# df['text'] = df['title'].astype(str) + " " + df['news_url'].astype(str) + " " + df['tweet_ids'].astype(str)

# -----------------------------
# 4) Attach dummy images or image paths if available
# -----------------------------
# If you have images stored somewhere, map them via id
# For now, just make empty list to keep pipeline consistent
df['images'] = [[] for _ in range(len(df))]

print(df.head())

# 3) Embedding extraction
#    - Text: sentence-transformers all-MiniLM
#    - Image: ResNet50 pretrained (pooling layer)
# -----------------------------

# Initialize models
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # small & fast
text_model.max_seq_length = 512

# Image model:  without final classification layer
img_model =  ViTModel.from_pretrained("google/vit-base-patch16-224")
img_model.eval()
# remove final fc: we'll use avgpool output (2048-d)
img_model = torch.nn.Sequential(*list(img_model.children())[:-1])
img_model.to(DEVICE)

# Image transforms
img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Batch extraction functions
BATCH_SIZE = 32

def extract_text_embeddings(texts, batch_size=64):
    embeddings = text_model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embeddings)


def extract_image_embedding_single(path_or_url):
    try:
        if str(path_or_url).startswith('http'):
            # download to /tmp
            import requests, io
            r = requests.get(path_or_url, timeout=10)
            img = Image.open(io.BytesIO(r.content)).convert('RGB')
        else:
            img = Image.open(path_or_url).convert('RGB')
    except Exception as e:
        # fallback: return zero vector
        return np.zeros((2048,), dtype=float)

    t = img_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = img_model(t).squeeze().cpu().numpy()
    feat = feat.reshape(-1)
    return feat


def extract_image_embeddings(paths_list, batch_size=32):
    # paths_list: list of image paths or urls (one per sample). If multiple images, we use first.
    embs = []
    for p in tqdm(paths_list):
        if isinstance(p, list) and len(p) > 0:
            chosen = p[0]
        else:
            chosen = p
        emb = extract_image_embedding_single(chosen)
        embs.append(emb)
    return np.vstack(embs)

# Prepare inputs
texts = df['text'].astype(str).tolist()
image_paths = df['images'].tolist()
labels = df['label'].astype(int).tolist()

# Extract (these may take time depending on dataset size; consider sampling or using a subset for testing)
print('Extracting text embeddings...')
text_embs = extract_text_embeddings(texts, batch_size=64)
print('Text embeddings shape:', text_embs.shape)

print('Extracting image embeddings...')
image_embs = extract_image_embeddings(image_paths)
print('Image embeddings shape:', image_embs.shape)

# Combine into multimodal features: concatenation + optional normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_text = text_embs
X_image = image_embs
X = np.concatenate([X_text, X_image], axis=1)
X = scaler.fit_transform(X)

y = np.array(labels)

# 4) Train / Eval
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 (macro):', f1_score(y_test, y_pred, average='macro'))
print('\nClassification report:\n', classification_report(y_test, y_pred))

# Confusion matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white')
plt.show()

# Save model & scaler
import joblib
os.makedirs('outputs', exist_ok=True)
joblib.dump(clf, 'outputs/mm_clf_logreg.joblib')
joblib.dump(scaler, 'outputs/mm_scaler.joblib')
np.save('outputs/text_embs.npy', text_embs)
np.save('outputs/image_embs.npy', image_embs)

print('\nSaved models/embeddings under outputs/')

# ROC Curve and AUC
from sklearn.metrics import roc_curve, auc

# Probabilities (needed for ROC)
y_proba = clf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=40)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(7,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='coolwarm', s=8)
plt.title('t-SNE of Multimodal Embeddings')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.colorbar()
plt.show()
