"""
mus_classifier.py
=================
Full pipeline for Mus facial signal classification.

Modes:
  python mus_classifier.py train              # extract embeddings + train head
  python mus_classifier.py run                # live webcam inference
  python mus_classifier.py eval               # evaluate on labeled set
  python mus_classifier.py train --csv path/to/mus_labels.csv --images path/to/images/

  # 1. Train — point --images at the folder containing your JPG/MP4/MOV files
python mus_classifier.py train --csv mus_labels.csv --images /mus_expressions/mus_photos/labeled1

python3 mus_expressions/mus_classifier.py train \
  --csv "mus_expressions/mus_labels.csv" \
  --images "mus_expressions/mus_photos/labeled1"
  
# 2. Evaluate — prints overall accuracy + full confusion matrix
python3 mus_expressions/mus_classifier.py eval --csv "mus_expressions/mus_labels.csv" --images "mus_expressions/mus_photos/labeled1"

# 3. Run live on webcam
python3 mus_expressions/mus_classifier.py run

Requirements:
  pip install transformers torch pillow opencv-python pandas scikit-learn
"""

import argparse
import os
import sys
import csv
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = [
    "neutral",
    "dos_reyes",
    "dos_ases",
    "medias_reyes",
    "medias_ases",
    "duples",
    "solomillo",
    "treinta_y_una",
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for i, c in enumerate(CLASSES)}

CLIP_MODEL   = "openai/clip-vit-base-patch32"
HEAD_PATH    = "mus_head.pt"
EMBED_DIM    = 1024
HIDDEN_DIM   = 256
DROPOUT      = 0.2

# Inference thresholds — tune after eval
THRESHOLD    = 0.55   # minimum softmax confidence to fire
MIN_MARGIN   = 0.10   # minimum gap between top-2 softmax scores

# Temporal smoother settings
WINDOW_SIZE  = 10
STREAK_NEEDED = 6

# Video: how many evenly-spaced frames to sample per clip
VIDEO_FRAMES = 8


# ── Model setup ──────────────────────────────────────────────────────────────

def load_clip():
    print("Loading CLIP …")
    model     = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()
    return model, processor


def build_head():
    return nn.Sequential(
        nn.Linear(EMBED_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN_DIM, len(CLASSES)),
    )


def load_head(path=HEAD_PATH):
    head = build_head()
    head.load_state_dict(torch.load(path, map_location="cpu"))
    head.eval()
    return head


# ── Embedding helpers ────────────────────────────────────────────────────────

def crop_to_face(pil_img, padding=0.05):
    """Crop PIL image tightly to the detected face.
    Falls back to a center crop if no face is found.
    """
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
        # Largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        ih, iw = img_bgr.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + w + pad_x)
        y2 = min(ih, y + h + pad_y)
        cropped = img_bgr[y1:y2, x1:x2]
    else:
        # Fallback: center crop 50 % of the frame
        ih, iw = img_bgr.shape[:2]
        crop_w = int(iw * 0.5)
        crop_h = int(ih * 0.5)
        x1 = (iw - crop_w) // 2
        y1 = (ih - crop_h) // 2
        cropped = img_bgr[y1:y1 + crop_h, x1:x1 + crop_w]

    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


def crop_to_lower_face(face_pil, ratio=0.55):
    """Crop the lower portion of a face image (roughly nose down).
    Returns a PIL Image focused on mouth/lips/tongue region.
    """
    img = np.array(face_pil)
    h, w = img.shape[:2]
    y_start = int(h * (1 - ratio))
    lower = img[y_start:h, 0:w]
    return Image.fromarray(lower)


def embed_image(img_pil, model, processor):
    """Return L2-normalised CLIP embedding for a PIL image.
    Extracts both a full-face crop and a lower-face (mouth) crop,
    concatenates their embeddings into a 1024-d vector.
    """
    face_pil = crop_to_face(img_pil)          # tight face crop
    lower_pil = crop_to_lower_face(face_pil)  # mouth region only

    with torch.no_grad():
        inputs = processor(images=[face_pil, lower_pil], return_tensors="pt")

        # Robust extraction: get_image_features might return a model output object
        try:
            feat = model.get_image_features(**inputs)
        except Exception:
            outputs = model(**inputs)
            feat = outputs.image_embeds

        # If it's still a model output, grab the tensor inside
        if not isinstance(feat, torch.Tensor):
            if hasattr(feat, "image_embeds"):
                feat = feat.image_embeds
            elif hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            else:
                raise TypeError(f"Unexpected feat type: {type(feat)}")

        feat = F.normalize(feat, dim=-1)      # (2, 512)
    fused = torch.cat([feat[0], feat[1]], dim=-1)  # (1024,)
    return fused


def embed_frame_bgr(frame_bgr, model, processor):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return embed_image(img, model, processor)


def embed_video(path, model, processor, n_frames=VIDEO_FRAMES):
    """Sample n_frames evenly from a video, return mean embedding."""
    cap    = cv2.VideoCapture(str(path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    feats   = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        feats.append(embed_frame_bgr(frame, model, processor))
    cap.release()
    if not feats:
        return None
    stacked = torch.stack(feats)
    mean    = stacked.mean(0)
    return F.normalize(mean, dim=-1)


# ── Training ─────────────────────────────────────────────────────────────────

def train(csv_path, images_dir, epochs=300, lr=1e-3, val_split=0.15):
    model, processor = load_clip()

    csv_path   = Path(csv_path)
    images_dir = Path(images_dir)

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((row["filename"], row["label"]))

    embeddings, labels, skipped = [], [], []

    print(f"\nExtracting embeddings from {len(rows)} files …")
    for i, (fname, label) in enumerate(rows):
        if label not in CLASS2IDX:
            print(f"  [skip] unknown label '{label}' in {fname}")
            skipped.append(fname)
            continue

        path = images_dir / fname
        if not path.exists():
            print(f"  [skip] file not found: {path}")
            skipped.append(fname)
            continue

        suffix = path.suffix.lower()
        try:
            if suffix in (".mp4", ".mov", ".avi", ".mkv"):
                feat = embed_video(path, model, processor)
            else:
                img  = Image.open(path).convert("RGB")
                feat = embed_image(img, model, processor)

            if feat is None:
                print(f"  [skip] could not read: {fname}")
                skipped.append(fname)
                continue

            embeddings.append(feat.numpy())
            labels.append(CLASS2IDX[label])

        except Exception as e:
            print(f"  [error] {fname}: {e}")
            skipped.append(fname)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)} done")

    print(f"\n{len(embeddings)} embeddings extracted, {len(skipped)} skipped")

    if len(embeddings) < 8:
        print("Not enough data to train. Check that --images points to the folder containing your files.")
        sys.exit(1)

    X = torch.tensor(np.array(embeddings))   # (N, 1024)
    y = torch.tensor(labels)                  # (N,)

    # Train / val split
    n_val = max(1, int(len(X) * val_split))
    perm  = torch.randperm(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    head    = build_head()
    opt     = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    print(f"\nTraining: {len(X_train)} train / {len(X_val)} val samples")
    print(f"Classes: {CLASSES}\n")

    for epoch in range(1, epochs + 1):
        head.train()
        logits = head(X_train)
        loss   = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 50 == 0 or epoch == epochs:
            head.eval()
            with torch.no_grad():
                train_acc = (head(X_train).argmax(1) == y_train).float().mean()
                val_acc   = (head(X_val).argmax(1) == y_val).float().mean()
            print(f"Epoch {epoch:4d}  loss {loss:.4f}  train {train_acc:.1%}  val {val_acc:.1%}")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in head.state_dict().items()}

    # Save best checkpoint
    torch.save(best_state, HEAD_PATH)
    print(f"\nSaved {HEAD_PATH}  (best val acc: {best_val_acc:.1%})")

    # Per-class accuracy on full set
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        preds = head(X).argmax(1)
    print("\nPer-class accuracy (full set):")
    for cls_idx, cls_name in IDX2CLASS.items():
        mask     = y == cls_idx
        if mask.sum() == 0:
            continue
        cls_acc  = (preds[mask] == y[mask]).float().mean()
        n        = mask.sum().item()
        print(f"  {cls_name:20s}  {cls_acc:.1%}  ({n} samples)")


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(csv_path, images_dir):
    if not Path(HEAD_PATH).exists():
        print(f"No trained head found at {HEAD_PATH}. Run 'train' first.")
        sys.exit(1)

    model, processor = load_clip()
    head = load_head()

    csv_path   = Path(csv_path)
    images_dir = Path(images_dir)

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((row["filename"], row["label"]))

    correct, total = 0, 0
    confusion = [[0]*len(CLASSES) for _ in range(len(CLASSES))]

    for fname, true_label in rows:
        if true_label not in CLASS2IDX:
            continue
        path = images_dir / fname
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in (".mp4", ".mov", ".avi", ".mkv"):
                feat = embed_video(path, model, processor)
            else:
                img  = Image.open(path).convert("RGB")
                feat = embed_image(img, model, processor)
            if feat is None:
                continue
        except Exception:
            continue

        with torch.no_grad():
            logits = head(feat.unsqueeze(0))
            pred_idx = logits.argmax(1).item()

        true_idx = CLASS2IDX[true_label]
        confusion[true_idx][pred_idx] += 1
        if pred_idx == true_idx:
            correct += 1
        total += 1

    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.1%}\n")
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{c[:8]:>10}" for c in CLASSES)
    print(f"{'':20s}{header}")
    for i, row in enumerate(confusion):
        print(f"{CLASSES[i]:20s}" + "".join(f"{v:10d}" for v in row))


# ── Inference helpers ────────────────────────────────────────────────────────

def predict_embedding(feat, head):
    """Return (class_name, confidence, margin) from a 1024-d embedding tensor."""
    with torch.no_grad():
        logits  = head(feat.unsqueeze(0))
        probs   = torch.softmax(logits, dim=1).squeeze()
    top2    = probs.topk(2)
    conf    = top2.values[0].item()
    margin  = (top2.values[0] - top2.values[1]).item()
    label   = IDX2CLASS[top2.indices[0].item()]
    return label, conf, margin


def should_fire(label, conf, margin):
    # Per-class confidence thresholds (global default = THRESHOLD)
    conf_needed = {"duples": 0.75}.get(label, THRESHOLD)
    return (label != "neutral"
            and conf >= conf_needed
            and margin >= MIN_MARGIN)


# ── Temporal smoother ────────────────────────────────────────────────────────

class TemporalSmoother:
    def __init__(self, window=WINDOW_SIZE, required=STREAK_NEEDED):
        self.history  = deque(maxlen=window)
        self.required = required
        self.last_fired = None

    def update(self, label):
        self.history.append(label)
        streak = sum(1 for l in self.history if l == label)
        if streak >= self.required and label not in ("neutral", "uncertain"):
            if label != self.last_fired:
                self.last_fired = label
                return label
        if label == "neutral":
            self.last_fired = None
        return None


# ── Webcam inference ─────────────────────────────────────────────────────────

# Per-class display colours (BGR)
CLASS_COLORS = {
    "neutral":       (150, 150, 150),
    "dos_reyes":     (80,  180, 255),
    "dos_ases":      (80,  255, 180),
    "medias_reyes":  (255, 180,  80),
    "medias_ases":   (255,  80, 180),
    "duples":        (180, 255,  80),
    "solomillo":     (255, 120, 120),
    "treinta_y_una": (120, 120, 255),
}


def run_webcam():
    if not Path(HEAD_PATH).exists():
        print(f"No trained head found at {HEAD_PATH}. Run 'train' first.")
        sys.exit(1)

    model, processor = load_clip()
    head     = load_head()
    smoother = TemporalSmoother()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    print("Running — press Q to quit")

    fired_label    = None
    fired_timer    = 0
    FIRED_DISPLAY  = 40   # frames to show the fired label banner

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        feat              = embed_frame_bgr(frame, model, processor)
        label, conf, margin = predict_embedding(feat, head)

        if not should_fire(label, conf, margin):
            display_label = "uncertain" if label != "neutral" else "neutral"
        else:
            display_label = label

        fired = smoother.update(display_label)
        if fired:
            fired_label = fired
            fired_timer = FIRED_DISPLAY

        # ── HUD ──────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        color = CLASS_COLORS.get(label, (200, 200, 200))

        # Current prediction bar (top left)
        cv2.rectangle(frame, (0, 0), (300, 36), (30, 30, 30), -1)
        cv2.putText(frame, f"{label}  {conf:.0%}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Confidence bar
        bar_w = int(conf * 280)
        cv2.rectangle(frame, (10, 30), (10 + bar_w, 34), color, -1)

        # Fired signal banner (bottom)
        if fired_timer > 0:
            fired_timer -= 1
            alpha = min(1.0, fired_timer / 10)
            banner_color = CLASS_COLORS.get(fired_label, (200, 200, 200))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
            cv2.putText(overlay, f">>> {fired_label.upper().replace('_', ' ')}",
                        (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        banner_color, 3)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow("Mus classifier  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mus signal classifier")
    parser.add_argument("mode", choices=["train", "run", "eval"],
                        help="train | run | eval")
    parser.add_argument("--csv",    default="mus_labels.csv",
                        help="Path to labeled CSV (default: mus_labels.csv)")
    parser.add_argument("--images", default=".",
                        help="Folder containing the image/video files (default: .)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.csv, args.images, epochs=args.epochs, lr=args.lr)
    elif args.mode == "run":
        run_webcam()
    elif args.mode == "eval":
        evaluate(args.csv, args.images)


if __name__ == "__main__":
    main()
