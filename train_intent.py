"""
Intent Prediction Model — CNN + LSTM
Binary classification: screwdriver interaction vs. no interaction
Outputs: class label + confidence score

Folder structure expected:
    data/
      video_data/
        reaches_for_screwdriver/
            clip1.mov
            clip2.mov
        no_interaction/
            clip1.mov
            clip2.mov

Usage:
    Train:
        python train_intent.py --mode train --video_path ./data/video_data --n_epochs 30 --batch_size 4

    Inference on a single video:
        python train_intent.py --mode infer --video_path ./data/video_data --resume_path ./snapshots/best_model.pth --infer_video path/to/video.mov
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
from pathlib import Path


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',         type=str,   default='train', choices=['train', 'infer'])
    parser.add_argument('--video_path',   type=str,   default='./data/video_data')
    parser.add_argument('--resume_path',  type=str,   default=None)
    parser.add_argument('--infer_video',  type=str,   default=None)
    parser.add_argument('--n_epochs',     type=int,   default=30)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--lr_rate',      type=float, default=1e-4)
    parser.add_argument('--n_frames',     type=int,   default=16,   help='Frames sampled per video')
    parser.add_argument('--img_size',     type=int,   default=112,  help='Resize each frame to img_size x img_size')
    parser.add_argument('--hidden_size',  type=int,   default=256,  help='LSTM hidden units')
    parser.add_argument('--n_layers',     type=int,   default=2,    help='LSTM layers')
    parser.add_argument('--val_split',    type=float, default=0.2,  help='Fraction of data used for validation')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--num_workers',  type=int,   default=0)
    parser.add_argument('--snapshot_dir', type=str,   default='./snapshots')
    return parser.parse_args()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class VideoDataset(Dataset):
    """
    Reads .mov videos directly from class-named subfolders.
    Samples n_frames evenly spaced frames from each video.
    """

    def __init__(self, samples, class_to_idx, n_frames, img_size, augment=False):
        self.samples     = samples        # list of (video_path, label_idx)
        self.class_to_idx = class_to_idx
        self.n_frames    = n_frames
        self.img_size    = img_size
        self.augment     = augment

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        tensor = torch.stack([self.transform(f) for f in frames])  # (n_frames, C, H, W)
        return tensor, label

    def _load_frames(self, path):
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)

        indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames  = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret or frame is None:
                # If frame read fails, use a black frame
                frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames


def build_datasets(video_path, n_frames, img_size, val_split, seed):
    video_path = Path(video_path)
    classes    = sorted([d.name for d in video_path.iterdir() if d.is_dir()])
    assert len(classes) >= 2, f"Need at least 2 class folders, found: {classes}"

    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"\nClasses found: {class_to_idx}\n")

    all_samples = []
    for cls in classes:
        for ext in ['*.mov', '*.mp4', '*.avi']:
            for vp in (video_path / cls).glob(ext):
                all_samples.append((vp, class_to_idx[cls]))

    assert len(all_samples) > 0, "No video files found. Check your video_path and that videos are .mov/.mp4/.avi"

    random.seed(seed)
    random.shuffle(all_samples)

    split      = int(len(all_samples) * (1 - val_split))
    train_data = all_samples[:split]
    val_data   = all_samples[split:]

    print(f"Total videos : {len(all_samples)}")
    print(f"Train        : {len(train_data)}")
    print(f"Val          : {len(val_data)}\n")

    train_ds = VideoDataset(train_data, class_to_idx, n_frames, img_size, augment=True)
    val_ds   = VideoDataset(val_data,   class_to_idx, n_frames, img_size, augment=False)

    return train_ds, val_ds, classes, class_to_idx


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class CNNLSTM(nn.Module):
    """
    ResNet-50 feature extractor (frozen) + LSTM sequence model + FC classifier.
    """

    def __init__(self, n_classes, hidden_size=256, n_layers=2):
        super().__init__()

        # CNN: ResNet-50 without final classification layer
        resnet       = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn     = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)
        self.cnn_dim = 2048

        # Freeze CNN weights
        for param in self.cnn.parameters():
            param.requires_grad = False

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3 if n_layers > 1 else 0.0
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Extract CNN features for each frame
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            feats = self.cnn(x)                 # (B*T, 2048, 1, 1)
        feats = feats.view(B, T, self.cnn_dim)  # (B, T, 2048)

        # LSTM over frame sequence
        out, _ = self.lstm(feats)               # (B, T, hidden)
        last   = out[:, -1, :]                  # take final timestep

        logits = self.classifier(last)          # (B, n_classes)
        return logits


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds, val_ds, classes, class_to_idx = build_datasets(
        args.video_path, args.n_frames, args.img_size, args.val_split, args.seed
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model     = CNNLSTM(n_classes=len(classes), hidden_size=args.hidden_size, n_layers=args.n_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if args.resume_path and os.path.exists(args.resume_path):
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Resumed from {args.resume_path}")

    os.makedirs(args.snapshot_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.n_epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(clips)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * clips.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += clips.size(0)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips)
                loss   = criterion(logits, labels)

                val_loss    += loss.item() * clips.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += clips.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        t_loss    = train_loss    / train_total
        v_loss    = val_loss      / val_total

        print(f"Epoch {epoch:03d}/{args.n_epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={v_loss:.4f}  val_acc={val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path    = os.path.join(args.snapshot_dir, 'best_model.pth')
            torch.save({'model': model.state_dict(), 'classes': classes, 'class_to_idx': class_to_idx}, save_path)
            print(f"  ✓ Saved best model → {save_path}  (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def infer(args):
    assert args.resume_path, "Provide --resume_path to the saved model"
    assert args.infer_video,  "Provide --infer_video path to a .mov file"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt         = torch.load(args.resume_path, map_location=device)
    classes      = ckpt['classes']
    class_to_idx = ckpt['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = CNNLSTM(n_classes=len(classes), hidden_size=args.hidden_size, n_layers=args.n_layers).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Load video
    dummy_label = 0
    ds      = VideoDataset([(args.infer_video, dummy_label)], class_to_idx, args.n_frames, args.img_size)
    clip, _ = ds[0]
    clip    = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        logits      = model(clip)
        probs       = torch.softmax(logits, dim=1)[0]
        pred_idx    = probs.argmax().item()
        pred_label  = idx_to_class[pred_idx]
        confidence  = probs[pred_idx].item() * 100

    print(f"\nVideo      : {args.infer_video}")
    print(f"Prediction : {pred_label}")
    print(f"Confidence : {confidence:.1f}%")
    print("\nAll class probabilities:")
    for i, cls in idx_to_class.items():
        print(f"  {cls:40s} {probs[i].item()*100:.1f}%")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args)
