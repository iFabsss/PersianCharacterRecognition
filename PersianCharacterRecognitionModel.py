"""
Persian Alphabet Character Recognition
CNN Classifier — PyTorch

Dataset format:
  {num}{character}.jpg   e.g. 1alef.jpg
  {num}{character}.txt   e.g. 1alef.txt  (contains "Alef")

Supported layouts:
  flat  : data_dir/{num}{char}.jpg + data_dir/{num}{char}.txt
  split : data_dir/images/*.jpg    + data_dir/labels/*.txt

Usage:
  # Train
  PersianCharacterRecognitionModel.py --data_dir ./dataset --epochs 30 --batch_size 32

  # Predict
  PersianCharacterRecognitionModel.py --predict path/to/image.jpg \
                    --output_dir ./output
"""

import os
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler

# Import Persian to English mapping
try:
    from PersianCharacterEnglishAlphabet import get_equivalent
    HAS_MAPPING = True
except ImportError:
    print("[Warning] PersianCharacterEnglishAlphabet module not found. "
          "Will use original Persian labels only.")
    HAS_MAPPING = False
    def get_equivalent(x): return x  # fallback identity function

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────

SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ─────────────────────────────────────────────
# 2. CHARACTER VOCABULARY
#    Built dynamically from .txt label files.
#    No static Persian↔English map — everything
#    is derived from the actual dataset labels.
# ─────────────────────────────────────────────

class CharVocab:
    """
    Scans all label (.txt) files and builds a class index mapping.

    Index layout:
      0, 1, 2, ...  →  real character classes (no reserved blank)

    Labels are normalised (stripped + title-cased) so minor
    capitalisation differences don't create duplicate classes.
    """

    def __init__(self, use_english_labels: bool = True) -> None:
        self.char_to_idx: dict[str, int] = {}
        self.idx_to_char: dict[int, str] = {}
        self.use_english_labels = use_english_labels and HAS_MAPPING
        self._built = False

    # ── building ──────────────────────────────

    def build_from_label_files(self, label_paths: list[str]) -> None:
        unique_chars: set[str] = set()
        for path in label_paths:
            label = self._read_label(path)
            if label:
                # Convert to English equivalent if requested
                if self.use_english_labels:
                    label = get_equivalent(label)
                unique_chars.add(label)

        if not unique_chars:
            raise RuntimeError("[Vocab] No valid labels found. Check your .txt files.")

        for i, ch in enumerate(sorted(unique_chars)):   # deterministic order
            self.char_to_idx[ch] = i
            self.idx_to_char[i]  = ch

        self._built = True
        print(f"[Vocab] {len(self.char_to_idx)} classes: {sorted(self.char_to_idx)}")
        if self.use_english_labels:
            print(f"[Vocab] Using English label equivalents for classification")

    # ── I/O helpers ───────────────────────────

    @staticmethod
    def _read_label(txt_path: str) -> str | None:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip().title()
        except Exception as e:
            print(f"[Vocab] Warning: could not read {txt_path}: {e}")
            return None

    def encode(self, label: str) -> int:
        """Return the integer class index for a label string."""
        label = label.strip().title()
        # Convert to English equivalent if requested
        if self.use_english_labels:
            label = get_equivalent(label)
        
        if label not in self.char_to_idx:
            raise ValueError(
                f"[Vocab] Unknown label '{label}'. "
                f"Known: {sorted(self.char_to_idx)}"
            )
        return self.char_to_idx[label]

    def decode(self, idx: int) -> str:
        """Return the label string for a class index."""
        return self.idx_to_char.get(idx, "<unknown>")

    # ── persistence ───────────────────────────

    def save(self, path: str) -> None:
        payload = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "use_english_labels": self.use_english_labels,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Vocab] Saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_idx = data["char_to_idx"]
        self.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
        self.use_english_labels = data.get("use_english_labels", False)
        self._built = True
        print(f"[Vocab] Loaded {len(self.char_to_idx)} classes from {path}")
        if self.use_english_labels:
            print(f"[Vocab] Using English label equivalents")

    # ── property ──────────────────────────────

    @property
    def num_classes(self) -> int:
        return len(self.char_to_idx)

# ─────────────────────────────────────────────
# 3. DATASET DISCOVERY
# ─────────────────────────────────────────────

def discover_pairs(data_dir: str) -> list[tuple[str, str]]:
    """
    Walk data_dir for image files and locate their matching .txt label.

    Search order per image:
      1. Same directory as the image  ({stem}.txt)
      2. Sister  labels/  subdirectory

    Returns a list of (image_path, label_path) tuples.
    """
    data_dir = Path(data_dir)

    img_files: list[Path] = (
        list(data_dir.rglob("*.jpg"))
        + list(data_dir.rglob("*.jpeg"))
        + list(data_dir.rglob("*.png"))
    )

    if not img_files:
        raise FileNotFoundError(f"[Dataset] No image files found under {data_dir}")

    pairs:   list[tuple[str, str]] = []
    missing: int = 0

    for img_path in sorted(img_files):
        # 1. Same directory
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            # 2. Sister labels/ directory
            alt = data_dir / "labels" / (img_path.stem + ".txt")
            if alt.exists():
                txt_path = alt
            else:
                missing += 1
                continue
        pairs.append((str(img_path), str(txt_path)))

    print(
        f"[Dataset] {len(pairs)} valid pairs found "
        f"({missing} image(s) skipped — no matching .txt)."
    )
    if not pairs:
        raise RuntimeError(
            "[Dataset] No image–label pairs found. Check your data_dir layout."
        )
    return pairs

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (stratified)
# ─────────────────────────────────────────────

def split_dataset(
    pairs: list[tuple[str, str]],
    vocab: CharVocab,
    test_size: float = 0.20,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Stratified 80/20 split so every class appears in both partitions.
    Pairs whose label is absent from the vocab are silently dropped.
    """
    valid_pairs: list[tuple[str, str]] = []
    strata:      list[str]             = []

    for img_path, txt_path in pairs:
        lbl = CharVocab._read_label(txt_path)
        if lbl:
            # Convert to English equivalent if needed
            if vocab.use_english_labels:
                lbl = get_equivalent(lbl)
            if lbl in vocab.char_to_idx:
                valid_pairs.append((img_path, txt_path))
                strata.append(lbl)

    train_pairs, test_pairs = train_test_split(
        valid_pairs,
        test_size=test_size,
        stratify=strata,
        random_state=SEED,
    )[:2]

    print(f"[Split] Train: {len(train_pairs)} | Test: {len(test_pairs)}")
    return train_pairs, test_pairs

# ─────────────────────────────────────────────
# 5. TRANSFORMS
# ─────────────────────────────────────────────

IMG_H = 64    # height fed to the CNN
IMG_W = 64    # width  fed to the CNN  (square crop suits single characters)

def get_transforms(augment: bool = False) -> transforms.Compose:
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_H, IMG_W)),
    ]

    if augment:
        ops += [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                fill=255,
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]

    return transforms.Compose(ops)

# ─────────────────────────────────────────────
# 6. DATASET
# ─────────────────────────────────────────────

class PersianCharDataset(Dataset):
    def __init__(
        self,
        pairs:   list[tuple[str, str]],
        vocab:   CharVocab,
        augment: bool = False,
    ) -> None:
        self.pairs     = pairs
        self.vocab     = vocab
        self.transform = get_transforms(augment=augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img_path, txt_path = self.pairs[idx]

        # ── load image ────────────────────────
        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"[Dataset] Cannot open {img_path}: {e}. Using blank fallback.")
            image = Image.new("RGB", (IMG_W, IMG_H), color=255)

        image = self.transform(image)   # Tensor (1, H, W)

        # ── load label ────────────────────────
        label_str = CharVocab._read_label(txt_path)
        label_idx = self.vocab.encode(label_str)

        return image, label_idx, label_str

# ─────────────────────────────────────────────
# 7. CNN MODEL
#    Four convolutional blocks + Global Average
#    Pooling + fully-connected head.
#    Designed for single-character images.
# ─────────────────────────────────────────────

def _conv_block(
    in_ch:  int,
    out_ch: int,
    pool:   bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))   # halves spatial dims
    return nn.Sequential(*layers)


class CNNClassifier(nn.Module):
    """
    Input  : (B, 1, IMG_H, IMG_W)
    Output : (B, num_classes)  — raw logits for CrossEntropyLoss
    """

    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()

        # ── feature extractor ─────────────────
        # 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → GAP
        self.features = nn.Sequential(
            _conv_block(  1,  64, pool=True),   # 64 → 32
            _conv_block( 64, 128, pool=True),   # 32 → 16
            _conv_block(128, 256, pool=True),   # 16 →  8
            _conv_block(256, 512, pool=True),   #  8 →  4
            nn.AdaptiveAvgPool2d((1, 1)),        # 4  →  1×1
        )

        # ── classifier head ───────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x   # raw logits

# ─────────────────────────────────────────────
# 8. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_one_epoch(
    model:     CNNClassifier,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    GradScaler,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=False):
            logits = model(images)          # (B, num_classes)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:     CNNClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# ─────────────────────────────────────────────
# 9. INFERENCE  (single image)
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(
    image_path: str,
    model:      CNNClassifier,
    vocab:      CharVocab,
    device:     torch.device,
    return_original: bool = False,
) -> tuple[str, float]:
    """
    Run inference on one image.
    Returns (predicted_label, confidence) where confidence is the
    max softmax probability.
    If return_original=True and mapping is available, also returns original Persian label.
    """
    model.eval()
    tf = get_transforms(augment=False)

    try:
        img = Image.open(image_path).convert("L")
    except Exception as e:
        raise RuntimeError(f"[Predict] Cannot open image {image_path}: {e}")

    img_tensor = tf(img).unsqueeze(0).to(device)   # (1, 1, H, W)

    logits     = model(img_tensor)                  # (1, num_classes)
    probs      = torch.softmax(logits, dim=1)       # (1, num_classes)
    confidence, pred_idx = probs.max(dim=1)

    label = vocab.decode(pred_idx.item())
    
    if return_original and HAS_MAPPING:
        # Try to find original Persian label (reverse mapping would require lookup)
        # For now, just return the English equivalent as we're using English labels
        pass
    
    return label, confidence.item()

# ─────────────────────────────────────────────
# 10. CHECKPOINT HELPERS
# ─────────────────────────────────────────────

def save_checkpoint(
    model:     CNNClassifier,
    optimizer: optim.Optimizer,
    epoch:     int,
    best_acc:  float,
    path:      str,
) -> None:
    torch.save(
        {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc":  best_acc,
        },
        path,
    )
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(
    model:     CNNClassifier,
    optimizer: optim.Optimizer | None,
    path:      str,
    device:    torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch    = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_acc", 0.0)
    print(f"[Checkpoint] Loaded epoch {epoch} (best_acc={best_acc:.4f}) from {path}")
    return epoch, best_acc

# ─────────────────────────────────────────────
# 11. ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Persian Character Recognition — CNN Classifier"
    )

    # paths
    p.add_argument("--data_dir",   type=str, default="./dataset",
                   help="Root directory containing image+label pairs.")
    p.add_argument("--output_dir", type=str, default="./output",
                   help="Where to save checkpoints, vocab JSON, and training log.")
    p.add_argument("--vocab_path", type=str, default=None,
                   help="Path to a saved vocab.json "
                        "(auto-resolved to output_dir/vocab.json if omitted).")

    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout",    type=float, default=0.4,
                   help="Dropout rate for the classifier head (default 0.4).")
    p.add_argument("--test_size",  type=float, default=0.20,
                   help="Fraction of data used as test set (default 0.20).")
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader worker processes.")
    
    # label options
    p.add_argument("--use_english_labels", action="store_true", default=True,
                   help="Convert Persian labels to English equivalents using mapping module")
    p.add_argument("--no_english_labels", dest="use_english_labels", action="store_false",
                   help="Use original Persian labels instead of English equivalents")

    # misc
    p.add_argument("--resume",  type=str, default=None,
                   help="Checkpoint path to resume training from.")
    p.add_argument("--predict", type=str, default=None,
                   help="Image path for inference (skips training).")

    return p.parse_args()

# ─────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────

def main() -> None:

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_save_path = args.vocab_path or str(output_dir / "vocab.json")
    best_ckpt_path  = str(output_dir / "best_model.pth")
    last_ckpt_path  = str(output_dir / "last_model.pth")
    log_path        = output_dir / "training_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── vocab ─────────────────────────────────
    vocab = CharVocab(use_english_labels=args.use_english_labels)

    if args.predict:
        # Inference mode — load pre-built vocab
        if not os.path.exists(vocab_save_path):
            raise FileNotFoundError(
                f"[Vocab] No vocab file at '{vocab_save_path}'. "
                "Train the model first, or pass --vocab_path."
            )
        vocab.load(vocab_save_path)
    else:
        # Training mode — discover dataset + build vocab from scratch
        all_pairs = discover_pairs(args.data_dir)
        vocab.build_from_label_files([p[1] for p in all_pairs])
        vocab.save(vocab_save_path)

    # ── model ─────────────────────────────────
    model = CNNClassifier(
        num_classes=vocab.num_classes,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    # ── inference only ────────────────────────
    if args.predict:
        ckpt_path = args.resume or best_ckpt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[Predict] No checkpoint at '{ckpt_path}'. Train first."
            )
        load_checkpoint(model, None, ckpt_path, device)

        label, confidence = predict(args.predict, model, vocab, device)

        # ✅ GET ENGLISH EQUIVALENT
        equivalent = get_equivalent(label)

        print(f"\n{'─'*45}")
        print(f"  Image              : {args.predict}")
        print(f"  Prediction         : {label}")
        print(f"  English Equivalent : {equivalent}")   # ✅ NEW LINE
        print(f"  Confidence         : {confidence*100:.1f}%")

        if HAS_MAPPING and vocab.use_english_labels:
            print(f"  Note       : Model was trained on English label equivalents")

        print(f"{'─'*45}\n")
        return

    # ── training ──────────────────────────────
    train_pairs, test_pairs = split_dataset(
        all_pairs, vocab, test_size=args.test_size
    )

    train_dataset = PersianCharDataset(train_pairs, vocab, augment=True)
    test_dataset  = PersianCharDataset(test_pairs,  vocab, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")

    start_epoch = 0
    best_acc    = 0.0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_acc = load_checkpoint(
            model, optimizer, args.resume, device
        )
        start_epoch += 1

    # ── training loop ─────────────────────────
    log_lines = ["epoch,train_loss,test_loss,test_acc,lr,elapsed_s"]

    print(f"\n{'═'*60}")
    print(f"  Epochs      : {args.epochs}   Batch : {args.batch_size}   LR : {args.lr}")
    print(f"  Train       : {len(train_dataset):,}   Test : {len(test_dataset):,}")
    print(f"  Classes     : {vocab.num_classes}")
    print(f"  Image size  : {IMG_H}×{IMG_W}")
    if vocab.use_english_labels and HAS_MAPPING:
        print(f"  Labels      : English equivalents (mapped from Persian)")
    else:
        print(f"  Labels      : Original Persian")
    print(f"{'═'*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0         = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"lr={current_lr:.2e}  "
            f"train_loss={train_loss:.4f}  "
            f"test_loss={test_loss:.4f}  "
            f"test_acc={test_acc*100:.2f}%  "
            f"({elapsed:.1f}s)"
        )

        log_lines.append(
            f"{epoch+1},{train_loss:.6f},{test_loss:.6f},"
            f"{test_acc:.6f},{current_lr:.2e},{elapsed:.1f}"
        )

        save_checkpoint(model, optimizer, epoch, best_acc, last_ckpt_path)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, best_acc, best_ckpt_path)
            print(f"  ★ New best accuracy: {best_acc*100:.2f}%")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n{'═'*60}")
    print(f"  Training complete.")
    print(f"  Best test accuracy : {best_acc*100:.2f}%")
    print(f"  Best model         : {best_ckpt_path}")
    print(f"  Vocab              : {vocab_save_path}")
    print(f"  Training log       : {log_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()