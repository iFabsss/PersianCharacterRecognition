"""
Persian Alphabet Character Recognition
CNN Classifier -- PyTorch

Dataset layout:
  dataset/
    images/
      1alef.jpg
      aleph_1.png
    labels/
      1alef.txt           (contains a raw Persian character)
      aleph_1.txt         (contains a raw Persian character)
    Persian Characters List.txt   <- one Persian character per line, 68 total

Usage:
  # Train (char list auto-resolved to <data_dir>/Persian Characters List.txt)
  python PersianCharacterRecognitionModel.py --data_dir ./dataset --epochs 30

  # Train with an explicit char list path
  python PersianCharacterRecognitionModel.py --data_dir ./dataset \\
      --char_list "./dataset/Persian Characters List.txt" --epochs 30

  # Predict
  python PersianCharacterRecognitionModel.py --predict path/to/image.jpg \\
      --output_dir ./output
"""

import os
import json
import time
import random
import argparse
import cv2
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from sklearn.model_selection import train_test_split

from PIL import ImageFilter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler

# Import Persian-to-English mapping (used only for display at prediction time)
try:
    from PersianCharacterEnglishAlphabet import get_equivalent, get_symbol, get_ambiguous_hint
    HAS_MAPPING = True
except ImportError:
    print("[Warning] PersianCharacterEnglishAlphabet module not found. "
          "English equivalents will not be shown in predictions.")
    HAS_MAPPING = False
    def get_equivalent(x): return x  # fallback: identity

# -----------------------------------------------------------------------------
# 1. REPRODUCIBILITY
# -----------------------------------------------------------------------------

SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------------------------------------------------------------
# 2. CHARACTER VOCABULARY
#
#    Built from "Persian Characters List.txt" -- NOT inferred from label files.
#    This guarantees exactly 68 classes regardless of which datasets are mixed.
#
#    Why not get_equivalent() here?
#      Two datasets each cover 34 distinct Persian characters.  When both are
#      mapped to English, they produce the SAME 34 English names, making the
#      model see only 34 classes instead of 68.  Using raw Persian Unicode as
#      the class identifier avoids that collision entirely.
# -----------------------------------------------------------------------------

class CharVocab:
    """
    Fixed-vocabulary class list loaded from a canonical text file.

    Index layout:
      0, 1, 2, ...  ->  Persian characters in file order (not sorted).

    Label .txt files must contain exactly one of the listed characters
    (raw Unicode, stripped of surrounding whitespace).
    """

    def __init__(self) -> None:
        self.char_to_idx: dict[str, int] = {}
        self.idx_to_char: dict[int, str] = {}
        self._built = False

    # -- building --------------------------------------------------------------

    def build_from_character_list(self, list_path: str) -> None:
        """
        Read the canonical Persian Characters List and register every
        non-blank line as a class, preserving file order.

        Parameters
        ----------
        list_path : str
            Path to "Persian Characters List.txt".
            Each line must contain exactly one Persian character.
        """
        list_path = Path(list_path)
        if not list_path.exists():
            raise FileNotFoundError(
                f"[Vocab] Character list not found: {list_path}\n"
                f"        Pass --char_list <path> or place the file at the "
                f"default location (<data_dir>/Persian Characters List.txt)."
            )

        chars: list[str] = []
        seen:  set[str]  = set()

        with open(list_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                ch = raw.strip()
                if not ch:
                    continue                     # skip blank lines
                if ch in seen:
                    print(f"[Vocab] Warning: duplicate '{ch}' on line {lineno} -- skipped.")
                    continue
                chars.append(ch)
                seen.add(ch)

        if not chars:
            raise RuntimeError(
                f"[Vocab] No characters found in {list_path}. "
                "Is the file empty or using an unexpected encoding?"
            )

        for i, ch in enumerate(chars):          # order from file, not sorted()
            self.char_to_idx[ch] = i
            self.idx_to_char[i]  = ch

        self._built = True
        print(f"[Vocab] {len(self.char_to_idx)} classes loaded from {list_path}")
        print(f"[Vocab] Characters: {list(self.char_to_idx.keys())}")

    # -- I/O helpers -----------------------------------------------------------

    @staticmethod
    def _read_label(txt_path: str) -> "str | None":
        """Read a label .txt and return the raw Persian character (stripped)."""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()          # do NOT title-case Persian text
        except Exception as e:
            print(f"[Vocab] Warning: could not read {txt_path}: {e}")
            return None

    def encode(self, label: str) -> int:
        """Return the integer class index for a raw Persian character string."""
        label = label.strip()                    # no case-folding for Persian
        if label not in self.char_to_idx:
            raise ValueError(
                f"[Vocab] Unknown label '{label}'. "
                f"It is not in the Persian Characters List.\n"
                f"        Known chars: {list(self.char_to_idx.keys())}"
            )
        return self.char_to_idx[label]

    def decode(self, idx: int) -> str:
        """Return the Persian character for a class index."""
        return self.idx_to_char.get(idx, "<unknown>")

    # -- persistence -----------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Vocab] Saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_idx = data["char_to_idx"]
        self.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
        self._built = True
        print(f"[Vocab] Loaded {len(self.char_to_idx)} classes from {path}")

    # -- property --------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.char_to_idx)

# -----------------------------------------------------------------------------
# 3. DATASET DISCOVERY
# -----------------------------------------------------------------------------

def _resolve_subdirs(data_dir: Path) -> tuple[Path, Path]:
    """
    Locate the images/ and labels/ subdirectories under data_dir.

    Layout A (preferred):
        data_dir/
            images/   <- image files
            labels/   <- matching .txt files

    Layout B (flat fallback):
        data_dir/
            1alef.jpg + 1alef.txt  (same folder)

    Returns (images_dir, labels_dir).
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if images_dir.is_dir() and labels_dir.is_dir():
        print(f"[Dataset] images -> {images_dir}")
        print(f"[Dataset] labels -> {labels_dir}")
        return images_dir, labels_dir

    print(
        f"[Dataset] Warning: could not find both 'images/' and 'labels/' "
        f"under {data_dir}.\n"
        f"          Falling back to flat layout -- searching {data_dir} for "
        f"image+label pairs in the same directory."
    )
    return data_dir, data_dir


def discover_pairs(
    data_dir: str,
    vocab: "CharVocab | None" = None,
) -> list[tuple[str, str]]:
    """
    Walk images/ for image files and match each to its .txt label in labels/.

    Matching is purely by filename stem (no extension):
        images/1alef.jpg   ->  labels/1alef.txt
        images/aleph_1.png ->  labels/aleph_1.txt

    If `vocab` is supplied, any label whose content is NOT in the
    vocab's character list is skipped with a warning.  This is what
    prevents mismatched cross-dataset labels from silently corrupting
    the class indices.

    Returns a list of (image_path, label_path) tuples.
    """
    data_dir   = Path(data_dir)
    images_dir, labels_dir = _resolve_subdirs(data_dir)

    img_files: list[Path] = sorted(
        list(images_dir.rglob("*.jpg"))
        + list(images_dir.rglob("*.jpeg"))
        + list(images_dir.rglob("*.png"))
    )

    if not img_files:
        raise FileNotFoundError(
            f"[Dataset] No image files found under {images_dir}"
        )

    pairs:        list[tuple[str, str]] = []
    missing:      int = 0
    out_of_vocab: int = 0

    for img_path in img_files:
        try:
            relative_stem = img_path.relative_to(images_dir).with_suffix("")
        except ValueError:
            relative_stem = Path(img_path.stem)

        txt_path = (labels_dir / relative_stem).with_suffix(".txt")

        if not txt_path.exists():
            missing += 1
            if missing <= 10:
                print(f"[Dataset] No label for {img_path.name} (expected {txt_path})")
            continue

        if vocab is not None:
            raw_label = CharVocab._read_label(str(txt_path))
            if raw_label is None or raw_label not in vocab.char_to_idx:
                out_of_vocab += 1
                if out_of_vocab <= 10:
                    print(f"[Dataset] Skipping {img_path.name}: "
                          f"label '{raw_label}' not in Persian Characters List.")
                continue

        pairs.append((str(img_path), str(txt_path)))

    if missing > 10:
        print(f"[Dataset] ... and {missing - 10} more image(s) without labels.")
    if out_of_vocab > 10:
        print(f"[Dataset] ... and {out_of_vocab - 10} more with out-of-vocab labels.")

    print(
        f"[Dataset] {len(pairs)} valid pairs found "
        f"({missing} skipped - no .txt | {out_of_vocab} skipped - not in char list)."
    )

    if not pairs:
        raise RuntimeError(
            "[Dataset] No image-label pairs found.\n"
            "  Expected layout:\n"
            "    <data_dir>/images/<n>.jpg  (or .jpeg / .png)\n"
            "    <data_dir>/labels/<n>.txt  (raw Persian character inside)\n"
            "  Label .txt files must contain one of the 68 Persian characters\n"
            "  listed in 'Persian Characters List.txt'.\n"
            "  Make sure filenames (without extension) match exactly."
        )
    return pairs

# -----------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT  (stratified)
# -----------------------------------------------------------------------------

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
        lbl = CharVocab._read_label(txt_path)   # raw Persian character
        if lbl and lbl in vocab.char_to_idx:
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

# -----------------------------------------------------------------------------
# 5. TRANSFORMS
# -----------------------------------------------------------------------------

IMG_H = 128    # height fed to the CNN
IMG_W = 128    # width  fed to the CNN

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
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]

    return transforms.Compose(ops)

# -----------------------------------------------------------------------------
# 6. DATASET
# -----------------------------------------------------------------------------

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

        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"[Dataset] Cannot open {img_path}: {e}. Using blank fallback.")
            image = Image.new("L", (IMG_W, IMG_H), color=255)

        image = self.transform(image)   # Tensor (1, H, W)

        label_str = CharVocab._read_label(txt_path)
        label_idx = self.vocab.encode(label_str)

        return image, label_idx, label_str

# -----------------------------------------------------------------------------
# 7. CNN MODEL
# -----------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class CNNClassifier(nn.Module):
    """
    Input  : (B, 1, IMG_H, IMG_W)
    Output : (B, num_classes)  -- raw logits for CrossEntropyLoss
    """

    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()

        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> GAP
        self.features = nn.Sequential(
            _conv_block(  1,  64, pool=True),
            _conv_block( 64, 128, pool=True),
            _conv_block(128, 256, pool=True),
            _conv_block(256, 512, pool=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

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
        return self.classifier(self.features(x))

# -----------------------------------------------------------------------------
# 8. TRAINING & EVALUATION
# -----------------------------------------------------------------------------

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
            logits = model(images)
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

# -----------------------------------------------------------------------------
# 9. INFERENCE  (single image)
# -----------------------------------------------------------------------------


@torch.no_grad()
def predict(image_path, model, vocab, device, tta_runs: int = 7):
    model.eval()
    tf_base = get_transforms(augment=False)

    # Small TTA augmentation — subtle enough not to distort,
    # varied enough to produce meaningful spread across runs
    tf_tta = transforms.Compose([
        transforms.RandomRotation(degrees=8, fill=255),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.93, 1.07),
            fill=255,
        ),
    ])

    # --- Load & binarize ---
    try:
        img = Image.open(image_path).convert("L")
    except Exception as e:
        raise RuntimeError(f"[Predict] Cannot open image {image_path}: {e}")

    img_cv = np.array(img)
    _, binary = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(binary) < 128:
        binary = 255 - binary

    img = Image.fromarray(binary)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    max_dim = max(img.size)
    padded = Image.new("L", (max_dim, max_dim), 255)
    padded.paste(img, ((max_dim - img.size[0]) // 2,
                       (max_dim - img.size[1]) // 2))

    # --- TTA: collect raw probability vectors across runs ---
    # Shape will be (tta_runs, num_classes)
    all_probs = []

    # Run 0: clean baseline (no augmentation)
    base_tensor = tf_base(padded).unsqueeze(0).to(device)
    base_probs  = torch.softmax(model(base_tensor), dim=1)  # (1, C)
    all_probs.append(base_probs)

    # Runs 1..tta_runs-1: augmented variants
    for _ in range(tta_runs - 1):
        aug_img    = tf_tta(padded)                              # PIL → PIL
        aug_tensor = tf_base(aug_img).unsqueeze(0).to(device)   # normalize
        aug_probs  = torch.softmax(model(aug_tensor), dim=1)    # (1, C)
        all_probs.append(aug_probs)

    # Stack → (tta_runs, num_classes)
    all_probs = torch.cat(all_probs, dim=0)

    # --- Consensus: find the modal predicted class ---
    run_preds = all_probs.argmax(dim=1)          # (tta_runs,) — one pred per run
    pred_counts = torch.bincount(run_preds, minlength=all_probs.shape[1])

    max_count   = pred_counts.max().item()
    modal_idxs  = (pred_counts == max_count).nonzero(as_tuple=True)[0]

    if len(modal_idxs) == 1:
        # Clear winner — use the mode
        final_idx = modal_idxs[0].item()
    else:
        # Tie — fall back to mean probability across all runs
        # This gives the class whose average confidence is highest
        mean_probs = all_probs.mean(dim=0)           # (num_classes,)
        # Only consider the tied classes, not all classes
        tied_probs = mean_probs[modal_idxs]
        final_idx  = modal_idxs[tied_probs.argmax()].item()

    final_confidence = all_probs[:, final_idx].mean().item()
    label = vocab.decode(final_idx)
    return label, final_confidence

# -----------------------------------------------------------------------------
# 10. CHECKPOINT HELPERS
# -----------------------------------------------------------------------------

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
    print(f"[Checkpoint] Saved -> {path}")


def load_checkpoint(
    model:     CNNClassifier,
    optimizer: "optim.Optimizer | None",
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

# -----------------------------------------------------------------------------
# 11. ARGUMENT PARSER
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Persian Character Recognition -- CNN Classifier"
    )

    # paths
    p.add_argument("--data_dir",   type=str, default="./dataset",
                   help=(
                       "Root directory.  Expected layout:\n"
                       "  <data_dir>/images/<n>.jpg  (or .jpeg / .png)\n"
                       "  <data_dir>/labels/<n>.txt  (raw Persian char inside)\n"
                       "  <data_dir>/Persian Characters List.txt\n"
                       "A flat layout (images and labels in the same folder) is also "
                       "accepted as a fallback."
                   ))
    p.add_argument("--char_list",  type=str, default=None,
                   help=(
                       "Path to the Persian Characters List .txt file "
                       "(one Persian character per line, 68 total). "
                       "Defaults to <data_dir>/Persian Characters List.txt."
                   ))
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

    # misc
    p.add_argument("--resume",  type=str, default=None,
                   help="Checkpoint path to resume training from.")
    p.add_argument("--predict", type=str, default=None,
                   help="Image path for inference (skips training).")

    return p.parse_args()

# -----------------------------------------------------------------------------
# 12. MAIN
# -----------------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_save_path = args.vocab_path or str(output_dir / "vocab.json")
    best_ckpt_path  = str(output_dir / "best_model.pth")
    last_ckpt_path  = str(output_dir / "last_model.pth")
    log_path        = output_dir / "training_log.csv"

    # Resolve character list path
    char_list_path = args.char_list or str(
        Path(args.data_dir) / "Persian Characters List.txt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # -- vocab -----------------------------------------------------------------
    vocab = CharVocab()

    if args.predict:
        # Inference mode -- load pre-built vocab
        if not os.path.exists(vocab_save_path):
            raise FileNotFoundError(
                f"[Vocab] No vocab file at '{vocab_save_path}'. "
                "Train the model first, or pass --vocab_path."
            )
        vocab.load(vocab_save_path)
    else:
        # Training mode -- build vocab from the canonical character list
        vocab.build_from_character_list(char_list_path)
        vocab.save(vocab_save_path)

    # -- model -----------------------------------------------------------------
    model = CNNClassifier(
        num_classes=vocab.num_classes,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    # -- inference only --------------------------------------------------------
    if args.predict:
        ckpt_path = args.resume or best_ckpt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"[Predict] No checkpoint at '{ckpt_path}'. Train first."
            )
        load_checkpoint(model, None, ckpt_path, device)

        persian_char, confidence = predict(args.predict, model, vocab, device)

        if HAS_MAPPING:
            english_equiv = get_equivalent(persian_char)
            persian_symbol = get_symbol(persian_char)
            ambiguous_hint = get_ambiguous_hint(persian_char)
        else:
            english_equiv = "N/A"
            persian_symbol = "N/A"
            ambiguous_hint = None

        print(f"\n{'-' * 45}")
        print(f"  Image               : {args.predict}")
        print(f"  Prediction (Persian): {persian_char}  {persian_symbol}")
        print(f"  English Equivalent  : {english_equiv}")
        print(f"  Confidence          : {confidence * 100:.1f}%")
        if ambiguous_hint:
            print(f"  Note                : {ambiguous_hint}")
        print(f"{'-' * 45}\n")
        return

    # -- training --------------------------------------------------------------
    # discover_pairs validates each label against the vocab, skipping unknowns
    all_pairs = discover_pairs(args.data_dir, vocab=vocab)

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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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

    # -- training loop ---------------------------------------------------------
    log_lines = ["epoch,train_loss,test_loss,test_acc,lr,elapsed_s"]

    print(f"\n{'='*60}")
    print(f"  Epochs      : {args.epochs}   Batch : {args.batch_size}   LR : {args.lr}")
    print(f"  Train       : {len(train_dataset):,}   Test : {len(test_dataset):,}")
    print(f"  Classes     : {vocab.num_classes}")
    print(f"  Image size  : {IMG_H}x{IMG_W}")
    print(f"  Char list   : {char_list_path}")
    print(f"{'='*60}\n")

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
            print(f"  * New best accuracy: {best_acc*100:.2f}%")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best test accuracy : {best_acc*100:.2f}%")
    print(f"  Best model         : {best_ckpt_path}")
    print(f"  Vocab              : {vocab_save_path}")
    print(f"  Training log       : {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()