# Persian Alphabet Character Recognition (CNN - PyTorch)

A deep learning-based project for recognizing Persian alphabet characters using a Convolutional Neural Network (CNN). This project includes dataset preparation, training, and inference.

---

## 📌 Project Overview

This system:

* Converts labeled image datasets into a usable format
* Trains a CNN model for Persian character classification
* Supports prediction on new images

---

## 📁 Project Structure

```
project/
│
├── PrepareDataset.py
├── PersianCharacterRecognitionModel.py
├── requirements.txt
│
├── PersianAlphabetDataset/   # Raw dataset (input)
│   ├── 01-alef/
│   ├── 02-beh/
│   └── ...
│
├── dataset/                 # Processed dataset (used for training)
│
└── output/
    ├── best_model.pth
    ├── last_model.pth
    ├── vocab.json
    └── training_log.csv
```

---

## ⚙️ Installation

### 1. Clone or Download the Project

```
git clone https://github.com/iFabsss/PersianCharacterRecognition.git
cd <project-folder>
```

### 2. Create Virtual Environment (Recommended)

```
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## 🧹 Step 1: Prepare Dataset

### Dataset Format (Before)

Your dataset should be structured like this:

```
PersianAlphabetDataset/
├── 01-alef/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── 02-beh/
│   └── ...
```

### Run Dataset Preparation Script

```
python PrepareDataset.py
```

### What it does:

* Creates `.txt` label files for each image
* Example:

  ```
  1.jpg  →  1.txt  (contains "Alef")
  ```
* Generates:

  ```
  Persian Characters List.txt
  ```

---

## 📦 Step 2: Dataset Format (After)

After preparation, each folder will contain:

```
01-alef/
├── 1.jpg
├── 1.txt
├── 2.jpg
├── 2.txt
```

You can either:

* Use this structure directly (`flat`)
* OR move files into:

  ```
  dataset/
  ├── images/
  └── labels/
  ```

---

## 🧠 Step 3: Train the Model

Run:

```
python PersianCharacterRecognitionModel.py --data_dir ./dataset --epochs 30 --batch_size 32
```

### Parameters

| Argument       | Description               |
| -------------- | ------------------------- |
| `--data_dir`   | Path to dataset           |
| `--epochs`     | Number of training epochs |
| `--batch_size` | Batch size                |
| `--lr`         | Learning rate             |
| `--workers`    | DataLoader workers        |

---

## 📊 Training Output

During training, you will see:

```
Epoch 001/30  train_loss=...  test_loss=...  test_acc=...
```

Files saved in `output/`:

* `best_model.pth` → best accuracy model
* `last_model.pth` → latest checkpoint
* `vocab.json` → label mappings
* `training_log.csv` → metrics per epoch

---

## 🔍 Step 4: Run Inference (Prediction)

```
python PersianCharacterRecognitionModel.py --predict path/to/image.jpg --output_dir ./output
```

### Output Example:

```
Image      : test.jpg
Prediction : Alef
Confidence : 97.5%
```

---

## 🧪 Model Details

* Input: `(1, 64, 64)` grayscale image
* Architecture:

  * 4 Convolutional Blocks
  * Batch Normalization
  * ReLU Activation
  * MaxPooling
  * Global Average Pooling
  * Fully Connected Layers
* Loss Function: CrossEntropyLoss (with label smoothing)
* Optimizer: AdamW
* Scheduler: Cosine Annealing

---

## ⚠️ Notes & Tips

### 1. Autocast Issue (Important)

If training crashes with AMP/autocast:

* Use:

  ```python
  with autocast(enabled=False):
  ```
* Mixed precision may not work on all setups

---

### 2. Windows Multiprocessing Fix

If you get multiprocessing errors:

* Set workers to 0:

  ```
  --workers 0
  ```

---

### 3. Grayscale Input

Model expects:

```
(1, H, W)
```

So always use grayscale images.

---

### 4. Overfitting Check

| Scenario              | Meaning                                           |
| --------------------- | ------------------------------------------------- |
| Train ↓, Test ↑       | Overfitting                                       |
| Train high, Test high | Underfitting                                      |
| Train ≈ Test          | Good                                              |
| Test < Train          | Strong generalization (often due to augmentation) |

---

## 📌 Requirements

See `requirements.txt`:

```
torch
torchvision
Pillow
numpy
scikit-learn
```

---

## 🚀 Future Improvements

* Add CRNN + CTC for sequence recognition
* Use larger datasets
* Add real-time webcam inference
* Deploy as web/mobile app

---

## 👨‍💻 Author

Developed as part of a Persian Character Recognition project using PyTorch.

---

## 📄 License

This project is for educational and research purposes.
