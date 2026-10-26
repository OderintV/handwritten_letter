# Handwritten Letter Recognition (A-Z)

Give the program a photo of a handwritten English letter and it tells you which letter it is.
It is a convolutional neural network (CNN) trained on the **EMNIST Letters** dataset, plus an image
pre-processing pipeline that makes ordinary phone photos look like the training data.

**Headline result: 95.50% accuracy on the 20,800-image EMNIST test set (99.69% top-3), and 29/30 (96.7%) on our own photos.**
Peer projects on this task typically report around 80-90% accuracy, so this model is roughly 5-15 points higher.

---

## 1. Results at a glance

| Dataset | Images | Accuracy | Other metrics |
|---|---|---|---|
| EMNIST Letters test set (never seen in training) | 20,800 | **95.50%** | Top-3 99.69%, macro F1 0.9551, macro precision 0.9553, macro recall 0.9550 |
| EMNIST validation split (used to pick the best epoch) | 12,480 | 95.49% | |
| `my_samples` - our own labelled photos (one letter each) | 30 | **97%** (29/30) | Only error: a plain vertical stroke "l" read as "I" |
| `my_samples2` - harder photos (other people, red ink, shadows, 4-letter photo) | 11 usable | run it and see section 5 | |

Best letters (F1): Z 0.995, M 0.993, P 0.992, W 0.991. Weakest: L 0.752, I 0.763, G 0.886, Q 0.896.
I/L is weak because EMNIST merges upper and lower case, so a capital "I", a lowercase "l" and a simple line look identical.

---

## 2. Project files - what each program does

| File | What it does |
|---|---|
| `model.py` | Defines the CNN (3 conv blocks + 2 fully connected layers, about 0.6 M parameters) and the A-Z label list. |
| `train.py` | Downloads EMNIST Letters, trains the CNN for 15 epochs, keeps the best checkpoint, evaluates on the test set and saves `model.pt`, `metrics.json` and `confusion_matrix.png`. |
| `preprocess.py` | Converts a photo into 28x28 EMNIST-style letter images: grayscale, lighting correction, thresholding, letter detection, cropping, resizing. |
| `predict.py` | Loads `model.pt`, runs the pre-processing, classifies each letter and prints the prediction with the top-3 guesses. |
| `evaluate_custom.py` | Runs `predict` on a whole folder of labelled photos and prints per-image results, accuracy and a per-letter report. |
| `my_samples2_labels.csv` | The true labels for the `my_samples2` folder (copy it in as `labels.csv`, see section 5). |
| `requirements.txt` | Python packages needed. |

Files created by training: `model.pt` (the trained weights), `metrics.json`, `confusion_matrix.png`.
Files created by `--debug`: `debug_binary.png`, `debug_model_input.png`.

---

## 3. How to run it (macOS Terminal)

### One-time setup
```bash
cd ~/handwritten_letter             # the project folder
python3 -m venv venv                # create an isolated Python environment
source venv/bin/activate            # activate it (prompt now starts with "(venv)")
pip install -r requirements.txt     # install torch, torchvision, opencv, scikit-learn, matplotlib, numpy
```
Every time you open a new Terminal window, run the first and third lines again:
```bash
cd ~/handwritten_letter
source venv/bin/activate
```

### Train the model (only needed once; skip if `model.pt` already exists)
```bash
python train.py --epochs 15
```
Downloads EMNIST the first time (about 550 MB), uses the Apple GPU (MPS) automatically if available,
and prints validation accuracy after each epoch and the full test report at the end.

### Predict one letter from a photo
```bash
python predict.py letter.jpg --single --debug
open debug_model_input.png          # see exactly what the model saw
```
### Predict several separate letters in one photo (read left to right) [remove from mysamples_2 and then run]
```bash
python predict.py cats.jpg
```
### Evaluate a folder of labelled photos
```bash
python evaluate_custom.py my_samples/
```
### Show the saved EMNIST metrics
```bash
python -c "import json; m=json.load(open('metrics.json')); print('Test accuracy:', round(m['test_accuracy']*100,2), '%'); print('Top-3:', round(m['top3_accuracy']*100,2), '%'); print('Macro F1:', round(m['macro_f1'],4))"
open confusion_matrix.png
```
**Notes:** iPhone `.heic` photos must be converted first: `sips -s format jpeg IMG.HEIC --out IMG.jpg`.
If `python` is not found, activate the venv (`source venv/bin/activate`) or use `python3`.

---

## 4. How it works

### 4.1 Training (`train.py`)
| Setting | Value |
|---|---|
| Dataset | EMNIST Letters: 124,800 train + 20,800 test images, 28x28 grayscale, 26 classes (upper/lower case merged) |
| Split | 112,320 train / 12,480 validation (random 10%) / 20,800 test |
| Epochs | **15** |
| Batch size | 256 |
| Optimiser | AdamW, weight decay 1e-4, OneCycle learning-rate schedule (peak 0.003) |
| Loss | Cross-entropy with label smoothing 0.1 |
| Augmentation | Random rotation +-12 deg, shift 10%, scale 0.85-1.1, shear 8 deg |
| Regularisation | Batch normalisation + dropout (0.1 / 0.2 / 0.3 in the conv blocks, 0.4 before the output) |
| Model selection | Best validation accuracy checkpoint is saved, then tested once on the test set |
| Hardware | Apple Silicon GPU via PyTorch MPS |
| Seed | 0 (reproducible split) |

Validation accuracy went 91.2% (epoch 1) to about 95.5% (epochs 13-15); test accuracy 95.50% matches it, so there is no overfitting.

### 4.2 Pre-processing a photo (`preprocess.py`) - the grayscale pipeline
A phone photo is a large coloured image; the network only understands small 28x28 white-on-black letters, so every image goes through these steps:

1. **Convert to grayscale first.** The image is read as grayscale (colour is thrown away), so the network never sees colour. This is why **red ink works**: red becomes a dark gray, which is still much darker than the paper. Very large photos are shrunk to at most 1000 px.
2. **Blur lightly** (5x5 Gaussian) to remove camera noise.
3. **Remove shadows and uneven light.** The paper background is estimated (a large morphological closing removes the thin ink) and the image is divided by it. This flattens shadows so the ink stands out.
4. **Threshold with Otsu's method** to get a black/white image where ink is white. If the result is mostly white (light ink on dark background) it is automatically inverted.
5. **Clean up** specks with a small morphological opening.
6. **Find the letters** (section 4.3).
7. **Make each letter look like EMNIST:** crop the letter, pad to a square (keeps the shape), measure the pen thickness with a distance transform and thicken thin pen lines to match EMNIST strokes, add a small border, shrink to 28x28 and apply a slight blur. EMNIST was built the same way.
8. **Normalise and classify:** pixel values are scaled and normalised with the EMNIST mean/std, the CNN returns 26 probabilities, and the top 3 are printed.

### 4.3 One letter vs several letters (how `--single` works, and how the 4-letter photo works)
After step 5 the program finds every separate blob of ink (connected components) and ignores tiny specks.

* **`--single`:** all blobs are merged into one box, so the whole image is one letter. This keeps pieces together, e.g. the dot of an "i" or the bar of a "T", and gives one prediction.
* **Without `--single` (multi-letter mode):**
  1. Each blob gets its own bounding box.
  2. Boxes that overlap horizontally by more than 30% are merged, so the dot of an "i"/"j" stays attached to its stem.
  3. Tiny leftovers (under 20% of the biggest letter) are dropped.
  4. Boxes are sorted **left to right**.
  5. Each box is turned into its own 28x28 image and classified separately; the predictions are joined into one string.

  That is how `cats.jpg` (the letters c, a, t, s written with gaps) is read as four separate letters and printed as one word. Our check on that photo found exactly 4 letters, in order.

Limits: letters must not touch (touching letters become one blob), they must be on one line, and cursive is not supported.

---

## 5. Test photos

### `my_samples` - first run, labelled
30 photos of single letters on plain paper, each file named with its true letter (`a.jpg`, `b1.jpg`, ...).
```bash
python evaluate_custom.py my_samples/
```
Result: **29/30 correct = 96.7%**. The only mistake: `l.jpg` (a plain vertical line) predicted as I - the same I/L ambiguity seen in the EMNIST report.

### `my_samples2` - harder, "real world" photos
Other people's letters, red ink, black ink, a photo with a shadow, and one photo with four letters. Contents:

| File | What it is | True label |
|---|---|---|
| `IMG_0002.jpeg` | red marker "c" on a gray gradient background | C |
| `IMG_33.jpeg` | red marker "B" on a gray background | B |
| `IMG_1.jpeg`, `IMG_2.jpeg`, `IMG_3.jpeg`, `IMG_4.jpeg` | black ink letters from another person (a, t, S, c) | A, T, S, C |
| `IMG_1234.jpeg`, `letter.jpg`, `letter.jpeg` | the letter "A" in thin pen (same photo saved three ways) | A |
| `z_shadow.jpg` | blurry "z" with a strong shadow across the paper | Z |
| `cats.jpg` | four separate letters in one photo | CATS |
| `IMG_0243.heic`, `IMG_0245.heic` | iPhone format - convert to jpg first, then add them to `labels.csv` | |

To run it:
```bash
cp my_samples2_labels.csv my_samples2/labels.csv
python evaluate_custom.py my_samples2/
```
`labels.csv` tells the program the true answer for each file (needed because these files are named `IMG_...`).
Single letters are scored as letters; `cats.jpg` is run in multi-letter mode and compared as a word.

Pre-processing check on this set: every single-letter photo (including red ink and the shadow photo) produced one clean white-on-black letter, and `cats.jpg` was split into 4 letters.

---

## 6. Limitations
* Only A-Z; digits and symbols are forced into a letter.
* Upper and lower case cannot be told apart (EMNIST merges them).
* I / L (and sometimes G / Q) are confused.
* No cursive, no touching letters.
* The custom photo sets are small and were written by a few people - a sanity check, not a benchmark.

## 7. Ideas to improve
EMNIST ByClass (keeps case), a "not a letter" reject class, more writers for the photo test set, and the IAM database for whole words.

## 8. Troubleshooting
| Problem | Fix |
|---|---|
| `zsh: command not found: python` | `source venv/bin/activate` (and `cd` into the project first) |
| `FileNotFoundError: x.jpg` | the photo is not in the current folder - `mv` it in or give the full path |
| `can't open/read file` on a `.heic` | convert with `sips -s format jpeg IMG.HEIC --out IMG.jpg` |
| Venv stopped working after moving the folder | `rm -rf venv`, then repeat the one-time setup |
| SSL error downloading EMNIST | `pip install certifi` then `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` |
