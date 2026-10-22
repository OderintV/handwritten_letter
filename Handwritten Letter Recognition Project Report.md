# Handwritten Letter Recognition (A to Z)

A CNN trained on EMNIST Letters, with a preprocessing pipeline that works on real photos of handwriting.

## 1. Problem Statement

Given an image of a handwritten English letter, predict which letter (A to Z) it represents. The system must run on a new photo of someone else's handwriting (the grader's), so it has to work beyond the training dataset and not only on clean benchmark images.

## 2. Approach in One Paragraph

We trained a convolutional neural network (CNN) on the EMNIST Letters dataset, which contains 28×28 grayscale images of handwritten letters. Because real photos look very different from EMNIST images (colour, lighting, paper, thin pen strokes, large size), we built a preprocessing step that converts any photo into the same format EMNIST uses. We then evaluated the model on two datasets: the official EMNIST test set and our own handwriting photos.

## 3. Dataset

| Item | Detail |
| --- | --- |
| Dataset | EMNIST Letters (NIST derived) |
| Image format | 28×28 grayscale, white letter on black background |
| Classes | 26 (A to Z). Upper and lower case are merged into one class |
| Training images | 112,320 |
| Validation images | 12,480 (a random 10% held out from the original training split) |
| Test images | 20,800 (800 per letter, never used during training) |

A note on EMNIST: its images are stored rotated, so the code transposes them before use. Merging upper and lower case means the model cannot tell "A" from "a", and this causes the main errors described later.

## 4. Model

The network has three convolutional blocks followed by two fully connected layers, about 0.6 million parameters in total.

- Each block has two 3×3 convolutions, each followed by batch normalisation and ReLU, then max pooling and dropout.
- The channel widths are 32, 64 and 128, and the image shrinks 28 → 14 → 7 → 3 through the blocks.
- The classifier is a 256-unit fully connected layer with dropout, then 26 output scores (one per letter).

## 5. Training Setup

| Setting | Value |
| --- | --- |
| Framework | PyTorch (ran on an Apple Silicon Mac GPU via MPS) |
| Epochs | 15 |
| Optimiser | AdamW with a OneCycle learning rate schedule (peak 0.003) |
| Loss | Cross entropy with label smoothing 0.1 |
| Batch size | 256 |
| Data augmentation | Random rotation (±12°), shift (10%), scale (0.85 to 1.1), shear (8°) |
| Model selection | The checkpoint with the best validation accuracy was kept, then tested once on the test set |

Augmentation is important here because it makes the model tolerant of the tilted, shifted and resized letters that appear in real photos.

Validation accuracy rose quickly (91.2% after epoch 1) and levelled off at about 95.5% by epoch 15, with no sign of overfitting.

## 6. Handling Real Photos (Preprocessing Pipeline)

This step is what makes the system work on a grader's photo and not only on EMNIST. For every input image the code does the following:

1. **Grayscale and resize.** Convert to grayscale and shrink very large phone photos.
2. **Lighting correction.** Estimate the paper background and divide it out, so shadows and uneven light do not matter.
3. **Thresholding.** Otsu's method separates ink from paper. If the result is inverted (light ink on a dark background), it is flipped automatically.
4. **Letter detection.** Connected components find each letter. Pieces that overlap horizontally are merged, so the dot of an "i" or "j" stays with its stem. Letters are read left to right.
5. **EMNIST styling.** The letter is cropped, padded to a square, thin pen strokes are thickened to match EMNIST stroke width, then resized to 28×28 with a small border and slight blur (matching how EMNIST itself was made).
6. **Prediction.** The 28×28 image goes through the CNN, which returns probabilities. The program prints the top 3 guesses.

The `--single` option tells the program the image holds exactly one letter, and `--debug` saves an image of what the model actually sees.

## 7. Results

### 7.1 EMNIST Letters Test Set (20,800 images)

| Metric | Value |
| --- | --- |
| Test accuracy | **95.50%** |
| Top-3 accuracy | **99.69%** |
| Macro F1 score | **0.9551** |
| Best validation accuracy | 95.49% |
| Macro precision / recall | 0.9553 / 0.9550 |

The test and validation scores are almost identical, which shows the model generalises and is not memorising the training data.

**Best and worst letters (F1 score):**

| Strongest | F1 | Weakest | F1 |
| --- | --- | --- | --- |
| Z | 0.995 | L | 0.752 |
| M | 0.993 | I | 0.763 |
| P | 0.992 | G | 0.886 |
| W | 0.991 | Q | 0.896 |

### 7.2 Our Own Handwriting Photos

We wrote letters on paper, photographed them, and ran every photo through the same pipeline.

| Metric | Value |
| --- | --- |
| Images | 30 |
| Correct | 29 |
| Accuracy | **96.7%** |
| Macro precision / recall / F1 | 0.95 / 0.96 / 0.95 |

The only error was a plain vertical stroke written as lowercase "l", which the model read as "I".

### 7.3 Summary

| Dataset | Size | Accuracy |
| --- | --- | --- |
| EMNIST Letters test set | 20,800 | 95.50% |
| Own handwriting photos | 30 | 96.7% |

## 8. Error Analysis

The weakest letters are I and L (F1 about 0.75), then G and Q.

- **I and L.** EMNIST merges cases, so a capital "I", a lowercase "l" and a simple vertical line are visually the same shape. This is a limit of the labels, not a bug. A human could not reliably separate them without context either. The same confusion showed up in our own photo test.
- **G and Q.** Both are round letters with a tail, so they are sometimes mixed up.
- **Top-3 accuracy of 99.69%** shows that when the model is wrong, the correct letter is almost always its second or third choice.

## 9. Limitations

- It outputs only A to Z. Digits and symbols will be forced into a letter, because there is no "not a letter" class.
- It cannot tell upper case from lower case.
- Joined cursive handwriting is not supported. Letters must be separate.
- Our own photo test is small (30 images) and comes from one writer on plain paper, so it is an encouraging check and not a rigorous benchmark.
- Photos work best with dark ink, even lighting and the letter filling a good part of the frame.

## 10. Possible Improvements

- Train on a dataset that keeps case (for example EMNIST ByClass) to separate A from a, and I from l.
- Add a "not a letter" class or a confidence threshold to reject digits and scribbles.
- Collect handwriting from many people to build a larger, more varied photo test set.
- Use the IAM database to read whole words and lines.

## 11. How to Run the Project

```
cd handwritten_letter
source venv/bin/activate
python train.py --epochs 15                      # trains and saves model.pt, metrics.json, confusion_matrix.png
python predict.py letter.jpg --single --debug    # predict one letter from a photo
python predict.py word.jpg                       # predict several separate letters, left to right
python evaluate_custom.py my_samples/            # accuracy on photos named like A_1.jpg, b_2.jpg
```

| File | Purpose |
| --- | --- |
| `model.py` | CNN architecture |
| `train.py` | Training and test-set evaluation |
| `preprocess.py` | Photo to 28×28 EMNIST-style conversion |
| `predict.py` | Predict letters from an image |
| `evaluate_custom.py` | Metrics on our own labelled photos |

## 12. Presentation Talking Points

1. **The problem:** read a handwritten letter from a photo, including the grader's own handwriting.
2. **The data:** EMNIST Letters, 124,800 training images and 20,800 test images, 26 classes.
3. **The model:** a small CNN with augmentation. It reaches 95.5% test accuracy and 99.7% top-3 accuracy.
4. **The key idea:** benchmark data is clean, photos are not, so a preprocessing pipeline makes real photos look like EMNIST.
5. **The proof:** 96.7% on our own photos, so the pipeline transfers to real handwriting.
6. **The honest limits:** I vs L is an ambiguity in the dataset itself, and the model only knows A to Z.
