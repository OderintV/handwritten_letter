# Handwritten Letter Recognition (A-Z)

CNN trained on **EMNIST Letters** (124,800 train / 20,800 test). Given a photo or scan of handwriting,
it predicts the letter(s). EMNIST merges upper/lower case, so output is one of A-Z.

## Run
```bash
pip install -r requirements.txt
python train.py --epochs 15          # downloads EMNIST, trains, writes model.pt, metrics.json, confusion_matrix.png
python predict.py sehraj.jpg         # predict (reads several letters left-to-right)
python predict.py sehraj.jpg --single --debug   # one letter; saves debug images of what the model sees
python evaluate_custom.py my_samples/           # optional: metrics on your own labelled photos (A_1.jpg, b_2.png ...)
```
Training is fast on a GPU (Google Colab free T4: a few minutes); CPU works but is slower.

## How it works
1. **Model** (`model.py`): 3 conv blocks (conv-BN-ReLU x2, pool, dropout) + 2 FC layers, ~0.5M params.
2. **Training** (`train.py`): 90/10 train/val split, random affine augmentation (rotation, shift, scale, shear),
   AdamW + OneCycle LR, label smoothing. Best-val checkpoint is evaluated once on the untouched test set.
3. **Real-world preprocessing** (`preprocess.py`) - this is what makes it work on a grader's photo:
   illumination correction -> Otsu threshold -> auto-invert (dark ink on paper -> white on black) ->
   connected-component segmentation (dots of i/j merged) -> crop, square-pad, thicken thin pen strokes
   to EMNIST thickness -> 28x28 with 2px border + slight blur (matches how EMNIST was built).

## Tips for the demo photo
Dark pen on plain paper, good even light, letter fills a good part of the frame, no ruled lines through it.
Use `--single` when the image holds exactly one letter.

## Results (fill in from `metrics.json` after training)
| Metric | Value |
|---|---|
| EMNIST Letters test accuracy | |
| Top-3 accuracy | |
| Macro F1 | |
Known confusions (see `confusion_matrix.png`): I/L, O/Q, G/Q, U/V - partly because EMNIST merges cases (e.g. "i" vs "l" vs "I").
