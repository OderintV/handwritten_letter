"""Evaluate on your own labelled photos. Name files with the true letter first: A_1.jpg, b_02.png ...

    python evaluate_custom.py my_samples/
"""
import sys, glob, os
from sklearn.metrics import accuracy_score, classification_report
from predict import predict_image, load_model

folder = sys.argv[1]
model = load_model()
y, p = [], []
for f in sorted(glob.glob(os.path.join(folder, "*"))):
    truth = os.path.basename(f)[0].upper()
    if not truth.isalpha():
        continue
    res, *_ = predict_image(f, model, single=True)
    if res:
        y.append(truth); p.append(res[0][0][0])
        print(f"{os.path.basename(f):20s} true {truth}  pred {p[-1]}")
print(f"\nAccuracy on {len(y)} custom images: {accuracy_score(y, p):.3f}")
print(classification_report(y, p, zero_division=0))
