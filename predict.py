"""Predict handwritten letter(s) from an image.

    python predict.py my_letter.jpg              # one or more letters, read left-to-right
    python predict.py my_letter.jpg --single     # force the whole image to be ONE letter
    python predict.py my_letter.jpg --debug      # also saves what the model actually sees
"""
import argparse
import cv2, numpy as np, torch
from model import Net, MEAN, STD, LETTERS
from preprocess import letters_from_image


def load_model(path="model.pt"):
    m = Net(); m.load_state_dict(torch.load(path, map_location="cpu")); return m.eval()


@torch.no_grad()
def classify(model, imgs28):
    x = torch.from_numpy(np.stack(imgs28)).float().div(255).unsqueeze(1)
    return torch.softmax(model((x - MEAN) / STD), 1)


def predict_image(path, model=None, single=False):
    model = model or load_model()
    imgs, bw, boxes = letters_from_image(path, single)
    if not imgs:
        return [], imgs, bw, boxes
    probs = classify(model, imgs)
    res = []
    for p in probs:
        top = p.topk(3)
        res.append([(LETTERS[i], float(v)) for v, i in zip(top.values, top.indices)])
    return res, imgs, bw, boxes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--model", default="model.pt")
    ap.add_argument("--single", action="store_true"); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    res, imgs, bw, boxes = predict_image(a.image, load_model(a.model), a.single)
    if not res:
        raise SystemExit("No handwriting found - try a clearer / closer photo.")
    print("Prediction:", "".join(r[0][0] for r in res))
    for i, r in enumerate(res, 1):
        print(f"  letter {i}: " + " | ".join(f"{l} {p:.1%}" for l, p in r))
    if a.debug:
        cv2.imwrite("debug_binary.png", bw)
        strip = np.hstack([cv2.resize(i, (112, 112), interpolation=cv2.INTER_NEAREST) for i in imgs])
        cv2.imwrite("debug_model_input.png", strip)
        print("saved debug_binary.png and debug_model_input.png")
