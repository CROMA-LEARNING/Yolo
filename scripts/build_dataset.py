"""
Monta o dataset de deteccao para o YOLO a partir das imagens ja curadas do
projeto NeuralNetworks-TransferLearning (boi Garantido e boi Caprichoso),
restrito as imagens onde a figura fisica do boi (cabeca/estatua/costume)
esta de fato visivel (lista revisada manualmente, ver garantido_aprovados.json
e caprichoso_aprovados.json), porque o dataset original de classificacao
tambem inclui fotos so de torcida/simbolos sem o boi em quadro, o que nao
serve para detecao de objetos.

Como essas imagens tem so o rotulo de classe (classificacao), nao bounding
box, uso o rembg (modelo U2Net pre-treinado, open source) para segmentar o
objeto principal de cada foto e derivar a caixa delimitadora automaticamente
a partir da mascara alpha. E uma forma de auto-anotacao fraca, mais rapida
que rotular manualmente no LabelMe, com qualidade suficiente para uma
demonstracao de fine-tuning do YOLO em classes novas.
"""

import json
import os
import random

import numpy as np
from PIL import Image
from rembg import remove, new_session

random.seed(42)

SRC_ROOT = "/media/jose/19E994FA289AF06B/NeuralNetworks-TransferLearning/dataset"
DST_ROOT = "/media/jose/19E994FA289AF06B/Yolo/dataset"
APPROVED_DIR = "/tmp/claude-1000/-media-jose-19E994FA289AF06B-NeuralNetworks-TransferLearning/b599e0c8-b0b2-454e-a6db-5fc9e4bf4786/scratchpad/yolo_sheets"
CLASSES = ["boi_garantido", "boi_caprichoso"]
SRC_DIRS = ["garantido", "caprichoso"]
VAL_FRACTION = 0.2
MIN_AREA_FRAC = 0.05
MAX_AREA_FRAC = 0.95

session = new_session("u2net")


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 10)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1, y1


def process_image(src_path, class_id, dst_img_path, dst_label_path):
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    result = remove(img, session=session)
    mask = np.array(result)[:, :, 3]

    box = bbox_from_mask(mask)
    if box is None:
        return False
    x0, y0, x1, y1 = box
    area_frac = ((x1 - x0) * (y1 - y0)) / (w * h)
    if area_frac < MIN_AREA_FRAC or area_frac > MAX_AREA_FRAC:
        return False

    xc = ((x0 + x1) / 2) / w
    yc = ((y0 + y1) / 2) / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h

    img.save(dst_img_path, quality=92)
    with open(dst_label_path, "w") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    return True


def main():
    stats = {"train": 0, "val": 0, "descartadas": 0}
    for class_id, (cls_name, src_dir) in enumerate(zip(CLASSES, SRC_DIRS)):
        files = json.load(open(os.path.join(APPROVED_DIR, f"{src_dir}_aprovados.json"), encoding="utf-8"))
        files = sorted(files)
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_FRACTION))
        val_files = set(files[:n_val])

        for fname in files:
            split = "val" if fname in val_files else "train"
            src_path = os.path.join(SRC_ROOT, src_dir, fname)
            stem = f"{cls_name}_{os.path.splitext(fname)[0]}".replace(" ", "_")
            dst_img = os.path.join(DST_ROOT, "images", split, stem + ".jpg")
            dst_label = os.path.join(DST_ROOT, "labels", split, stem + ".txt")
            ok = process_image(src_path, class_id, dst_img, dst_label)
            if ok:
                stats[split] += 1
            else:
                stats["descartadas"] += 1

    print("Imagens em train:", stats["train"])
    print("Imagens em val:", stats["val"])
    print("Descartadas (segmentacao ruim):", stats["descartadas"])


if __name__ == "__main__":
    main()
