"""
Roda o detector treinado em uma imagem e desenha as caixas encontradas.

Uso:
    python scripts/predict.py caminho/da/imagem.jpg
"""

import os
import sys

os.environ.setdefault("USE_TORCH_XLA", "0")

from ultralytics import YOLO

WEIGHTS = "runs/boi_detector/weights/best.pt"


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/predict.py caminho/da/imagem.jpg")
        sys.exit(1)

    model = YOLO(WEIGHTS)
    results = model.predict(sys.argv[1], conf=0.4, iou=0.3, max_det=3, save=True)
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"{cls}: {conf:.2f}")


if __name__ == "__main__":
    main()
