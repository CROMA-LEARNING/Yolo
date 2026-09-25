"""
Fine-tuning (transfer learning) do YOLOv8n, pre-treinado no COCO, para
detectar duas classes novas: boi_garantido e boi_caprichoso.
"""

import os

os.environ.setdefault("USE_TORCH_XLA", "0")

from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")  # pesos pre-treinados no COCO (80 classes)

    model.train(
        data="dataset/data.yaml",
        epochs=150,
        patience=30,
        imgsz=640,
        batch=4,
        lr0=0.001,
        mosaic=0.0,
        freeze=10,  # congela o backbone pre-treinado, so o head aprende (dataset muito pequeno)
        project="runs",
        name="boi_detector",
        exist_ok=True,
        seed=42,
        verbose=True,
    )

    metrics = model.val(data="dataset/data.yaml", project="runs", name="boi_detector_val", exist_ok=True)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
