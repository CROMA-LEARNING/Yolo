"""
Validacao cruzada k-fold para o detector YOLOv8n de boi_garantido/boi_caprichoso.

Com só 26 imagens no total, uma unica divisao treino/validacao (a que usamos
em scripts/train.py) da uma estimativa de desempenho muito ruidosa: trocar
uma unica imagem do conjunto de validacao pode mudar o mAP em 20 pontos.
K-fold treina o modelo k vezes, cada vez com uma fatia diferente dos dados
como validacao, e reporta a media e o desvio padrao das metricas, uma
estimativa bem mais honesta do desempenho real do modelo.

Usa a mesma receita que resolveu a instabilidade do treino original:
backbone congelado (freeze=10), lr baixo, sem mosaico.
"""

import glob
import os
import shutil

os.environ.setdefault("USE_TORCH_XLA", "0")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

N_SPLITS = 5
SEED = 42
REPO_ROOT = "/media/jose/19E994FA289AF06B/Yolo"
DATASET_ROOT = "dataset"
KFOLD_ROOT = "kfold_runs"
RUNS_PROJECT = os.path.join(REPO_ROOT, "runs", "kfold")
CLASSES = ["boi_garantido", "boi_caprichoso"]


def collect_all_files():
    files = []
    labels = []
    for split in ["train", "val"]:
        for path in glob.glob(f"{DATASET_ROOT}/images/{split}/*.jpg"):
            files.append(os.path.abspath(path))
            labels.append(0 if "boi_garantido" in os.path.basename(path) else 1)
    return files, labels


def label_path_for(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for split in ["train", "val"]:
        candidate = os.path.join(DATASET_ROOT, "labels", split, stem + ".txt")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(stem)


def build_fold_dirs(fold_idx, train_files, val_files):
    fold_dir = os.path.join(KFOLD_ROOT, f"fold{fold_idx}")
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    for split, files in [("train", train_files), ("val", val_files)]:
        img_dir = os.path.join(fold_dir, "images", split)
        lbl_dir = os.path.join(fold_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for f in files:
            os.symlink(f, os.path.join(img_dir, os.path.basename(f)))
            lbl = label_path_for(f)
            os.symlink(lbl, os.path.join(lbl_dir, os.path.basename(lbl)))

    yaml_path = os.path.join(fold_dir, "data.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {os.path.abspath(fold_dir)}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("names:\n  0: boi_garantido\n  1: boi_caprichoso\n")
    return fold_dir, yaml_path


def main():
    files, labels = collect_all_files()
    print(f"Total de imagens: {len(files)} (garantido={labels.count(0)}, caprichoso={labels.count(1)})")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    results_summary = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        print(f"\n=== Fold {fold_idx+1}/{N_SPLITS} === treino={len(train_files)} val={len(val_files)}")

        fold_dir, yaml_path = build_fold_dirs(fold_idx, train_files, val_files)

        model = YOLO("yolov8n.pt")
        model.train(
            data=yaml_path,
            epochs=150,
            patience=30,
            imgsz=640,
            batch=4,
            lr0=0.001,
            mosaic=0.0,
            freeze=10,
            project=RUNS_PROJECT,
            name=f"fold{fold_idx}",
            exist_ok=True,
            seed=SEED,
            verbose=False,
            plots=False,
        )

        metrics = model.val(data=yaml_path, project=RUNS_PROJECT, name=f"fold{fold_idx}_val", exist_ok=True, plots=False)
        results_summary.append({
            "fold": fold_idx + 1,
            "n_val": len(val_files),
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
        })
        print(f"Fold {fold_idx+1}: mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f} "
              f"precision={metrics.box.mp:.3f} recall={metrics.box.mr:.3f}")

    print("\n" + "=" * 60)
    print("RESUMO K-FOLD")
    print("=" * 60)
    for r in results_summary:
        print(r)

    for key in ["mAP50", "mAP50-95", "precision", "recall"]:
        vals = [r[key] for r in results_summary]
        print(f"{key}: media={np.mean(vals):.4f} desvio={np.std(vals):.4f} "
              f"min={min(vals):.4f} max={max(vals):.4f}")

    import json
    os.makedirs("results", exist_ok=True)
    json.dump(results_summary, open("results/kfold_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
