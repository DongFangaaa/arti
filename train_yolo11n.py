from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data.zip/defect_dataset/merged_xanylabeling/split_811/dataset.yaml"
DEFAULT_PROJECT = ROOT / "runs_defects"
FINAL_MODEL_DIR = ROOT / "models"
AUGMENTATION_RE = re.compile(r"_(r90|r180|r270|g0\.6|g1\.6)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11n to detect hole, notch, scratch and stain defects."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch", type=float, default=32,
        help="Training batch size. Default: 32; use -1 for automatic selection.",
    )
    parser.add_argument("--device", default=None, help="Examples: 0, 0,1, cpu. Default: auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="yolo11n_four_defects")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def group_name(stem: str) -> str:
    stem = stem.split("__src_", 1)[0]
    return AUGMENTATION_RE.sub("", stem)


def validate_dataset(data_yaml: Path) -> dict[str, object]:
    import yaml

    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    expected_names = {0: "hole", 1: "notch", 2: "scratch", 3: "stain"}
    raw_names = config.get("names")
    names = (
        {int(key): value for key, value in raw_names.items()}
        if isinstance(raw_names, dict)
        else {index: value for index, value in enumerate(raw_names or [])}
    )
    if int(config.get("nc", -1)) != 4 or names != expected_names:
        raise ValueError(
            f"Expected four classes {expected_names}, got nc={config.get('nc')} names={names}"
        )

    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    split_groups: dict[str, set[str]] = {}
    report: dict[str, object] = {"dataset": str(data_yaml), "splits": {}}
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for split in ("train", "val", "test"):
        image_dir = dataset_root / config[split]
        label_dir = image_dir.parent / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in image_suffixes)
        class_counts: Counter[int] = Counter()
        groups: set[str] = set()
        for image in images:
            groups.add(group_name(image.stem))
            if split in {"val", "test"} and AUGMENTATION_RE.search(image.stem):
                raise ValueError(f"Augmented image is not allowed in {split}: {image.name}")
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO label: {label}:{line_number}")
                cls = int(fields[0])
                box = [float(value) for value in fields[1:]]
                if cls not in range(4) or any(value < 0.0 or value > 1.0 for value in box):
                    raise ValueError(f"Invalid class/box: {label}:{line_number}")
                class_counts[cls] += 1
        if not images:
            raise ValueError(f"No images found in {image_dir}")
        split_groups[split] = groups
        report["splits"][split] = {
            "images": len(images),
            "groups": len(groups),
            "class_counts": dict(sorted(class_counts.items())),
        }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[left] & split_groups[right]
        if overlap:
            raise ValueError(f"Dataset leakage between {left} and {right}: {len(overlap)} groups")
    return report


def main() -> None:
    args = parse_args()
    dataset_report = validate_dataset(args.data)
    print(json.dumps(dataset_report, ensure_ascii=False, indent=2))

    from ultralytics import YOLO

    if args.resume:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        train_result = model.train(resume=True)
    else:
        model = YOLO(args.model)
        options = {
            "data": str(args.data.resolve()),
            "epochs": args.epochs,
            "patience": 60,
            "imgsz": args.imgsz,
            "batch": int(args.batch) if args.batch >= 1 else args.batch,
            "workers": args.workers,
            "project": str(args.project.resolve()),
            "name": args.name,
            "exist_ok": False,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 0.002,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "amp": True,
            "seed": 20260905,
            "deterministic": True,
            "cache": False,
            "plots": True,
            "save": True,
            "save_period": 20,
            "val": True,
            "hsv_h": 0.005,
            "hsv_s": 0.15,
            "hsv_v": 0.20,
            "degrees": 5.0,
            "translate": 0.05,
            "scale": 0.15,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.2,
            "fliplr": 0.2,
            "mosaic": 0.10,
            "close_mosaic": 50,
            "mixup": 0.0,
            "cutmix": 0.0,
        }
        if args.device is not None:
            options["device"] = args.device
        train_result = model.train(**options)

    # Ultralytics returns metrics from train(); the run directory belongs to the trainer.
    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_path}")

    best_model = YOLO(str(best_path))
    test_options = {
        "data": str(args.data.resolve()),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": int(args.batch) if args.batch >= 1 else 32,
        "workers": args.workers,
        "plots": True,
        "project": str(args.project.resolve()),
        "name": f"{args.name}_test",
    }
    if args.device is not None:
        test_options["device"] = args.device
    metrics = best_model.val(**test_options)

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model = FINAL_MODEL_DIR / "yolo11n_four_defects_best.pt"
    shutil.copy2(best_path, final_model)
    summary = {
        "best_checkpoint": str(best_path),
        "copied_model": str(final_model),
        "test_map50": float(metrics.box.map50),
        "test_map50_95": float(metrics.box.map),
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
    }
    (save_dir / "final_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
