import json
import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(r"D:\Git\23年E题\yolo")
SOURCE_IMAGES_DIR = PROJECT_DIR / "data"
SOURCE_LABELS_DIR = PROJECT_DIR / "labels"
DATASET_DIR = PROJECT_DIR / "dataset_yolo11n"
DATA_YAML_PATH = PROJECT_DIR / "data.yaml"

MODEL_NAME = "yolo11n.pt"
CLASS_NAME = "ball"
IMAGE_SIZE = 640
EPOCHS = 150
BATCH_SIZE = 32
VALIDATION_RATIO = 0.20
DEVICE = 0
WORKERS = 4
SEED = 42

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_sort_key(path: Path) -> tuple[int, str]:
    """让 2.jpg 排在 10.jpg 前面。"""
    if path.stem.isdigit():
        return int(path.stem), path.name.lower()
    return 10**18, path.name.lower()


def validate_yolo_label(label_path: Path) -> tuple[int, bool]:
    """校验单类别 YOLO HBB 标签，返回目标数量和是否为空标签。"""
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, True

    object_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path} 第 {line_number} 行不是 YOLO HBB 的5列格式：{line}"
            )

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        if class_id != 0:
            raise ValueError(
                f"{label_path} 第 {line_number} 行类别编号为 {class_id}，"
                "当前数据集只允许 ball=0。"
            )
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            raise ValueError(
                f"{label_path} 第 {line_number} 行坐标不在 0～1 范围内：{line}"
            )
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"{label_path} 第 {line_number} 行框的宽高必须大于0：{line}"
            )

        object_count += 1

    return object_count, False


def collect_and_validate_samples() -> list[tuple[Path, Path]]:
    images = sorted(
        (
            path
            for path in SOURCE_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=natural_sort_key,
    )

    if len(images) < 2:
        raise RuntimeError(f"{SOURCE_IMAGES_DIR} 中至少需要2张图片。")

    samples: list[tuple[Path, Path]] = []
    total_objects = 0
    empty_label_count = 0

    for image_path in images:
        label_path = SOURCE_LABELS_DIR / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"图片缺少对应标签：{image_path.name}")

        object_count, is_empty = validate_yolo_label(label_path)
        total_objects += object_count
        empty_label_count += int(is_empty)
        samples.append((image_path, label_path))

    image_stems = {image_path.stem for image_path, _ in samples}
    orphan_labels = sorted(
        path.name
        for path in SOURCE_LABELS_DIR.glob("*.txt")
        if path.stem not in image_stems and path.stem.lower() != "classes"
    )
    if orphan_labels:
        raise RuntimeError(f"发现没有对应图片的标签：{orphan_labels[:10]}")

    print(
        f"数据校验通过：{len(samples)}张图片，"
        f"{total_objects}个目标，{empty_label_count}张负样本。"
    )
    return samples


def create_split(
    samples: list[tuple[Path, Path]],
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    positive_samples = [
        sample
        for sample in samples
        if sample[1].read_text(encoding="utf-8").strip()
    ]
    negative_samples = [
        sample
        for sample in samples
        if not sample[1].read_text(encoding="utf-8").strip()
    ]

    def split_group(
        group: list[tuple[Path, Path]],
    ) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
        if not group:
            return [], []
        validation_count = max(1, round(len(group) * VALIDATION_RATIO))
        return group[:-validation_count], group[-validation_count:]

    positive_train, positive_validation = split_group(positive_samples)
    negative_train, negative_validation = split_group(negative_samples)

    # 正、负样本分别按编号顺序取最后20%作为验证集。
    # 这样既保持验证集类别比例，又尽量减少视频相邻帧的随机泄漏。
    train_samples = positive_train + negative_train
    validation_samples = positive_validation + negative_validation

    if not train_samples:
        raise RuntimeError("训练集为空，请增加图片数量或降低验证集比例。")

    print(
        "划分计划："
        f"训练集 {len(positive_train)} 正样本 + {len(negative_train)} 负样本，"
        f"验证集 {len(positive_validation)} 正样本 + "
        f"{len(negative_validation)} 负样本。"
    )
    return train_samples, validation_samples


def copy_split(
    split_name: str,
    samples: list[tuple[Path, Path]],
) -> None:
    image_output_dir = DATASET_DIR / "images" / split_name
    label_output_dir = DATASET_DIR / "labels" / split_name
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in samples:
        shutil.copy2(image_path, image_output_dir / image_path.name)
        shutil.copy2(label_path, label_output_dir / label_path.name)


def prepare_dataset(samples: list[tuple[Path, Path]]) -> Path:
    manifest_path = DATASET_DIR / "split_manifest.json"
    current_source_files = [image_path.name for image_path, _ in samples]

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_files") != current_source_files:
            raise RuntimeError(
                f"{DATASET_DIR} 已存在，但源图片列表发生了变化。"
                "请先备份并删除该数据集目录，再重新运行。"
            )
        print(f"复用已准备的数据集：{DATASET_DIR}")
    else:
        existing_files = [
            path for path in DATASET_DIR.rglob("*") if path.is_file()
        ] if DATASET_DIR.exists() else []
        if existing_files:
            raise RuntimeError(
                f"{DATASET_DIR} 已有文件但缺少 split_manifest.json，"
                "为避免覆盖，请先检查或移动该目录。"
            )

        train_samples, validation_samples = create_split(samples)
        train_positive_count = sum(
            bool(label_path.read_text(encoding="utf-8").strip())
            for _, label_path in train_samples
        )
        validation_positive_count = sum(
            bool(label_path.read_text(encoding="utf-8").strip())
            for _, label_path in validation_samples
        )
        copy_split("train", train_samples)
        copy_split("val", validation_samples)

        manifest = {
            "seed": SEED,
            "split_method": "stratified_ordered_last_20_percent_for_validation",
            "train_count": len(train_samples),
            "validation_count": len(validation_samples),
            "train_positive_count": train_positive_count,
            "train_negative_count": len(train_samples) - train_positive_count,
            "validation_positive_count": validation_positive_count,
            "validation_negative_count": (
                len(validation_samples) - validation_positive_count
            ),
            "source_files": current_source_files,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"数据集划分完成：训练集 {len(train_samples)} 张，"
            f"验证集 {len(validation_samples)} 张。"
        )

    if not DATA_YAML_PATH.is_file():
        raise FileNotFoundError(f"缺少数据集配置文件：{DATA_YAML_PATH}")
    return DATA_YAML_PATH


def train() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前环境无法使用CUDA。请使用 "
            r"D:\python__3.12\.venv\Scripts\python.exe 运行本脚本。"
        )

    print(f"训练设备：{torch.cuda.get_device_name(DEVICE)}")
    samples = collect_and_validate_samples()
    data_yaml_path = prepare_dataset(samples)

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(data_yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_DIR / "runs"),
        name="steel_ball_yolo11n",
        pretrained=True,
        seed=SEED,
        deterministic=True,
        amp=True,
        plots=True,
        mosaic=0.5,
        scale=0.3,
        close_mosaic=20,
    )

    print(
        "训练结束。最佳权重通常位于："
        + str(
            PROJECT_DIR
            / "runs"
            / "steel_ball_yolo11n"
            / "weights"
            / "best.pt"
        )
    )


if __name__ == "__main__":
    random.seed(SEED)
    train()
