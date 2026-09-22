from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


SOURCE_CLASS_NAMES = {
    0: "stain",
    1: "scratch",
    2: "hole",
    3: "notch",
}
CANONICAL_CLASS_ORDER = ("hole", "notch", "scratch", "stain")
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
SOURCE_DATASETS = (
    ("v2", "yolo_dataset_v2"),
    ("v3", "yolo_dataset_v3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge data-new v2/v3 and convert YOLO boxes to XAnyLabeling JSON."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data-new",
    )
    parser.add_argument("--output-name", default="xanylabeling_merged")
    parser.add_argument(
        "--delete-source-masks",
        action="store_true",
        help="Delete only yolo_dataset_v2/masks and yolo_dataset_v3/masks after validation.",
    )
    return parser.parse_args()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo_label(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    text = label_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return shapes

    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row: {label_path}:{line_number}: {line}")
        class_id = int(fields[0])
        if class_id not in SOURCE_CLASS_NAMES:
            raise ValueError(f"Unknown class {class_id}: {label_path}:{line_number}")
        x_center, y_center, box_width, box_height = map(float, fields[1:])
        fields_to_check = (
            x_center,
            y_center,
            box_width,
            box_height,
        )
        if not all(0.0 <= value <= 1.0 for value in fields_to_check):
            raise ValueError(
                f"Normalized coordinate outside [0, 1]: {label_path}:{line_number}: "
                f"{fields_to_check}"
            )
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"Non-positive box size: {label_path}:{line_number}")

        left = max(0.0, (x_center - box_width / 2.0) * width)
        top = max(0.0, (y_center - box_height / 2.0) * height)
        right = min(float(width), (x_center + box_width / 2.0) * width)
        bottom = min(float(height), (y_center + box_height / 2.0) * height)
        if right <= left or bottom <= top:
            raise ValueError(f"Collapsed box: {label_path}:{line_number}")

        shapes.append(
            {
                "label": SOURCE_CLASS_NAMES[class_id],
                "score": None,
                "points": [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )
    return shapes


def xanylabeling_document(
    image_name: str,
    width: int,
    height: int,
    shapes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": "4.0.0-beta.13",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_dataset(root: Path, temp_output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    class_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    image_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    content_hashes: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for split in SPLITS:
        (temp_output / split).mkdir(parents=True, exist_ok=False)

    for prefix, folder_name in SOURCE_DATASETS:
        dataset_root = (root / folder_name).resolve()
        if dataset_root.parent != root or not dataset_root.is_dir():
            raise FileNotFoundError(f"Missing source dataset: {dataset_root}")

        for split in SPLITS:
            images_dir = dataset_root / "images" / split
            labels_dir = dataset_root / "labels" / split
            source_images = sorted(
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not source_images:
                raise ValueError(f"No images found: {images_dir}")

            for source_image in source_images:
                source_label = labels_dir / f"{source_image.stem}.txt"
                if not source_label.is_file():
                    raise FileNotFoundError(f"Missing label: {source_label}")
                width, height = image_size(source_image)
                shapes = parse_yolo_label(source_label, width, height)
                output_name = f"{prefix}_{source_image.name}"
                output_image = temp_output / split / output_name
                output_json = output_image.with_suffix(".json")
                if output_image.exists() or output_json.exists():
                    raise FileExistsError(f"Output collision: {output_image}")

                shutil.copy2(source_image, output_image)
                write_json(
                    output_json,
                    xanylabeling_document(output_name, width, height, shapes),
                )
                for shape in shapes:
                    class_counts[split][shape["label"]] += 1
                image_counts[split] += 1
                if not shapes:
                    empty_counts[split] += 1
                digest = sha256(source_image)
                content_hashes[digest].append(
                    {"split": split, "image": output_name, "source": str(source_image)}
                )
                records.append(
                    {
                        "split": split,
                        "image": f"{split}/{output_name}",
                        "annotation": f"{split}/{output_json.name}",
                        "source": str(source_image),
                        "source_label": str(source_label),
                        "negative": False,
                        "sha256": digest,
                    }
                )

    # These six BMP files are one real acquisition group. Keep all of them in train
    # so near-identical exposures do not leak into validation or test.
    negative_images = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not negative_images:
        raise ValueError(f"No root-level unannotated negative images found in {root}")
    for source_image in negative_images:
        width, height = image_size(source_image)
        output_name = f"negative_{source_image.name}"
        output_image = temp_output / "train" / output_name
        output_json = output_image.with_suffix(".json")
        if output_image.exists() or output_json.exists():
            raise FileExistsError(f"Output collision: {output_image}")
        shutil.copy2(source_image, output_image)
        write_json(output_json, xanylabeling_document(output_name, width, height, []))
        image_counts["train"] += 1
        empty_counts["train"] += 1
        digest = sha256(source_image)
        content_hashes[digest].append(
            {"split": "train", "image": output_name, "source": str(source_image)}
        )
        records.append(
            {
                "split": "train",
                "image": f"train/{output_name}",
                "annotation": f"train/{output_json.name}",
                "source": str(source_image),
                "source_label": None,
                "negative": True,
                "sha256": digest,
            }
        )

    duplicate_groups = [values for values in content_hashes.values() if len(values) > 1]
    cross_split_duplicates = [
        values
        for values in duplicate_groups
        if len({value["split"] for value in values}) > 1
    ]
    summary = {
        "format": "XAnyLabeling JSON rectangles",
        "class_names": list(CANONICAL_CLASS_ORDER),
        "source_class_id_map": SOURCE_CLASS_NAMES,
        "recommended_export_class_order": list(CANONICAL_CLASS_ORDER),
        "image_counts": dict(image_counts),
        "total_images": sum(image_counts.values()),
        "empty_negative_images": dict(empty_counts),
        "class_box_counts": {
            split: dict(class_counts[split]) for split in SPLITS
        },
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_exact_duplicate_groups": len(cross_split_duplicates),
        "negative_policy": (
            "All root-level unannotated BMP images are kept together in train because "
            "they are one correlated acquisition group."
        ),
    }
    return records, summary


def validate_output(output: Path, expected_summary: dict[str, Any]) -> dict[str, Any]:
    actual_images: Counter[str] = Counter()
    actual_empty: Counter[str] = Counter()
    actual_classes: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    errors: list[str] = []

    for split in SPLITS:
        split_dir = output / split
        images = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        json_files = sorted(split_dir.glob("*.json"))
        actual_images[split] = len(images)
        if len(images) != len(json_files):
            errors.append(
                f"{split}: image/json mismatch: {len(images)} images, {len(json_files)} json"
            )

        for image_path in images:
            annotation_path = image_path.with_suffix(".json")
            if not annotation_path.is_file():
                errors.append(f"Missing JSON: {annotation_path}")
                continue
            document = json.loads(annotation_path.read_text(encoding="utf-8"))
            width, height = image_size(image_path)
            if document.get("imagePath") != image_path.name:
                errors.append(f"imagePath mismatch: {annotation_path}")
            if document.get("imageWidth") != width or document.get("imageHeight") != height:
                errors.append(f"image dimension mismatch: {annotation_path}")
            shapes = document.get("shapes")
            if not isinstance(shapes, list):
                errors.append(f"shapes is not a list: {annotation_path}")
                continue
            if not shapes:
                actual_empty[split] += 1
            for shape in shapes:
                label = shape.get("label")
                if label not in SOURCE_CLASS_NAMES.values():
                    errors.append(f"Unknown label {label}: {annotation_path}")
                    continue
                if shape.get("shape_type") != "rectangle":
                    errors.append(f"Non-rectangle shape: {annotation_path}")
                points = shape.get("points")
                if not isinstance(points, list) or len(points) != 4:
                    errors.append(f"Rectangle must have four points: {annotation_path}")
                    continue
                for x, y in points:
                    if not (0.0 <= float(x) <= width and 0.0 <= float(y) <= height):
                        errors.append(f"Point outside image: {annotation_path}: {[x, y]}")
                actual_classes[split][label] += 1

    if dict(actual_images) != expected_summary["image_counts"]:
        errors.append(
            f"Image counts changed: expected={expected_summary['image_counts']} "
            f"actual={dict(actual_images)}"
        )
    if dict(actual_empty) != expected_summary["empty_negative_images"]:
        errors.append(
            f"Empty counts changed: expected={expected_summary['empty_negative_images']} "
            f"actual={dict(actual_empty)}"
        )
    expected_classes = expected_summary["class_box_counts"]
    actual_class_dict = {split: dict(actual_classes[split]) for split in SPLITS}
    if actual_class_dict != expected_classes:
        errors.append(
            f"Class counts changed: expected={expected_classes} actual={actual_class_dict}"
        )
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Merged dataset validation failed ({len(errors)} errors):\n{preview}")
    return {
        "status": "ok",
        "images": dict(actual_images),
        "empty": dict(actual_empty),
        "class_boxes": actual_class_dict,
    }


def write_metadata(
    output: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    (output / "classes.txt").write_text(
        "\n".join(CANONICAL_CLASS_ORDER) + "\n",
        encoding="utf-8",
    )
    with (output / "merge_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    full_summary = dict(summary)
    full_summary["validation"] = validation
    write_json(output / "dataset_summary.json", full_summary)

    total = summary["total_images"]
    image_counts = summary["image_counts"]
    class_counts = summary["class_box_counts"]
    report = f"""# data-new 合并与数据质量审计

## 合并结果

- 输出格式：XAnyLabeling JSON，矩形框使用4个角点，JSON与图片同目录。
- 类别：`hole`、`notch`、`scratch`、`stain`；此顺序与现有训练工程一致。
- 总图片数：{total}。
- 训练集：{image_counts['train']}（{image_counts['train'] / total:.2%}）。
- 验证集：{image_counts['val']}（{image_counts['val'] / total:.2%}）。
- 测试集：{image_counts['test']}（{image_counts['test'] / total:.2%}）。
- 空标注负样本：训练集 {summary['empty_negative_images'].get('train', 0)} 张。
- 精确重复图片组：{summary['exact_duplicate_groups']}。
- 跨划分精确重复图片组：{summary['cross_split_exact_duplicate_groups']}。

## 各划分类别框数量

| 划分 | stain | scratch | hole | notch |
|---|---:|---:|---:|---:|
| train | {class_counts['train'].get('stain', 0)} | {class_counts['train'].get('scratch', 0)} | {class_counts['train'].get('hole', 0)} | {class_counts['train'].get('notch', 0)} |
| val | {class_counts['val'].get('stain', 0)} | {class_counts['val'].get('scratch', 0)} | {class_counts['val'].get('hole', 0)} | {class_counts['val'].get('notch', 0)} |
| test | {class_counts['test'].get('stain', 0)} | {class_counts['test'].get('scratch', 0)} | {class_counts['test'].get('hole', 0)} | {class_counts['test'].get('notch', 0)} |

## 结论

- 四个类别在v2和v3中的源ID与名称一致；转换时按源ID读取并写为类别字符串。以后从XAnyLabeling导出YOLO时，应使用 `hole, notch, scratch, stain` 的现有工程顺序。
- `stain`、`scratch`、`hole`、`notch` 的概念可以保留；但合成图中 stain 多为纯黑实心块、hole 多为黑色轮廓，模型可能学到“填充/轮廓”而非真实缺陷纹理，应补充真实样本。
- v2 notch 偏大，v3 notch 偏小，合并后尺寸覆盖更好，但仍需检查真实设备上的缺口尺寸范围。
- 80/10/10 的数量比例合理，但v2和v3各自由单一底图生成，训练、验证和测试共享背景风格，内容独立性不足，测试指标可能明显偏高。
- 现有正样本全部是单缺陷图，没有多缺陷共存样本；若现场可能同时出现多种缺陷，应加入多缺陷图片。
- 六张无标注BMP来自同一采集批次，已全部作为负样本加入训练集，避免拆分到验证/测试造成近重复泄漏；负样本数量仍明显不足。

## 后续建议

1. 用独立的实拍批次分别建立验证集和测试集，不要从同一连拍或同一底图随机拆分。
2. 补充真实无缺陷图，建议负样本达到训练图片的5%到10%，并按采集批次分组划分。
3. 每类补充真实缺陷，重点检查 stain 与 hole、边缘 notch 与背景轮廓的混淆。
4. 若生产现场会出现多个缺陷，加入多框、多类别共存图片。
"""
    (output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8", newline="\n")


def delete_source_masks(root: Path) -> list[str]:
    deleted: list[str] = []
    for _, folder_name in SOURCE_DATASETS:
        dataset_root = (root / folder_name).resolve()
        mask_dir = (dataset_root / "masks").resolve()
        if mask_dir.parent != dataset_root or mask_dir.name != "masks":
            raise RuntimeError(f"Refusing unsafe mask deletion target: {mask_dir}")
        if mask_dir.is_dir():
            shutil.rmtree(mask_dir)
            deleted.append(str(mask_dir))
    return deleted


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if root.name != "data-new" or not root.is_dir():
        raise ValueError(f"Expected an existing data-new directory, got: {root}")
    output = (root / args.output_name).resolve()
    temp_output = (root / f"{args.output_name}.tmp").resolve()
    if output.parent != root or temp_output.parent != root:
        raise ValueError("Output must remain directly under data-new")
    if output.exists() or temp_output.exists():
        raise FileExistsError(f"Output already exists: {output} or {temp_output}")

    temp_output.mkdir()
    try:
        records, summary = convert_dataset(root, temp_output)
        validation = validate_output(temp_output, summary)
        write_metadata(temp_output, records, summary, validation)
        temp_output.rename(output)
    except Exception:
        if temp_output.is_dir():
            shutil.rmtree(temp_output)
        raise

    deleted_masks: list[str] = []
    if args.delete_source_masks:
        deleted_masks = delete_source_masks(root)
    result = {
        "output": str(output),
        "summary": summary,
        "validation": validation,
        "deleted_masks": deleted_masks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
