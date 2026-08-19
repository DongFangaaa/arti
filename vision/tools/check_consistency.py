"""检查 subject 项目内以下配置文件的一致性：
  - train/dataset.yaml
  - vision/config.yaml
  - data/{train,val,test}/ 子目录

校验项：
  1. dataset.yaml 与 config.yaml 的 classes.names 一致
  2. plc_defect_id 完整覆盖所有类别（0..3，0-based）
  3. data/{train,val,test}/ 子目录名等于类别名
  4. 各 split 的类别计数（提示数据不平衡）

退出码：0=通过；1=存在错误
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_CLASSES = ["scratch", "stain", "chip", "hole"]
# 0-based，与数据集标注对齐（scratch=0, stain=1, chip=2, hole=3）
# 失败/未识别使用 fail_safe_defect（默认 -1）
EXPECTED_DEFECT_IDS = {"scratch": 0, "stain": 1, "chip": 2, "hole": 3}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    cfg_path = ROOT / "vision" / "config.yaml"
    ds_path = ROOT / "train" / "dataset.yaml"
    data_dir = ROOT / "data"

    # 1) 文件存在
    if not cfg_path.exists():
        errors.append(f"missing {cfg_path}")
    if not ds_path.exists():
        errors.append(f"missing {ds_path}")
    if not data_dir.exists():
        warnings.append(f"missing {data_dir} (run prepare_cls_dataset.py first)")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return 1

    cfg = load_yaml(cfg_path)
    ds = load_yaml(ds_path)

    # 2) classes.names 一致
    cfg_names = cfg["classes"]["names"]
    ds_names = [ds["names"][i] for i in sorted(ds["names"].keys())] \
        if isinstance(ds["names"], dict) else ds["names"]

    if cfg_names != EXPECTED_CLASSES:
        errors.append(
            f"vision/config.yaml classes.names={cfg_names}, "
            f"expected {EXPECTED_CLASSES}")
    if ds_names != EXPECTED_CLASSES:
        errors.append(
            f"train/dataset.yaml names={ds_names}, "
            f"expected {EXPECTED_CLASSES}")
    if cfg_names != ds_names:
        errors.append("cfg.classes.names != dataset.yaml names (不一致)")

    # 3) plc_defect_id 完整
    plc_map = cfg["classes"]["plc_defect_id"]
    for cls, did in EXPECTED_DEFECT_IDS.items():
        if cls not in plc_map:
            errors.append(f"plc_defect_id missing class: {cls}")
        elif plc_map[cls] != did:
            errors.append(
                f"plc_defect_id[{cls}]={plc_map[cls]}, expected {did}")

    # 4) data 目录子目录名
    if data_dir.exists():
        for split in ("train", "val", "test"):
            sp = data_dir / split
            if not sp.exists():
                warnings.append(f"missing data/{split}/")
                continue
            subdirs = sorted(d.name for d in sp.iterdir() if d.is_dir())
            if set(subdirs) != set(EXPECTED_CLASSES):
                errors.append(
                    f"data/{split}/ subdirs={subdirs}, "
                    f"expected {EXPECTED_CLASSES}")
            else:
                counts = {d: len(list((sp / d).glob("*.png"))) +
                              len(list((sp / d).glob("*.bmp")))
                          for d in subdirs}
                total = sum(counts.values())
                print(f"  data/{split}: total={total}, counts={counts}")
                # 类别不平衡警告
                if max(counts.values()) > 0 and \
                   min(counts.values()) / max(counts.values()) < 0.5:
                    warnings.append(
                        f"data/{split} 类别不平衡: "
                        f"min={min(counts.values())}, "
                        f"max={max(counts.values())}")

    # 输出
    print("=" * 60)
    if warnings:
        for w in warnings:
            print(f"[WARN]  {w}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        return 1
    print("[OK] all consistency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())