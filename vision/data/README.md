# 数据集说明（缺陷分类版 v2.3 — YOLO-cls 架构）

## 1. 任务与类别

| 类别 | 名称 | defect_id | 库位列 |
|------|------|-----------|-------|
| 0 | scratch | 1 | 第 1 列 |
| 1 | stain | 2 | 第 2 列 |
| 2 | chip | 3 | 第 3 列 |
| 3 | hole | 4 | 第 4 列 |

> 类别顺序即 class id，**必须**与以下文件保持一致：
> - `train/dataset.yaml` 的 `names`
> - `vision/config.yaml` 的 `classes.names`
> - `vision/config.yaml` 的 `classes.plc_defect_id`

---

## 2. 目录结构（YOLO-cls 标准）

```
data/
├── train/
│   ├── scratch/*.png      # 140 张（70%）
│   ├── stain/*.png        # 140 张
│   ├── chip/*.png         # 140 张
│   └── hole/*.png         # 140 张
├── val/
│   ├── scratch/*.png      # 30 张（15%）
│   ├── stain/*.png
│   ├── chip/*.png
│   └── hole/*.png
└── test/
    ├── scratch/*.png      # 30 张（15%）
    ├── stain/*.png
    ├── chip/*.png
    └── hole/*.png
```

---

## 3. 当前数据规模

| 类别 | 总数 | 来源 |
|------|------|------|
| scratch（划痕） | 200 | 合成 + 实拍 |
| stain（污渍） | 200 | 合成 |
| chip（缺角） | 200 | 合成（原文件名 notch） |
| hole（孔洞） | 200 | 合成 |
| **合计** | **800** | — |

> 数据量偏少（比赛现场建议扩充至 500 张/类以上）。

---

## 4. 数据来源与重建

原始素材存放位置：`subject/data_cls/train/{hole,notch,scratch,stain}/*.png`（外部下载）。

重新划分：

```bash
cd subject
python data/prepare_cls_dataset.py
```

脚本会：
1. 把 `data_cls/train/notch/` 映射到 `chip/`
2. 按 70/15/15 划分到 `data/{train,val,test}/{cls}/`
3. 输出每类计数

---

## 5. 标注说明

YOLO-cls 任务**无需 bbox 标注**，类别即子目录名。

如需扩充数据集：
1. 拍照采集新缺陷样本（统一 720×540 BGR）
2. 命名规范：`picture_{cls}_{idx:03d}.png`
3. 放入对应类别的 train 或 val 目录
4. **不要**用 LabelImg 标注（cls 任务不需要）

---

## 6. 标注质量检查

```python
from pathlib import Path
root = Path("data/train")
for cls in ["scratch", "stain", "chip", "hole"]:
    n = len(list((root / cls).glob("*.png")))
    print(f"{cls}: {n}")
```

预期输出：
```
scratch: 140
stain: 140
chip: 140
hole: 140
```

如不平衡，建议在训练脚本中加 `class_weight` 或过采样。

---

## 7. 工具脚本

`prepare_cls_dataset.py` —— 原始 patches → YOLO-cls 划分

执行后：
- 自动清空旧的 `data/{train,val,test}`
- 重新划分并按类名前缀重命名
- 打印每类计数

---

## 8. 数据集目录约定

- `data/raw/`：原始未处理图像（git 忽略）
- `data/defect_patches/`：按"缺陷类型/images,masks"组织的原始素材（git 忽略）
- `data/{train,val,test}/`：**版本控制入库**（经过 prepare 脚本处理后的最终训练集）
- `data_cls/`：从外部下载的临时目录（git 忽略，prepare 脚本的输入）