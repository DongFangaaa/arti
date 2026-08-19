# 海康 MVS SDK 模块名兼容性说明

`vision/camera.py` 默认导入：
```python
from MvImport.CameraParams_header import (
    MV_GIGE_DEVICE, MV_USB_DEVICE, MV_ACCESS_Exclusive,
)
from MvImport.MvCameraControl_class import MvCamera
```

但海康 MVS SDK 的 Python 绑定在不同版本下**模块名不同**。

## 版本对照表

| MVS SDK 版本 | Python 模块名 | wheel 文件名示例 |
|-------------|--------------|------------------|
| 3.0.x | `MvImport` | `MVS-3.0.0_*`-Python.whl` |
| 3.1.x | `MvImport` | 同上 |
| 4.x.x | `MVS` | `MVS-4.x.x.x-Python.whl` |

## 切换方法

打开 `vision/camera.py`，把：

```python
try:
    from MvImport.CameraParams_header import (
        MV_GIGE_DEVICE, MV_USB_DEVICE, MV_ACCESS_Exclusive,
    )
    from MvImport.MvCameraControl_class import MvCamera
    MVS_AVAILABLE = True
except Exception:
    MVS_AVAILABLE = False
```

改成：

```python
try:
    from MVS.CameraParams_header import (
        MV_GIGE_DEVICE, MV_USB_DEVICE, MV_ACCESS_Exclusive,
    )
    from MVS.MvCameraControl_class import MvCamera
    MVS_AVAILABLE = True
except Exception:
    MVS_AVAILABLE = False
```

## 安装步骤

1. 从海康机器人官网下载 MVS Python wheel：
   https://www.hikrobotics.com/cn/machinevision/service/download

2. 安装（注意 Python 版本和系统架构匹配）：
   ```bash
   pip install /path/to/MVS-x.x.x.x-Python.whl
   ```

3. 验证：
   ```bash
   python -c "from MvImport.MvCameraControl_class import MvCamera; print('OK')"
   # 或
   python -c "from MVS.MvCameraControl_class import MvCamera; print('OK')"
   ```

4. 如果模块名不同，按上面的"切换方法"改 `vision/camera.py`。

## Docker 容器内注意事项

EPC 1502 (arm64) 上的 Python wheel 与开发机 (x86_64) 不通用，需要：
1. 在 EPC 上安装 arm64 版本
2. 或在容器内通过 `pip install` 安装 wheel 文件
3. 参考 [deploy/Dockerfile](../deploy/Dockerfile) 增加：
   ```dockerfile
   COPY mvs_whl/MVS-x.x.x.x-cp311-cp311-linux_aarch64.whl /tmp/
   RUN pip install /tmp/MVS-x.x.x.x-cp311-cp311-linux_aarch64.whl
   ```

## 调试命令

```bash
# 列出可见 GigE/USB 相机
python -m vision.main --list-ports

# 仅测试相机采集（dummy 模式）
python -m vision.main -c vision/config.yaml --dummy --once
```

如相机打不开：
1. 检查 `vision/config.yaml` 的 `camera.ip` 是否正确
2. 确认 MVS 客户端软件能正常打开
3. 检查 GigE 网卡是否启用巨帧（jumbo frame, MTU ≥ 9000）