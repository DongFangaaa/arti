# 视觉系统 PLC 接口规范文档

## Modbus TCP 通讯协议 — 缺陷分类视觉检测系统

**版本 v2.4 | 2026-08-19**

***

## 目录

1. [概述](#一概述)
2. [寄存器映射表](#二寄存器映射表)
3. [PLC 端配置步骤](#三plc-端配置步骤)
4. [数据通讯时序](#四数据通讯时序)
5. [缺陷类别映射表](#五缺陷类别映射表)
6. [异常处理约定](#六异常处理约定)
7. [附录](#七附录)

***

## 一、概述

本文档定义视觉检测系统与 PLC 之间的 Modbus TCP 通讯接口规范，适用于视觉逻辑算法应用赛缺陷分类项目。双方通过 Modbus TCP 协议交换触发信号与检测结果，实现物料缺陷的自动识别与分拣控制。

### 1.1 通讯协议参数

| 参数项            | 设定值          | 说明                          |
| -------------- | ------------ | --------------------------- |
| 通讯协议           | Modbus TCP   | 赛题要求，主路径                    |
| PLC IP 地址      | 192.168.3.99 | PLC地址                       |
| 端口号            | 502          | Modbus TCP 标准端口             |
| 从站号 (unit\_id) | 1            | PLC 端需配置相同                  |
| 超时时间           | 2.0 秒        | 单次读写超时                      |
| 通讯频率           | 每工件 1 次      | 5 分钟内 12 工件                 |
| 触发方向           | PLC → Vision | PLC 主动                      |
| 上报方向           | Vision → PLC | Vision 被动                   |
| 通讯介质           | TCP          | 容器内 → 容器外 PLC Modbus Server |

### 1.2 网络拓扑

视觉系统以 Docker 容器形式运行于 EPC 工控机，通过 host 网络模式直接访问 PLC 的 Modbus TCP Server。容器与 PLC 处于同一局域网段，无需额外的端口映射或 NAT 配置。

***

## 二、寄存器映射表

以下保持寄存器（Holding Register）地址为默认配置，可在视觉端 `config.yaml` 中调整。若 PLC 端已有固定地址规划，双方协商后由视觉侧修改配置即可。

| 寄存器地址   | 类型               | 方向           | 功能                    | 数据格式                                      |
| ------- | ---------------- | ------------ | --------------------- | ----------------------------------------- |
| **100** | Holding Register | PLC → Vision | 触发信号 `xVisionTrigger` | 0 = 无触发 / 1 = 触发拍照                        |
| **110** | Holding Register | Vision → PLC | 缺陷类别 `xDefectClass`   | 0=hole, 1=chip, 2=scratch, 3=stain |
| **111** | Holding Register | Vision → PLC | 置信度 `fConfidence`     | 0\~1000（实际值 × 1000）                       |
| **112** | Holding Register | Vision → PLC | 结果有效 `xResultValid`   | 0 = 无效 / 1 = 有效                           |
| **113** | Holding Register | Vision → PLC | 序列号 `iResultSeq`      | 递增整数，PLC 可校验丢包                            |

**寄存器地址可在** **`vision/config.yaml`** **中调整**：

```yaml
plc:
  protocol: "modbus_tcp"
  modbus_tcp:
    registers:
      trigger: 100
      defect_class: 110
      confidence: 111
      valid: 112
      seq: 113
```

> 注：缺陷类别直接使用 0\~3 全占位，不再保留哨兵值。PLC 收到的值直接对应上表，无需再做偏移计算。

***

## 三、PLC 端配置步骤

### 3.1 变量定义

在 PLCnext Engineer 的 MAIN 程序中定义以下 5 个 INT 类型变量，并与对应的保持寄存器绑定：

| PLC 变量名          | 数据类型 | 对应寄存器  | 方向           |
| ---------------- | ---- | ------ | ------------ |
| `xVisionTrigger` | INT  | HR 100 | PLC → Vision |
| `xDefectClass`   | INT  | HR 110 | Vision → PLC |
| `fConfidence`    | INT  | HR 111 | Vision → PLC |
| `xResultValid`   | INT  | HR 112 | Vision → PLC |
| `iResultSeq`     | INT  | HR 113 | Vision → PLC |

### 3.2 启用 Modbus TCP Server

- 在 PLCnext 中启用 Modbus TCP Server 功能
- 设置监听端口：`502`
- 设置从站号 (Unit ID)：`1`
- 将上述 5 个变量映射到对应的保持寄存器地址

> OPC UA / Modbus 在 PLCnext 中是可选集成——通过 `Arp.Plc.Eclr` 程序框架，或用 `pms-protocols-modbus` 库。

### 3.3 触发逻辑

PLC 在检测到物料到达工位后，将 HR\[100] 置为 1，视觉系统检测到触发信号后执行拍照、推理，完成后将结果写入 HR\[110]\~HR\[113]。PLC 在确认 `valid=1` 后读取结果并控制三轴执行分类放置，随后将 HR\[100] 清零，等待下一次触发。

***

## 四、数据通讯时序

单次完整的检测周期时序如下：

| 步骤 | 发起方    | 操作        | 寄存器           | 数值示例        |
| -- | ------ | --------- | ------------- | ----------- |
| 1  | PLC    | 物料到达，触发视觉 | HR\[100]      | 1           |
| 2  | Vision | 读取触发信号    | HR\[100]      | 1           |
| 3  | Vision | 抓帧        | —             | —           |
| 4  | Vision | 图像预处理     | —             | —           |
| 5  | Vision | 模型推理      | —             | —           |
| 6  | Vision | 写入缺陷类别    | HR\[110]      | 3 (scratch) |
| 7  | Vision | 写入置信度     | HR\[111]      | 920 (0.92)  |
| 8  | Vision | 写入有效标志    | HR\[112]      | 1           |
| 9  | Vision | 写入序列号     | HR\[113]      | N           |
| 10 | PLC    | 读取结果，执行搬运 | HR\[110\~113] | —           |
| 11 | PLC    | 清零触发      | HR\[100]      | 0           |

时序图：

```
PLC                                          Vision (Python Container)
 │                                                    │
 │ (物料到达, 触发拍照)                                 │
 ├─── HR[100] = 1 ──────────────────────────────▶│ read_trigger() → True
 │                                                    │ 抓帧
 │                                                    │ 预处理
 │                                                    │ 模型推理
 │                                                    │ 转换为 defect_id
 │                                                    │
 │                                                    │ write_result() →
 │ ◀── HR[110] = 3 ──────────────────────────────────│ (defect, conf, valid, seq)
 │ ◀── HR[111] = 920 ────────────────────────────────┤
 │ ◀── HR[112] = 1 ──────────────────────────────────┤
 │ ◀── HR[113] = 42 ─────────────────────────────────┤
 │                                                    │
 │ (PLC 程序读取 4 个寄存器, 控制三轴)                │
 │ ...                                                │
```

正常单次检测周期约为 70ms（视觉端推理 26ms + 通讯开销），远小于赛题 3 分钟的总时限要求。

***

## 五、缺陷类别映射表

视觉系统识别 4 类缺陷，通过 HR\[110] 上报给 PLC。PLC 根据该值控制三轴将物料放置到对应工位。

| HR\[110] 值 | 缺陷名称 | 英文标识    | PLC 建议动作       |
| ---------- | ---- | ------- | -------------- |
| 0          | 孔洞   | hole    | 放置到 hole 工位    |
| 1          | 缺口   | chip    | 放置到 chip 工位    |
| 2          | 划痕   | scratch | 放置到 scratch 工位 |
| 3          | 污渍   | stain   | 放置到 stain 工位   |

注意：HR\[110] 全占位 0\~3，不再保留任何哨兵值。**当无法识别或低置信度时，视觉端仍返回置信度最高的类别**，PLC 仅根据 `valid` 字段判断本次结果是否可信。

***

## 六、异常处理约定

双方在通讯异常或视觉检测失败时的处理约定如下：

| 场景        | 视觉端行为                            | PLC 端应处理     |
| --------- | -------------------------------- | ------------ |
| 正常检测      | `valid=1`, `seq=N`, `class=0~3`  | 读取结果，执行搬运    |
| 相机故障      | `valid=0`, `class=任意`（推理仍返回 top-1） | 报警停机         |
| 推理超时      | `valid=0`, `class=任意`           | 报警停机         |
| 低置信度 / 无法识别 | `valid=0`, 但 `class` 仍为置信度最高的类别   | 报警停机         |
| Modbus 断开 | 持续重连，不上报                         | 等待 N 秒无响应则报警 |

**设计原则**：宁可报错停机，不可放错物料。当 `valid=0` 时，PLC 应立即停止当前周期并触发报警，等待人工干预或复位信号。

***

## 七、附录

### 7.1 视觉端配置片段（config.yaml）

以下配置片段供 PLC 工程师参考，了解视觉端的寄存器映射设定：

```yaml
plc:
  protocol: "modbus_tcp"
  modbus_tcp:
    host: "192.168.3.99"
    port: 502
    unit_id: 1
    timeout_s: 2.0
    registers:
      trigger: 100
      defect_class: 110
      confidence: 111
      valid: 112
      seq: 113
```

### 7.2 容器网络

Podman 容器与 PLC 之间的网络：

```
容器（Podman）─── host 网络 ───▶ 宿主机 EPC
                                └──▶ Modbus TCP Server（PLC 进程内）
```

**推荐启动方式**（直接用 host 网络，避免端口映射问题）：

```bash
podman run --network=host -d --restart=always \
    --name vision \
    -e USE_DUMMY_CAMERA=0 \
    -v /opt/mvs_sdk:/opt/mvs_sdk:ro \
    localhost/vision:2.3
```

### 7.3 调试工具

| 工具                                           | 用途                                 |
| -------------------------------------------- | ---------------------------------- |
| `python -m src.dummy_modbus_server`          | 本地 PC 启动 Modbus TCP Server（端口 502） |
| `python tools/_smoke_modbus.py`              | 端到端冒烟测试（启 server + vision.comm 读写） |
| `python -m test.test_plc_comm --mode modbus` | 通讯连通性测试                            |
| Modbus Poll / Modbus Slave（GUI）              | 第三方 Modbus 调试工具                    |

### 7.4 答辩要点

| 问题                   | 答案                                    |
| -------------------- | ------------------------------------- |
| "为什么用 Modbus TCP？"   | 赛题要求 TCP 协议；Modbus 是工业标准，PLCnext 原生支持 |
| "为什么不用 OPC UA？"      | 赛题明确说 TCP 协议                          |
| "序列号 seq 有什么用？"      | PLC 端可校验是否丢包/重发                       |
| "valid=0 时 PLC 怎么办？" | 报警停机，宁可报错不可放错                         |

### 7.5 联系方式

如对接口规范有疑问或需要调整寄存器地址、协议参数，请联系视觉端负责人协商修改 `config.yaml` 配置。

***

*文档版本 v3.1 — 2026-08-19（合并 PLC 接口规范文档.docx 与通讯协议.md；缺陷类别改为 0~3 全占位，无法识别仍返回置信度最高类别）*
