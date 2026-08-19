# PLC 程序目录（占位）

> 📂 **此目录用于存放 PLC 程序**——比赛最后集成阶段由 PLC 工程师填入。
> 
> 当前**仅含 `.gitkeep`**，实际 ST 文件由 PLC 工程师在 PLCnext Engineer 工程里编写后导出。

## 应包含的文件（比赛最终版）

```
src/plc/
├── README.md                  # 本文件
├── main_program.st            # 主状态机（ST_INIT / ST_WAIT_VISION / ST_CALC_TARGET / ST_UPDATE / ST_READY）
├── fb_shelf_manager.st        # 货架管理功能块（缺陷类型→库位映射）
└── gds_var_definition.md      # OPC UA 节点定义说明（5 个视觉交互变量）
```

## PLC 工程师需要做的

1. 在 **PLCnext Engineer** 里新建工程
2. 定义 OPC UA 5 个交互变量（与 [vision/comm/opcua_comm.py](../../vision/comm/opcua_comm.py) 一致）：
   - `xVisionTrigger`（BOOL，PLC→Vision）
   - `xDefectClass`（INT，Vision→PLC，1~4）
   - `fConfidence`（LREAL，Vision→PLC）
   - `xResultValid`（BOOL，Vision→PLC）
   - `iResultSeq`（INT，Vision→PLC，序列号）
3. 编写 ST 代码（main_program.st 状态机）
4. 编译下载到 EPC 1502
5. 启用 OPC UA Server（端口 4840）
6. 用 UaExpert 浏览，把 NodeId 填到 [vision/config.yaml](../../vision/config.yaml) 的 `plc.opcua.nodes`

## 关于命名

- PLCnext 与我们的 `.st` 文件**都是 IEC 61131-3 Structured Text 标准**，可互拷
- PLCnext 工程的 `.st` 依赖 PLCnext 元数据，**不能直接放进此目录**——需要 PLC 工程师手动把核心逻辑块复制出来

## 比赛前自检

- [ ] `main_program.st` 编译通过（PLCnext Engineer）
- [ ] OPC UA Server 启动成功（UaExpert 能连）
- [ ] 5 个变量可在 UaExpert 浏览树中看到
- [ ] NodeId 已填到 [vision/config.yaml](../../vision/config.yaml)
- [ ] PLC 程序能在 EPC 1502 上独立运行（视觉容器没起时不应崩溃）

---

*目录版本 v2.3 — 2026-08-09*