(* ==========================================================================*)
(* 视觉通讯功能块 — Modbus TCP（赛题要求 TCP 协议）*)
(* ==========================================================================*)
(* 通讯方式：PLC 作为 Modbus TCP Server，端口 502，从站号 unit_id=1*)
(* 视觉程序（容器内）通过 host 网络连接到本 Server。*)
(* *)
(* 寄存器映射：*)
(*   HR100  xVisionTrigger  (BOOL→INT 0/1)  PLC→Vision：触发视觉*)
(*   HR110  xDefectClass    (INT)            Vision→PLC：1=hole, 2=notch, 3=scratch, 4=stain, 0=未识别（+1 偏移）*)
(*   HR111  fConfidence     (INT×1000)       Vision→PLC：置信度 0~1000*)
(*   HR112  xResultValid    (BOOL→INT 0/1)   Vision→PLC：结果有效*)
(*   HR113  iResultSeq      (INT)            Vision→PLC：序列号*)
(* ==========================================================================*)
FUNCTION_BLOCK Vision_ModbusTCP
VAR_INPUT
    bEnable          : BOOL;            (* TRUE 时启动通讯循环*)
    plc_ip           : STRING[16];      (* PLC 本机 IP，如 "192.168.3.99"*)
END_VAR
VAR_OUTPUT
    bConnected       : BOOL;            (* Modbus 连接状态*)
    bTriggerReady    : BOOL;            (* 视觉可触发 *)
    iLastDefect      : INT;             (* 最新 defect_class *)
    rLastConfidence  : REAL;            (* 最新置信度（0.0~1.0）*)
    iError           : INT;             (* 错误码：0=OK, 1=无连接, 2=Modbus 错误 *)
END_VAR
VAR
    (* Modbus TCP Server 句柄 *)
    hServer          : DINT := 0;
    poll_ms          : INT := 50;       (* 50ms 轮询一次 *)

    (* HR 寄存器映射（与视觉程序 vision/config.yaml 一致）*)
    reg_trigger      : INT := 100;
    reg_defect       : INT := 110;
    reg_confidence   : INT := 111;
    reg_valid        : INT := 112;
    reg_seq          : INT := 113;

    (* 内部状态 *)
    raw_defect       : INT;
    raw_conf         : INT;
    raw_valid        : INT;
    raw_seq          : INT;
    bFirstCycle      : BOOL := TRUE;
END_VAR

(* ==========================================================================*)
(* 主体：每 50ms 读取一次视觉上报的寄存器，更新输出*)
(* ==========================================================================*)
IF bEnable THEN

    IF bFirstCycle THEN
        (* 初始化 *)
        bFirstCycle := FALSE;
        iError := 0;
        (* 这里 PLCnext Engineer 需要添加 Arp.Plc.Eclr 的 Modbus Server FB*)
        (* 或使用 PLCnext 自带的 Modbus library*)
        (* 此处仅示意，真实代码由 PLC 工程师按 PLCnext API 补全 *)
    END_IF;

    (* 简化的寄存器读取（实际 PLCnext API 略复杂）*)
    IF TRUE THEN
        (* 读取 Vision→PLC 的 4 个寄存器 *)
        raw_defect := MB_ReadHoldingRegister(reg_defect);
        raw_conf   := MB_ReadHoldingRegister(reg_confidence);
        raw_valid  := MB_ReadHoldingRegister(reg_valid);
        raw_seq    := MB_ReadHoldingRegister(reg_seq);

        (* 输出 *)
        bConnected := (raw_defect >= -1);
        iLastDefect := raw_defect;
        rLastConfidence := INT_TO_REAL(raw_conf) / 1000.0;
        bTriggerReady := (raw_valid = 0);  (* valid=0 表示视觉空闲 *)
    END_IF;

END_IF;

(* ==========================================================================*)
(* 触发视觉：将 xVisionTrigger 置 1，上升沿触发拍照）
(* ==========================================================================*)
(* PLC 端在主程序 (Main.prg) 中通过写 HR100 = 1 实现 *)
(* 此 FB 不实现，由 Main.prg 直接操作寄存器 *)
END_FUNCTION_BLOCK