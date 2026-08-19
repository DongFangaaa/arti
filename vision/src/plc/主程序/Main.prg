(* ==========================================================================*)
(* 视觉逻辑算法应用赛 — PLC 主程序（ST 状态机）*)
(* 目标控制器：PLCnext EPC 1502*)
(* 协议：Modbus TCP（视觉程序在容器内通过 host 网络通讯）*)
(* ==========================================================================*)
(* 状态机：*)
(*   0  ST_INIT              初始化*)
(*   10 ST_WAIT_VISION       等待视觉返回*)
(*   20 ST_CALC_TARGET       根据 defect_class 计算目标库位*)
(*   30 ST_MOVE_AX           三轴联动*)
(*   40 ST_RELEASE           放料*)
(*   50 ST_RETURN_HOME       三轴回零*)
(*   60 ST_READY             准备下一个物料*)
(*   99 ST_ALARM             报警停机*)
(* ==========================================================================*)
PROGRAM Main
VAR
    (* 状态机 *)
    state            : INT := 0;       (* 当前状态，初始为 ST_INIT *)

    (* 视觉通讯寄存器映射（详见 Vision_ModbusTCP.st）*)
    xVisionTrigger   : INT := 0;       (* HR100，PLC→Vision：触发拍照 *)
    xDefectClass     : INT := -1;      (* HR110，Vision→PLC：0=hole, 1=chip, 2=scratch, 3=stain, -1=未识别 *)
    fConfidence      : INT := 0;       (* HR111，×1000 *)
    xResultValid     : INT := 0;       (* HR112 *)
    iResultSeq       : INT := 0;       (* HR113 *)

    (* 内部变量 *)
    last_seq         : INT := -1;      (* 上一次收到的 seq，用于校验丢包 *)
    target_col       : INT := 0;       (* 目标库位列 1~4 *)
    target_row       : INT := 0;       (* 目标库位层 1~3 *)
    cur_x, cur_y, cur_z : REAL := 0.0;
    home_x           : REAL := 0.0;
    home_y           : REAL := 0.0;
    home_z           : REAL := 100.0;  (* 安全高度 *)

    (* 报警 *)
    alarm_code       : INT := 0;

    (* 计数器 *)
    processed_count  : INT := 0;       (* 已处理的物料数 *)
END_VAR

(* ==========================================================================*)
(* 主状态机*)
(* ==========================================================================*)
CASE state OF

    0: (* ST_INIT — 初始化 *)
        IF TRUE THEN
            (* 复位所有视觉触发 *)
            xVisionTrigger := 0;
            processed_count := 0;
            (* 复位到 home *)
            cur_x := 0.0;
            cur_y := 0.0;
            cur_z := 100.0;
            state := 10;  (* ST_WAIT_VISION *)
        END_IF;

    10: (* ST_WAIT_VISION — 等待视觉检测结果 *)
        (* 上升沿触发：写 xVisionTrigger = 1 *)
        IF TRUE THEN
            xVisionTrigger := 1;
        END_IF;

        (* 检查视觉返回 *)
        IF xResultValid = 1 AND iResultSeq > last_seq THEN
            last_seq := iResultSeq;
            (* 复位 trigger，准备下次触发 *)
            xVisionTrigger := 0;

            (* 检查类别是否在合法范围 *)
            IF xDefectClass >= 0 AND xDefectClass <= 3 THEN
                state := 20;  (* ST_CALC_TARGET *)
            ELSIF xDefectClass = -1 THEN
                (* 未识别，报警 *)
                alarm_code := 1001;  (* "未识别" *)
                state := 99;
            ELSE
                (* 非法值 *)
                alarm_code := 1002;
                state := 99;
            END_IF;
        END_IF;

    20: (* ST_CALC_TARGET — 计算目标库位 *)
        (* 根据 defect_class 计算目标列 *)
        CASE xDefectClass OF
            0: target_col := 1;   (* hole → 列1 *)
            1: target_col := 2;   (* chip → 列2 *)
            2: target_col := 3;   (* scratch → 列3 *)
            3: target_col := 4;   (* stain → 列4 *)
        END_CASE;

        (* 层：第 1、5、9 号物料放第1层；2、6、10放第2层；3、7、11放第3层；4、8、12放第1层（循环）*)
        target_row := ((processed_count - 1) MOD 3) + 1;
        state := 30;

    30: (* ST_MOVE_AX — 调用 Axis_XYZ 移动到目标 *)
        IF TRUE THEN
            (* 此处调用 FB_Axis_XYZ 的实例，控制三轴*)
            (* 详见 Axis_XYZ.st *)
            IF TRUE THEN
                state := 40;
            END_IF;
        END_IF;

    40: (* ST_RELEASE — 放料 *)
        IF TRUE THEN
            (* 释放气爪/吸盘 *)
            processed_count := processed_count + 1;
            state := 50;
        END_IF;

    50: (* ST_RETURN_HOME — 回归零位 *)
        IF TRUE THEN
            cur_x := home_x;
            cur_y := home_y;
            cur_z := home_z;
            state := 60;
        END_IF;

    60: (* ST_READY — 准备下一个 *)
        IF processed_count >= 12 THEN
            (* 已完成 12 个物料，循环结束 *)
            state := 0;  (* 回到 ST_INIT，准备下一轮 *)
        ELSE
            state := 10;  (* 继续 ST_WAIT_VISION *)
        END_IF;

    99: (* ST_ALARM — 报警 *)
        (* 此处触发蜂鸣器/指示灯/停机信号*)
        (* 等用户按"复位"按钮后再恢复 *)
        IF FALSE THEN
            state := 0;
        END_IF;

ELSE
    state := 99;
END_CASE;
END_PROGRAM