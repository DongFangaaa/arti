(* ==========================================================================*)
(* 三轴伺服控制功能块 — X/Y/Z 联动*)
(* ==========================================================================*)
(* 依赖：PLCnext Engineer 库的 MC_MoveAbsolute, MC_Home 等*)
(* 调用方式：在 Main.prg 中实例化 MC_MoveAbsolute_X/Y/Z 三次实现联动*)
(* ==========================================================================*)
FUNCTION_BLOCK FB_Axis_XYZ
VAR_INPUT
    bExecute         : BOOL;            (* 上升沿触发移动 *)
    target_x         : REAL;            (* X 目标位置 (mm) *)
    target_y         : REAL;            (* Y 目标位置 (mm) *)
    target_z         : REAL;            (* Z 目标位置 (mm) *)
END_VAR
VAR_OUTPUT
    bDone            : BOOL;            (* 到达目标 *)
    bError           : BOOL;            (* 错误 *)
    iErrorId         : INT;             (* 错误码 *)
END_VAR
VAR
    (* 三轴实例（PLCnext MC 库的 FB 实例）*)
    MC_X : MC_MoveAbsolute;             (* X 轴 *)
    MC_Y : MC_MoveAbsolute;             (* Y 轴 *)
    MC_Z : MC_MoveAbsolute;             (* Z 轴 *)

    (* 内部状态 *)
    state            : INT := 0;
    exec_old         : BOOL := FALSE;

    (* 速度/加速度参数（可在 PLCnext 中调整）*)
    vel_x            : REAL := 100.0;   (* mm/s *)
    vel_y            : REAL := 100.0;
    vel_z            : REAL := 50.0;    (* Z 轴慢一点 *)
    accel            : REAL := 500.0;   (* mm/s² *)
    decel            : REAL := 500.0;
END_VAR

(* ==========================================================================*)
(* 主体*)
(* ==========================================================================*)
(* 上升沿触发 *)
IF bExecute AND NOT exec_old THEN
    exec_old := TRUE;
    state := 10;

    (* 先升 Z 到安全高度 *)
    MC_Z(
        Execute := TRUE,
        Position := 50.0,
        Velocity := vel_z,
        Acceleration := accel,
        Deceleration := decel,
    );
END_IF;

IF NOT bExecute THEN
    exec_old := FALSE;
END_IF;

CASE state OF

    10: (* Z 升到安全高度 *)
        IF MC_Z.Done THEN
            MC_X(
                Execute := TRUE,
                Position := target_x,
                Velocity := vel_x,
                Acceleration := accel,
                Deceleration := decel,
            );
            MC_Y(
                Execute := TRUE,
                Position := target_y,
                Velocity := vel_y,
                Acceleration := accel,
                Deceleration := decel,
            );
            state := 20;
        ELSIF MC_Z.Error THEN
            iErrorId := MC_Z.ErrorID;
            bError := TRUE;
            state := 99;
        END_IF;

    20: (* X/Y 同时移动 *)
        IF MC_X.Done AND MC_Y.Done THEN
            MC_Z(
                Execute := TRUE,
                Position := target_z,
                Velocity := vel_z,
                Acceleration := accel,
                Deceleration := decel,
            );
            state := 30;
        ELSIF MC_X.Error OR MC_Y.Error THEN
            bError := TRUE;
            state := 99;
        END_IF;

    30: (* Z 下降到目标 *)
        IF MC_Z.Done THEN
            bDone := TRUE;
            state := 0;
        ELSIF MC_Z.Error THEN
            bError := TRUE;
            state := 99;
        END_IF;

    99: (* 错误状态 *)
        bDone := FALSE;
        (* 等用户复位 *)
        IF NOT bExecute THEN
            state := 0;
            bError := FALSE;
            iErrorId := 0;
        END_IF;

ELSE
    state := 0;
END_CASE;
END_FUNCTION_BLOCK