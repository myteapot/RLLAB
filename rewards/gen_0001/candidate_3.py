def reward_fn(obs, action, info) -> float:
    import numpy as np
    import math

    # Safe getters
    gripper_pos = np.asarray(info.get("gripper_pos", np.zeros(3)), dtype=float)
    object_pos = np.asarray(info.get("object_pos", np.zeros(3)), dtype=float)
    obj_h = float(info.get("object_height", float(object_pos[2]) if object_pos.shape[0] >= 3 else 0.0))
    dist = float(info.get("distance", float(np.linalg.norm(gripper_pos - object_pos))))
    is_grasped = bool(info.get("is_grasped", False))
    g_open = float(info.get("gripper_open", 1.0))
    step = int(info.get("step", 0))
    horizon = int(info.get("horizon", 200))

    # ----- Constants (tuned for ~2.5cm cube) -----
    table_z = 0.0

    # Approach / align geometry
    reach_sigma = 0.10
    xy_sigma = 0.025
    z_sigma = 0.025
    target_hover = 0.040  # desired gripper z above object pregrasp
    pregrasp_dist = 0.05
    grasp_dist = 0.028

    # Lift
    lift_min = 0.03
    lift_target = 0.12

    # Action intent
    a = np.asarray(action, dtype=float)
    close_intent = float(np.clip(-a[-1], -1.0, 1.0))  # positive means "close"
    open_intent = float(np.clip(a[-1], -1.0, 1.0))
    closedness = 1.0 - float(np.clip(g_open, 0.0, 1.0))

    # ----- Stage 1: Reach -----
    # Bounded [0,1], strong gradient when far
    reach_r = float(np.exp(-((dist / reach_sigma) ** 2)))

    # ----- Stage 2: Align (center in XY and hover Z) -----
    dxy = float(np.linalg.norm(gripper_pos[:2] - object_pos[:2]))
    dz = float(gripper_pos[2] - (object_pos[2] + target_hover))
    align_xy = float(np.exp(-((dxy / xy_sigma) ** 2)))
    align_z = float(np.exp(-((dz / z_sigma) ** 2)))
    # Gate alignment by proximity so it doesn't pay at distance
    prox_gate = float(np.exp(-max(0.0, dist - 0.12) / 0.05))
    align_r = align_xy * align_z * prox_gate

    # Extra shaping: be above the object (avoid side approaches that can "cheat" distance)
    above_gate = 1.0 if (gripper_pos[2] >= object_pos[2] + 0.01) else 0.6
    align_r *= above_gate

    # ----- Stage 3: Grasp (only when close & aligned) -----
    near_pregrasp = dist < pregrasp_dist
    near_grasp = dist < grasp_dist

    # Encourage closing when ready; discourage closing too early
    ready = float(near_pregrasp) * align_xy
    close_when_ready = float(close_intent > 0.1) * ready
    premature_close = float(close_intent > 0.1) * float(not near_pregrasp) * (1.0 - reach_r)

    # Reward being closed only when actually in grasping region (prevents "close and hover")
    grasp_shaping = closedness * float(near_grasp) * align_xy

    # Anti-hacking: penalize being very close, well-aligned, but not attempting to close
    no_close_pen = 0.0
    if (not is_grasped) and (dist < 0.035) and (align_xy > 0.6) and (close_intent < 0.05) and (closedness < 0.4):
        t = step / max(1, horizon)
        no_close_pen = (0.15 + 0.25 * t)

    # Bonus for actual grasped state (primary success signal)
    grasp_bonus = 2.5 if is_grasped else 0.0

    # ----- Stage 4: Lift (mostly only if grasped) -----
    lift_amt = max(0.0, obj_h - table_z)
    lift_prog = float(np.clip((lift_amt - lift_min) / max(1e-6, (lift_target - lift_min)), 0.0, 1.0))
    # Strongly gate lift by is_grasped to avoid hover/lift hacking
    lift_r = (lift_prog if is_grasped else 0.05 * lift_prog)

    # Extra bonus for maintaining closed grip while grasped (stability)
    grip_hold_bonus = (0.4 * closedness) if is_grasped else 0.0

    # Time bonus for successful high lift
    time_bonus = 0.0
    if is_grasped and lift_prog > 0.95:
        time_bonus = 1.2 * (1.0 - step / max(1, horizon))

    # ----- Anti-hacking: hovering near object without lift -----
    hover_pen = 0.0
    if (not is_grasped) and (dist < 0.030) and (lift_amt < 0.015):
        t = step / max(1, horizon)
        # penalize especially if gripper is open / not closing
        hover_pen = (0.10 + 0.35 * t) * (1.0 - min(1.0, close_intent + closedness))

    # ----- Smoothness penalties -----
    # L2 action penalty (jerk / energy)
    act_pen = 0.06 * float(np.sum(a[:-1] ** 2)) + 0.02 * float(a[-1] ** 2)

    # Penalize gripper oscillation near object (open/close chatter)
    grip_osc_pen = 0.03 * float(abs(a[-1])) * float(dist < 0.07)

    # Penalize opening while grasped / lifting (dropping)
    open_while_grasped_pen = 0.0
    if is_grasped and open_intent > 0.1:
        open_while_grasped_pen = 0.6 * float(open_intent)

    # ----- Compose reward -----
    r = 0.0
    r += 2.2 * reach_r
    r += 2.6 * align_r
    r += 1.0 * grasp_shaping
    r += 0.7 * close_when_ready
    r -= 0.7 * premature_close
    r -= no_close_pen
    r += grasp_bonus
    r += 4.6 * lift_r
    r += grip_hold_bonus
    r += time_bonus

    r -= hover_pen
    r -= act_pen
    r -= grip_osc_pen
    r -= open_while_grasped_pen

    return float(np.clip(r, -1.0, 10.0))