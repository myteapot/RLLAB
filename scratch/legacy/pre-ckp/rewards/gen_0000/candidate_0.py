def reward_fn(obs, action, info):
    import numpy as np
    import math

    # --- Extract ---
    dist = float(info.get("distance", 0.0))
    gpos = np.array(info.get("gripper_pos", np.zeros(3)), dtype=float)
    opos = np.array(info.get("object_pos", np.zeros(3)), dtype=float)
    obj_h = float(info.get("object_height", opos[2] if opos.shape[0] >= 3 else 0.0))
    is_grasped = bool(info.get("is_grasped", False))
    gr_open = float(info.get("gripper_open", 1.0))

    # --- Constants / targets ---
    table_z = 0.0
    # If table height isn't known, infer a conservative estimate from object height (cube on table)
    # cube is 2.5cm; center ~ table + 1.25cm when resting
    if obj_h > 0.0:
        table_z = min(table_z, obj_h - 0.0125) if table_z != 0.0 else (obj_h - 0.0125)

    # Stage thresholds
    d_close = 0.04       # within 4cm: start grasping
    d_good = 0.02        # very close
    z_above = 0.035      # desired gripper height above object center during approach
    lift_target = (table_z + 0.0125) + 0.10  # lift cube center by ~10cm above resting

    # --- Helper shaping ---
    def exp_shaping(x, scale):
        return math.exp(- (x * x) / max(scale * scale, 1e-8))

    # --- Reach: encourage reducing distance robustly ---
    # In [0, 1]
    reach = exp_shaping(dist, 0.08)

    # --- Align: prefer gripper above object and laterally centered ---
    lateral = float(np.linalg.norm((gpos - opos)[:2]))
    above_err = float((gpos[2] - (opos[2] + z_above)))
    align_xy = exp_shaping(lateral, 0.04)
    align_z = exp_shaping(above_err, 0.05)
    align = align_xy * align_z  # in [0,1]

    # --- Grasp intent: close when near, open when far ---
    # action[-1] < 0 means closing
    a_grip = float(action[-1]) if action is not None and len(np.atleast_1d(action)) > 0 else 0.0
    close_cmd = max(0.0, -a_grip)  # 0..1-ish
    open_cmd = max(0.0, a_grip)

    near = 1.0 / (1.0 + math.exp((dist - d_close) / 0.01))  # ~1 when dist<d_close
    very_near = 1.0 / (1.0 + math.exp((dist - d_good) / 0.005))

    # Encourage closing only when near and well-aligned; discourage closing when far
    grasp_cmd_reward = (0.8 * close_cmd * near * align) - (0.4 * close_cmd * (1.0 - near))
    # Encourage being closed when very near (prevents "hover open")
    closure_reward = (1.0 - gr_open) * very_near * align * 0.6
    # Discourage opening when near/aligned (prevents backing off without grasp)
    open_penalty = open_cmd * near * align * 0.4

    # --- Lift: reward actual object height only after grasp (avoid hover hacking) ---
    # Use is_grasped as a gate (provided: True if object lifted above table).
    # Still give small continuous lift reward based on height but gated.
    resting_center = (table_z + 0.0125)
    lift_progress = (obj_h - resting_center) / max((lift_target - resting_center), 1e-6)
    lift_progress = float(np.clip(lift_progress, 0.0, 1.0))
    lift_reward = (6.0 * lift_progress) if is_grasped else (0.2 * lift_progress)

    # Bonus for actually being grasped (sparse-ish)
    grasp_bonus = 2.0 if is_grasped else 0.0

    # --- Anti-hacking: penalize being close above object but never grasping (hovering) ---
    hover_like = (align_xy * exp_shaping(max(0.0, gpos[2] - opos[2]), 0.08))  # close in xy and near z vicinity
    hover_penalty = 0.0
    if not is_grasped:
        # If very near and aligned for a while, require closing; otherwise penalize
        hover_penalty = 0.8 * very_near * align * gr_open  # penalize being open when perfectly set up

    # --- Smoothness / action penalty ---
    a = np.array(action, dtype=float).ravel() if action is not None else np.zeros(7, dtype=float)
    act_mag = float(np.mean(np.square(np.clip(a, -1.0, 1.0))))
    action_penalty = 0.15 * act_mag

    # --- Total reward composition ---
    reward = 0.0
    reward += 1.6 * reach
    reward += 1.2 * align
    reward += grasp_cmd_reward + closure_reward - open_penalty
    reward += grasp_bonus
    reward += lift_reward
    reward -= hover_penalty
    reward -= action_penalty

    # Keep roughly within [-1, 10] without hard clipping artifacts too much
    reward = float(np.clip(reward, -1.0, 10.0))
    return reward