def reward_fn(obs, action, info):
    # Safe getters
    d = float(info.get("distance", 1.0))
    gripper_pos = np.array(info.get("gripper_pos", np.zeros(3)), dtype=float)
    object_pos = np.array(info.get("object_pos", np.zeros(3)), dtype=float)
    obj_h = float(info.get("object_height", object_pos[2] if object_pos.shape[0] >= 3 else 0.0))
    is_grasped = bool(info.get("is_grasped", False))
    gopen = float(info.get("gripper_open", 1.0))

    # --- Stage 1: Reach (dense distance shaping) ---
    # Smooth bounded shaping in [0,1)
    reach = 1.0 - np.tanh(4.0 * d)  # close -> ~1, far -> ~0

    # --- Stage 2: Align (favor being above and laterally centered) ---
    rel = object_pos - gripper_pos
    xy_dist = float(np.linalg.norm(rel[:2]))
    z_rel = float(rel[2])  # positive means object above gripper

    # Prefer small lateral offset and gripper being slightly above the object
    xy_align = np.exp(- (xy_dist / 0.03) ** 2)  # ~1 within 3cm
    # target gripper to be ~2cm above object during approach
    z_target = 0.02
    z_align = np.exp(- ((-z_rel - z_target) / 0.02) ** 2)  # using -z_rel = gripper_z - obj_z

    align = xy_align * z_align

    # --- Stage 3: Grasp (encourage closing only when appropriately positioned) ---
    # action[-1] < 0 means closing
    grip_cmd = float(action[-1]) if np.size(action) >= 1 else 0.0
    closing = np.clip(-grip_cmd, 0.0, 1.0)  # 0..1
    close_enough = 1.0 if (d < 0.045) else 0.0
    well_aligned = 1.0 if (xy_dist < 0.02 and (-z_rel) > 0.005 and (-z_rel) < 0.05) else 0.0

    # Encourage gripper to be closed when close+aligned; discourage closing when far/misaligned
    grasp_intent = closing * close_enough * well_aligned
    premature_close_pen = closing * (1.0 - close_enough) * 0.5

    # Encourage actually being closed near the object (state-based)
    closed_state = 1.0 - np.clip(gopen, 0.0, 1.0)
    grasp_state_bonus = closed_state * close_enough * well_aligned

    # --- Stage 4: Lift (reward height only after grasp) ---
    # Estimate table height from early episode object height
    step = int(info.get("step", 0))
    key0 = "_table_h_est"
    if key0 not in info:
        info[key0] = float(obj_h)  # initial estimate
    # Update estimate for first few steps to be robust
    if step < 5:
        info[key0] = 0.8 * float(info[key0]) + 0.2 * float(obj_h)
    table_h = float(info.get(key0, obj_h))
    lift_h = max(0.0, obj_h - table_h)

    # Height shaping (up to ~8cm)
    lift_progress = np.clip(lift_h / 0.08, 0.0, 1.0)

    # Only pay lift if grasped; otherwise hovering shouldn't pay much
    lift_reward = (6.0 * lift_progress) if is_grasped else (0.5 * lift_progress * align * reach)

    # --- Anti-hover / anti-hack: require contact-like conditions for big rewards ---
    # If very close and aligned but gripper remains open, lightly penalize lingering.
    hover_pen = 0.0
    if (d < 0.035) and (xy_dist < 0.02) and (align > 0.5) and (gopen > 0.7) and (not is_grasped):
        hover_pen = 0.2

    # --- Action smoothness penalty ---
    a = np.array(action, dtype=float).ravel()
    act_mag = float(np.linalg.norm(a))
    key1 = "_prev_action"
    prev = info.get(key1, np.zeros_like(a))
    prev = np.array(prev, dtype=float).ravel()
    if prev.shape != a.shape:
        prev = np.zeros_like(a)
    jerk = float(np.linalg.norm(a - prev))
    info[key1] = a.copy()

    action_pen = 0.05 * act_mag + 0.1 * jerk

    # --- Combine ---
    reward = 0.0
    reward += 1.5 * reach
    reward += 1.0 * align * (0.5 + 0.5 * reach)
    reward += 1.0 * grasp_intent
    reward += 0.5 * grasp_state_bonus
    reward -= premature_close_pen
    reward += lift_reward
    reward -= hover_pen
    reward -= action_pen

    # Success bonus for being grasped and lifted noticeably
    if is_grasped and lift_h > 0.03:
        reward += 2.0

    # Keep within a reasonable range
    reward = float(np.clip(reward, -1.0, 10.0))
    return reward