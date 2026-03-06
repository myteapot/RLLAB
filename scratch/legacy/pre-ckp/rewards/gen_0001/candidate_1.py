def reward_fn(obs, action, info) -> float:
    # Safe getters
    gripper_pos = np.asarray(info.get("gripper_pos", np.zeros(3)), dtype=float)
    object_pos = np.asarray(info.get("object_pos", np.zeros(3)), dtype=float)
    obj_h = float(info.get("object_height", object_pos[2] if object_pos.shape[0] >= 3 else 0.0))
    dist = float(info.get("distance", np.linalg.norm(gripper_pos - object_pos)))
    is_grasped = bool(info.get("is_grasped", False))
    g_open = float(info.get("gripper_open", 1.0))
    step = int(info.get("step", 0))
    horizon = int(info.get("horizon", 200))

    # Optional temporal info (for stability / smoothness)
    prev_is_grasped = bool(info.get("prev_is_grasped", info.get("was_grasped", False)))
    prev_action = info.get("prev_action", None)

    # Constants (tuned for a ~2.5cm cube)
    table_z = 0.0
    reach_sigma = 0.08     # meters
    xy_sigma = 0.04        # meters
    z_sigma = 0.04         # meters
    target_hover = 0.035   # gripper above object center for pregrasp
    near_thresh = 0.03     # distance considered "close"
    lift_target = 0.10     # meters above table considered good lift
    lift_success = 0.06    # minimal lift to count as lifted in reward shaping
    close_cmd = float(np.clip(-action[-1], -1.0, 1.0))  # positive -> closing intent

    # --- Stage 1: Reach (distance shaping) ---
    reach_r = float(np.exp(-(dist / reach_sigma) ** 2))
    # Slightly stronger near-object shaping to help low success-rate
    reach_boost = float(np.exp(-(dist / 0.04) ** 2))

    # --- Stage 2: Align (be above and centered over object) ---
    dxy = float(np.linalg.norm(gripper_pos[:2] - object_pos[:2]))
    dz = float(gripper_pos[2] - (object_pos[2] + target_hover))
    align_xy = float(np.exp(-(dxy / xy_sigma) ** 2))
    align_z = float(np.exp(-(dz / z_sigma) ** 2))
    align_r = align_xy * align_z

    # Gate alignment by being reasonably close, so it doesn't pay from far away
    align_gate = float(np.exp(-max(0.0, dist - 0.12) / 0.06))
    align_r *= align_gate

    # --- Stage 3: Grasp (close when close & aligned) ---
    near = dist < near_thresh
    closedness = 1.0 - float(np.clip(g_open, 0.0, 1.0))

    # Encourage "approach open, close when ready" more explicitly
    open_when_far = float((not near) and (dist > 0.06)) * float(np.clip(g_open, 0.0, 1.0)) * (1.0 - reach_r)

    close_ready_gate = float(near) * align_xy * float(np.clip(align_z, 0.0, 1.0))
    close_when_ready = float(close_cmd > 0.05) * close_ready_gate

    # Penalize premature closing, but soften to avoid discouraging recovery
    premature_close = float(close_cmd > 0.05) * float(not near) * (1.0 - reach_r) * 0.8

    # Reward being closed when near & aligned (pre-grasp hold)
    grasp_shaping = closedness * close_ready_gate

    # Extra bonus for actual grasped state (anti-hacking)
    grasp_bonus = 2.0 if is_grasped else 0.0

    # Stability: reward sustaining a grasp; penalize dropping once grasped
    sustain_bonus = 0.0
    drop_pen = 0.0
    if is_grasped:
        # prefer keeping gripper closed while grasped
        sustain_bonus = 0.35 + 0.25 * closedness
    if (not is_grasped) and prev_is_grasped:
        drop_pen = 0.6

    # --- Stage 4: Lift (height shaping) ---
    lift_amt = max(0.0, obj_h - table_z)
    lift_progress = float(np.clip((lift_amt - lift_success) / max(1e-6, (lift_target - lift_success)), 0.0, 1.0))
    # Slightly more shaping even before grasp to encourage correct vertical motion near object
    lift_r = lift_progress if is_grasped else 0.25 * lift_progress * float(dist < 0.06) * align_xy

    # --- Anti-hacking: penalize hovering very near without grasp/lift ---
    hover_hack = 0.0
    if (dist < 0.025) and (not is_grasped) and (lift_amt < 0.02):
        t = step / max(1, horizon)
        hover_hack = (0.2 + 0.5 * t) * (1.0 - closedness)

    # --- Time pressure (efficiency) ---
    # Mild per-step penalty; slightly reduced once grasped to avoid discouraging careful lift
    tfrac = step / max(1, horizon)
    step_pen = 0.01 + 0.01 * tfrac
    if is_grasped:
        step_pen *= 0.6

    # --- Action smoothness / jerk penalty ---
    a = np.asarray(action, dtype=float)
    act_pen = 0.06 * float(np.sum(a ** 2))  # slightly stronger for smoothness
    grip_osc_pen = 0.03 * float(abs(a[-1])) * float(dist < 0.06)

    # Jerk penalty if prev_action available
    jerk_pen = 0.0
    if prev_action is not None:
        pa = np.asarray(prev_action, dtype=float)
        if pa.shape == a.shape:
            da = a - pa
            jerk_pen = 0.02 * float(np.sum(da ** 2))
            # extra discourage gripper dithering near object
            jerk_pen += 0.01 * float((da[-1] ** 2)) * float(dist < 0.06)

    # Compose reward with staged weights
    r = 0.0
    r += 2.2 * reach_r
    r += 0.6 * reach_boost
    r += 2.1 * align_r
    r += 0.9 * grasp_shaping
    r += 0.7 * close_when_ready
    r += 0.25 * open_when_far
    r -= 0.5 * premature_close
    r += grasp_bonus
    r += sustain_bonus
    r -= drop_pen
    r += 4.2 * lift_r

    # Small time bonus for completing lift early
    if is_grasped and lift_progress >= 0.95:
        r += 1.2 * (1.0 - step / max(1, horizon))

    r -= hover_hack
    r -= step_pen
    r -= act_pen
    r -= grip_osc_pen
    r -= jerk_pen

    r = float(np.clip(r, -1.0, 10.0))
    return r