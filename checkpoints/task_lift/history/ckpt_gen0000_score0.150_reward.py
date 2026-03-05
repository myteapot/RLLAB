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
    # Smooth, bounded in [0,1]
    reach_r = float(np.exp(-(dist / reach_sigma) ** 2))

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
    # Encourage closing only when near and aligned; discourage premature closing
    close_when_ready = float(close_cmd > 0.05) * float(near) * align_xy
    premature_close = float(close_cmd > 0.05) * float(not near) * (1.0 - reach_r)

    # Reward being closed when near; but don't overreward hovering closed
    closedness = 1.0 - float(np.clip(g_open, 0.0, 1.0))
    grasp_shaping = closedness * float(near) * align_xy

    # Extra bonus for actual grasped state (anti-hacking)
    grasp_bonus = 2.0 if is_grasped else 0.0

    # --- Stage 4: Lift (height shaping) ---
    # Only pays substantially once grasped, to prevent hover hacking
    lift_amt = max(0.0, obj_h - table_z)
    lift_progress = float(np.clip((lift_amt - lift_success) / max(1e-6, (lift_target - lift_success)), 0.0, 1.0))
    lift_r = lift_progress if is_grasped else 0.15 * lift_progress  # tiny shaping if not grasped

    # --- Anti-hacking: penalize hovering very near without grasp/lift ---
    hover_hack = 0.0
    if (dist < 0.025) and (not is_grasped) and (lift_amt < 0.02):
        # stronger penalty as episode progresses
        t = step / max(1, horizon)
        hover_hack = (0.2 + 0.4 * t) * (1.0 - closedness)  # hovering while open

    # --- Action smoothness / jerk penalty ---
    a = np.asarray(action, dtype=float)
    act_pen = 0.05 * float(np.sum(a ** 2))
    # Penalize large gripper oscillations near object
    grip_osc_pen = 0.02 * float(abs(a[-1])) * float(dist < 0.06)

    # Compose reward with staged weights
    r = 0.0
    r += 2.0 * reach_r
    r += 2.0 * align_r
    r += 0.8 * grasp_shaping
    r += 0.6 * close_when_ready
    r -= 0.6 * premature_close
    r += grasp_bonus
    r += 4.0 * lift_r
    # Small time bonus for completing lift early
    if is_grasped and lift_progress >= 0.95:
        r += 1.0 * (1.0 - step / max(1, horizon))

    r -= hover_hack
    r -= act_pen
    r -= grip_osc_pen

    # Keep roughly within desired range
    r = float(np.clip(r, -1.0, 10.0))
    return r