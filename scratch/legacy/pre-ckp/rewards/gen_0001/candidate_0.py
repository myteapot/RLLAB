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
    reach_r = float(np.exp(-(dist / reach_sigma) ** 2))
    # Add a slightly "sharper" term that helps policy commit when close (better grasp guidance)
    reach_close_r = float(np.exp(-(dist / 0.04) ** 2))

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

    # Encourage closing only when near and aligned; discourage premature closing
    # Make "ready" depend on both xy and z alignment (prevents side-closing away from correct height)
    ready = float(near) * align_r
    close_when_ready = float(close_cmd > 0.05) * ready

    # Stronger premature close penalty (common failure mode: close early then push object)
    premature_close = float(close_cmd > 0.05) * float(not near) * (1.0 - reach_r)

    # Reward being closed only when in a good pregrasp pose; avoid rewarding "closed hover" too much
    grasp_shaping = closedness * ready

    # NEW: discourage staying open when very near and well aligned (helps success_rate)
    not_closing_pen = 0.0
    if (not is_grasped) and near:
        not_closing_pen = (1.0 - closedness) * align_r

    # Extra bonus for actual grasped state (anti-hacking)
    grasp_bonus = 2.0 if is_grasped else 0.0

    # NEW: stability reward for sustaining a grasp (scaled by time in episode)
    # Rewards maintaining grasp and being reasonably closed; ramps up later to prevent drop/oscillation.
    t = step / max(1, horizon)
    sustain_r = 0.0
    if is_grasped:
        sustain_r = (0.6 + 0.6 * t) * (0.5 + 0.5 * closedness)

    # --- Stage 4: Lift (height shaping) ---
    lift_amt = max(0.0, obj_h - table_z)
    lift_progress = float(np.clip((lift_amt - lift_success) / max(1e-6, (lift_target - lift_success)), 0.0, 1.0))
    # Reduce non-grasped lift shaping further to avoid "bump the object" hacks
    lift_r = lift_progress if is_grasped else 0.05 * lift_progress

    # --- Anti-hacking: penalize hovering very near without grasp/lift ---
    hover_hack = 0.0
    if (dist < 0.025) and (not is_grasped) and (lift_amt < 0.02):
        hover_hack = (0.25 + 0.55 * t) * (1.0 - closedness)  # hovering while open

    # NEW: mild time pressure to improve efficiency without overpowering sparse success reward
    step_pen = 0.0025  # up to ~0.5 over 200 steps

    # --- Action smoothness / jerk penalty ---
    a = np.asarray(action, dtype=float)
    # Slightly stronger general action penalty
    act_pen = 0.07 * float(np.sum(a ** 2))
    # Penalize large gripper oscillations near object a bit more
    grip_osc_pen = 0.03 * float(abs(a[-1])) * float(dist < 0.06)

    # Compose reward with staged weights
    r = 0.0
    r += 2.2 * reach_r
    r += 0.6 * reach_close_r
    r += 2.2 * align_r
    r += 1.0 * grasp_shaping
    r += 0.8 * close_when_ready
    r -= 0.8 * premature_close
    r -= 0.6 * not_closing_pen
    r += grasp_bonus
    r += 1.2 * sustain_r
    r += 4.2 * lift_r

    # Small time bonus for completing lift early
    if is_grasped and lift_progress >= 0.95:
        r += 1.2 * (1.0 - t)

    r -= hover_hack
    r -= step_pen
    r -= act_pen
    r -= grip_osc_pen

    r = float(np.clip(r, -1.0, 10.0))
    return r