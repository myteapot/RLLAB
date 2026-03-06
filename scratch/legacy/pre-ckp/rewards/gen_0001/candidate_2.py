def reward_fn(obs, action, info) -> float:
    gripper_pos = np.asarray(info.get("gripper_pos", np.zeros(3)), dtype=float)
    object_pos = np.asarray(info.get("object_pos", np.zeros(3)), dtype=float)
    obj_h = float(info.get("object_height", object_pos[2] if object_pos.shape[0] >= 3 else 0.0))
    dist = float(info.get("distance", np.linalg.norm(gripper_pos - object_pos)))
    is_grasped = bool(info.get("is_grasped", False))
    g_open = float(info.get("gripper_open", 1.0))
    step = int(info.get("step", 0))
    horizon = int(info.get("horizon", 200))

    a = np.asarray(action, dtype=float)
    if a.shape[0] < 1:
        a = np.zeros(7, dtype=float)

    # Constants (meters)
    table_z = 0.0
    cube_size = 0.025
    hover_z = 0.045  # pregrasp hover height above cube center
    pregrasp_z_tol = 0.03

    reach_sigma = 0.10
    xy_sigma = 0.03
    z_sigma = 0.035

    near_dist = 0.03
    contact_dist = 0.018

    lift_success = 0.05
    lift_target = 0.12

    # Helper terms
    dxyz = gripper_pos - object_pos
    dxy = float(np.linalg.norm(dxyz[:2]))
    dz_pre = float(gripper_pos[2] - (object_pos[2] + hover_z))
    above = float(gripper_pos[2] >= object_pos[2] + 0.005)

    closedness = 1.0 - float(np.clip(g_open, 0.0, 1.0))
    close_cmd = float(np.clip(-a[-1], -1.0, 1.0))  # + => closing intent
    open_cmd = float(np.clip(a[-1], -1.0, 1.0))    # + => opening intent
    t = float(step) / float(max(1, horizon))

    # --- Stage 1: Reach ---
    reach_r = float(np.exp(-((dist / reach_sigma) ** 2)))

    # --- Stage 2: Align (center in XY, and at hover Z) ---
    align_xy = float(np.exp(-((dxy / xy_sigma) ** 2)))
    align_z = float(np.exp(-((dz_pre / z_sigma) ** 2)))
    # Only care about alignment once within a reasonable radius
    align_gate = float(np.exp(-max(0.0, dist - 0.12) / 0.05))
    align_r = align_xy * align_z * align_gate

    # Encourage being above the object before descending (avoid side swipes)
    above_r = above * float(np.exp(-((dxy / 0.05) ** 2)))

    # --- Stage 3: Grasp shaping (only when close and well-aligned) ---
    near = dist < near_dist
    contact_like = dist < contact_dist

    # Reward closing when ready, penalize closing early/far
    ready = float(near) * align_xy * float(abs(dz_pre) < pregrasp_z_tol) * above
    close_when_ready = ready * float(close_cmd > 0.05)
    premature_close = float(close_cmd > 0.05) * float(not near) * (1.0 - reach_r)

    # Reward being closed only when in contact-like proximity and aligned (prevents hovering closed)
    grasp_shaping = closedness * float(contact_like) * align_xy * above

    # Strong bonus only for true grasp state
    grasp_bonus = 2.5 if is_grasped else 0.0

    # Discourage opening when near/contact (dropping / refusal to grasp)
    open_near_pen = float(open_cmd > 0.05) * float(dist < 0.04) * (0.5 + 0.5 * align_xy)

    # --- Stage 4: Lift (mostly gated by is_grasped) ---
    lift_amt = max(0.0, obj_h - table_z)
    lift_prog = float(np.clip((lift_amt - lift_success) / max(1e-6, (lift_target - lift_success)), 0.0, 1.0))

    # Tiny lift shaping when very near and closed (to help discover grasp->lift), but not from hover
    near_and_closed = float(near) * closedness * align_xy * above
    lift_r = (lift_prog if is_grasped else 0.08 * lift_prog * near_and_closed)

    # Extra "stability" bonus: lifted and gripper remains reasonably closed (reduces toss/drop hacks)
    lift_hold_bonus = 0.0
    if is_grasped and lift_prog > 0.2:
        lift_hold_bonus = 0.6 * closedness * lift_prog

    # --- Anti-hacking: hovering near without grasping/lifting ---
    hover_pen = 0.0
    if (dist < 0.025) and (not is_grasped) and (lift_amt < 0.02):
        # penalize especially if staying open and aligned (hovering)
        hover_pen = (0.25 + 0.55 * t) * (1.0 - closedness) * (0.3 + 0.7 * align_xy)

    # Penalize being too low next to object without grasp (table collisions / pushing)
    low_side_pen = 0.0
    if (not is_grasped) and (dxy < 0.04) and (gripper_pos[2] < object_pos[2] + 0.01):
        low_side_pen = 0.15 * (1.0 - float(np.clip((gripper_pos[2] - table_z) / 0.06, 0.0, 1.0))) * (0.5 + 0.5 * align_xy)

    # --- Smoothness penalties ---
    # Overall action magnitude
    act_mag_pen = 0.04 * float(np.sum(a * a))
    # Extra penalty on gripper command oscillation near the object
    grip_cmd_pen = 0.02 * float(abs(a[-1])) * float(dist < 0.06)

    # Compose
    r = 0.0
    r += 1.8 * reach_r
    r += 2.2 * align_r
    r += 0.3 * above_r

    r += 0.9 * grasp_shaping
    r += 0.8 * close_when_ready
    r -= 0.7 * premature_close
    r -= 0.4 * open_near_pen

    r += grasp_bonus

    r += 5.0 * lift_r
    r += lift_hold_bonus

    # Completion speed bonus
    if is_grasped and lift_prog >= 0.95:
        r += 1.2 * (1.0 - t)

    r -= hover_pen
    r -= low_side_pen
    r -= act_mag_pen
    r -= grip_cmd_pen

    return float(np.clip(r, -1.0, 10.0))