import numpy as np
import math

def reward_fn(obs, action, info) -> float:
    # --- Helpers / inputs ---
    dist = float(info.get("distance", np.linalg.norm(info["gripper_pos"] - info["object_pos"])))
    gpos = np.asarray(info.get("gripper_pos", np.zeros(3)), dtype=np.float64)
    opos = np.asarray(info.get("object_pos", np.zeros(3)), dtype=np.float64)
    obj_h = float(info.get("object_height", opos[2]))
    is_grasped = bool(info.get("is_grasped", False))
    gopen = float(info.get("gripper_open", 1.0))  # 0 closed, 1 open
    a = np.asarray(action, dtype=np.float64).reshape(-1)

    # Table height estimate (cube is 2.5cm, so center at table+0.0125 when resting)
    table_z = float(info.get("table_z", obj_h - 0.0125))
    lift_height = max(0.0, obj_h - (table_z + 0.0125))  # how much above resting center

    # --- Stage shaping terms ---
    # Reach: encourage reducing distance smoothly
    reach = math.exp(-6.0 * dist)  # in (0,1]

    # Align: encourage gripper to be above object and laterally centered
    lateral = float(np.linalg.norm((gpos - opos)[:2]))
    dz = float(gpos[2] - opos[2])
    above = math.exp(-20.0 * max(0.0, -dz))          # penalize being below object
    lateral_align = math.exp(-25.0 * lateral)        # tight XY alignment
    z_align = math.exp(-15.0 * abs(dz - 0.02))       # prefer gripper slightly above object (~2cm)
    align = lateral_align * above * z_align          # in (0,1]

    # Grasp: reward closing only when close; penalize premature closing and hovering-open at contact
    close_cmd = float(np.clip(-a[-1], 0.0, 1.0))     # negative action means close
    open_cmd = float(np.clip(a[-1], 0.0, 1.0))

    near = math.exp(-40.0 * dist)                    # strong gating for very close
    very_near = 1.0 if dist < 0.03 else 0.0

    # Encourage being open while approaching (avoid scraping/dragging), then closing at contact
    approach_open_bonus = (1.0 - near) * (gopen) * 0.15
    close_when_near_bonus = near * close_cmd * 1.25
    premature_close_pen = (1.0 - near) * close_cmd * 0.35
    open_when_very_near_pen = very_near * open_cmd * 0.35

    # Lift: only meaningful if grasped; otherwise small incentive to be above (but capped)
    lift_target = 0.10  # meters above resting center (~10cm lift)
    lift_progress = np.clip(lift_height / lift_target, 0.0, 1.0)
    lift_reward = (2.5 * lift_progress) * (1.0 if is_grasped else 0.0)

    # Success bonus for clear lift
    success_bonus = 4.0 if (is_grasped and lift_height > 0.08) else 0.0

    # Anti-hacking: hovering above without grasp gives limited reward, and penalize lingering near-open
    hover_reward = 0.15 * near * (0.5 + 0.5 * align) * (0.0 if is_grasped else 1.0)
    fake_grasp_pen = 0.4 * (near * (1.0 - gopen)) * (0.0 if is_grasped else 1.0)  # closed near object but not lifted

    # --- Smoothness / action penalties ---
    # Penalize large actions and jerky gripper commands
    act_pen = 0.05 * float(np.sum(np.square(a[:-1]))) + 0.02 * float(a[-1] ** 2)

    # Optional jerk penalty if previous action provided
    jerk_pen = 0.0
    prev_a = info.get("prev_action", None)
    if prev_a is not None:
        prev_a = np.asarray(prev_a, dtype=np.float64).reshape(-1)
        if prev_a.shape == a.shape:
            jerk_pen = 0.02 * float(np.sum(np.square(a - prev_a)))

    # --- Compose reward ---
    # Stage weights: reach -> align -> grasp -> lift
    r = 0.0
    r += 1.2 * reach
    r += 1.0 * align * (0.5 + 0.5 * reach)  # align matters more when close
    r += approach_open_bonus
    r += close_when_near_bonus
    r -= premature_close_pen
    r -= open_when_very_near_pen
    r += hover_reward
    r -= fake_grasp_pen
    r += lift_reward
    r += success_bonus

    r -= act_pen
    r -= jerk_pen

    # Keep within a reasonable range
    r = float(np.clip(r, -1.0, 10.0))
    return r