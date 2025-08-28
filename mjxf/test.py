import mujoco
import mujoco.viewer

import numpy as np
from scipy.interpolate import CubicSpline

CTL_TARGET = [
    [1.0, -1.5],
    [0.6, -1.5],  # initial position
    [0.30, -1.5], # wp 1, highest point after lifting
    [0.30, 0.015],# wp 2, endpoint after loading
    [0.4, 0.015], # wp 3, endpoint after unloading
]

# DEBUG ONLY
SPD_UP = 1
DURATION = [
    3.0/SPD_UP,
    3.0/SPD_UP,
    5.0/SPD_UP,
    3.0/SPD_UP,
]

TIME_STEP = 60/1000 # (s) Update 60 Hz

def traj_gen(ctl_target: list, duration: list, time_step: float):
    """
    Generate a smooth trajectory using cubic splines, where duration is the time
    to reach each waypoint.
    """
    if len(duration) != len(ctl_target) - 1:
        raise ValueError("Duration list must be one element shorter than the ctl_target list.")

    waypoints = np.array(ctl_target)
    time_pts = np.cumsum([0] + duration)

    # --- FIX ---
    # The transpose (.T) is removed.
    # `waypoints` is shape (4, 2) and `time_pts` is length 4. The dimensions now match.
    spline = CubicSpline(time_pts, waypoints, bc_type='clamped')

    total_duration = time_pts[-1]
    time_query_pts = np.arange(0, total_duration, time_step)

    # Ensure the final time point is included in the query.
    if not np.isclose(time_query_pts[-1], total_duration):
        time_query_pts = np.append(time_query_pts, total_duration)

    trajectory_coords = spline(time_query_pts)

    return trajectory_coords.tolist()


mjcf_filepath = '00-elevator.mjcf'

model = mujoco.MjModel.from_xml_path(mjcf_filepath)

TIME_STEP = model.opt.timestep
traj = traj_gen(CTL_TARGET, DURATION, TIME_STEP)
data = mujoco.MjData(model)

ctrl_joint_names = [
    'piston_motor',
    'rotate_motor'
]
piston_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "piston_motor")
rotate_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotate_motor")
support_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "support_motor")

data.ctrl[support_motor_id] = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Simulation launched. Press Ctrl+C in the terminal to exit.")

    # Apply a constant control signal to the piston to create movement
    data.ctrl[piston_motor_id] = CTL_TARGET[0][0]
    data.ctrl[rotate_motor_id] = CTL_TARGET[0][1]
    
    # Run the simulation loop
    i = 0;
    while viewer.is_running():
        # Advance the simulation by one step
        mujoco.mj_step(model, data)
        
        i += 1
        if i < len(traj):
            data.ctrl[piston_motor_id], data.ctrl[rotate_motor_id] = traj[i]
            data.ctrl[support_motor_id] = -(1-data.ctrl[piston_motor_id])
            print(f"Step {i}: Piston Control = {data.ctrl[piston_motor_id]}, Rotate Control = {data.ctrl[rotate_motor_id]}")
        import time
        time.sleep(TIME_STEP)
        
        # Don't play unntil it stablize XD
        # if i > DURATION[0] / TIME_STEP:
        viewer.sync()
