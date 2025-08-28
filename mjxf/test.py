import mujoco
import mujoco.viewer

CTL_TARGET = [
    [0.6, -1.5], # initial position
    [0.31, -1.5], # wp 1, highest point after lifting
    [0.31, 0.015], # wp 2, endpoint after loading
    [0.4, 0.015], # wp 3, endpoint after unloading
]

DURATION = [
    1.0,
    3.0,
    1.0,
]

TIME_STEP = 60/1000 # (s) Update 60 Hz

def traj_gen(ctl_target: list, duration: list, time_step: float):
    # Generate a smooth trajectory using cubic splines, where duration is the time to reach each waypoint
    if len(duration) != len(ctl_target) - 1:
        raise ValueError
    pass

mjcf_filepath = '00-elevator.mjcf'

model = mujoco.MjModel.from_xml_path(mjcf_filepath)

data = mujoco.MjData(model)

ctrl_joint_names = [
    'piston_motor',
    'rotate_motor'
]
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Simulation launched. Press Ctrl+C in the terminal to exit.")

    # Apply a constant control signal to the piston to create movement
    if model.nu > 0:  # Check if there are any actuators
        piston_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "piston_motor")
        data.ctrl[piston_motor_id] = 0.5
        rotate_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotate_motor")
        data.ctrl[rotate_motor_id] = -1.5

    # Run the simulation loop
    while viewer.is_running():
        # Advance the simulation by one step
        mujoco.mj_step(model, data)
        
        

        viewer.sync()
