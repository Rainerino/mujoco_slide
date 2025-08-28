import mujoco
import mujoco.viewer

mjcf_filepath = '00-elevator.mjcf'

model = mujoco.MjModel.from_xml_path(mjcf_filepath)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Simulation launched. Press Ctrl+C in the terminal to exit.")

    # Apply a constant control signal to the piston to create movement
    if model.nu > 0:  # Check if there are any actuators
        data.ctrl[0] = 0.5

    # Run the simulation loop
    while viewer.is_running():
        # Advance the simulation by one step
        mujoco.mj_step(model, data)

        viewer.sync()
