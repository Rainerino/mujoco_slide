import mujoco
import mujoco.viewer
import time
import os

# --- Configuration ---
# ⬇️ IMPORTANT: Change this to the path of your MJCF file.
mjcf_filepath = '00-elevator.mjcf'
# ---------------------


# --- Main Script ---

# 1. Check if the MJCF file exists before trying to load it
if not os.path.exists(mjcf_filepath):
    print(f"Error: MJCF file not found at '{mjcf_filepath}'")
    print("Please update the 'mjcf_filepath' variable in this script.")
    exit()

# 2. Load the model from the specified MJCF file
#    Note: The function is called 'from_xml_path', but it's the correct one for .mjcf files.
print(f"Loading model from: {mjcf_filepath}")
try:
    model = mujoco.MjModel.from_xml_path(mjcf_filepath)
except Exception as e:
    print(f"Error loading model: {e}")
    print("\n⚠️  Please ensure the MJCF is valid and that any mesh files (.STL) are located")
    print("   in the directory specified by the <compiler meshdir=.../> tag in your file.")
    exit()

# 3. Create the data object that holds the simulation state
data = mujoco.MjData(model)

# 4. Get the numerical ID of the site we want to track
try:
    site_id = model.site('top_socket').id
except KeyError:
    print("Error: Site 'top_socket' not found in the model.")
    exit()

# 5. Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Simulation launched. Press Ctrl+C in the terminal to exit.")

    # Apply a constant control signal to the piston to create movement
    if model.nu > 0:  # Check if there are any actuators
        data.ctrl[0] = 0.5

    # Run the simulation loop
    while viewer.is_running():
        # Advance the simulation by one step
        mujoco.mj_step(model, data)

        # Get the current world coordinates (x, y, z) of the site
        site_position = data.site_xpos[site_id]

        # Print the position to the console, overwriting the previous line
        print(f"Coordinates of 'top_socket': X={site_position[0]:.4f} Y={site_position[1]:.4f} Z={site_position[2]:.4f}\n", end='\r')

        # Synchronize the viewer to render the new state
        viewer.sync()

print("\nSimulation finished.")
