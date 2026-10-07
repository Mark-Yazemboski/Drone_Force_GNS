"""Experiment settings for the drone force model (two-stage training).
Generate data first:  python mujoco_drone_generator.py --out data/mj_drone --n 300
Then:                 python run_drone.py
"""

import json
import os

from drone_config import DroneConfig
from drone_data import build_drone_dataset
from train_drone_gns import DroneTrainSettings, train_drone_force_model

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------- data ----------------
trajectory_folder = os.path.join(script_dir, "data/mj_drone")
train_range = range(0, 75)
val_range = range(75, 95)
test_range = range(95, 100)
rotor_speed_in_rpm = False          # True if hardware logs are in RPM rather than rad/s
rotor_speed_hold = None             # None -> read from each file's meta ("aligned" for MuJoCo)

# ---------------- drone ----------------
# For MuJoCo data, use the config the generator wrote (mass, inertia, COM read
# from the compiled model); the loader refuses data that does not match.
config_path = os.path.join(trajectory_folder, "drone_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        drone = DroneConfig(**json.load(f))
else:
    drone = DroneConfig()

# ---------------- training ----------------
settings = DroneTrainSettings(
    # ---- model ----
    h=2,                            # velocity history: h+1 input frames
    # each GNN: MLP width, Linear layers per MLP, message-passing steps (own weights),
    # and repeats of those steps (same weights). Total rounds = steps x repeats.
    aero_latent_dim=128, aero_mlp_layers=2, aero_msg_passing_steps=4, aero_msg_passing_repeats=1,

    contact_latent_dim=128, contact_mlp_layers=2, contact_msg_passing_steps=3, contact_msg_passing_repeats=1,

    contact_d0=0.006, contact_tau=0.0015,   # gate: pad sphere radius 4 mm + margin

    # ---- physical coefficients (learned in stage 1, frozen in stage 2) ----
    learn_thrust_coeffs=True,       # k_f, k_m, held near thrust-stand values by w_prior

    k_f_scale=1.0, k_m_scale=1.0,   # k_f, k_m themselves come from drone_config.json (sim) 
                                    # This will just scale them initially. The model still tries to fit k_f and k_m


    learn_drag_coeffs=True,         # anchor-law coefficients below
    k_rot_init=5e-5,                # rotor drag  k_rot * w_j * u_perp   (per rotor)
    k_body_init=0.02,               # body drag   k_body * |u| u
    k_rod_init=0.05,                # rod drag    k_rod * |u_perp| u_perp
    k_pad_init=0.05,                # pad drag    k_pad * |u| u

    # ---- windows / batches ----
    multistep=4, batch_size=256,
    aero_min_pad_dist=0.026,        # stage-1 windows: pad stays > 2.6 cm from the wall throughout
    noise_frac=0.2,                 # input noise = 0.2 x each stage's median residual
    rotate_aug=True,

    # ---- stage 1: aero (contact-free windows) ----
    aero_epochs=200, aero_lr=3e-4,
    w_aero_anchor=0.1,              # each aero node toward its drag law
    w_aero_smooth=0.01,             # aero force smooth in time
    w_axial=0.1,                    # rotor axial thrust correction toward zero
    w_prior=1e-3,                   # k_f, k_m toward thrust-stand values

    # ---- stage 2: contact (all windows, aero frozen) ----
    contact_epochs=300, contact_lr=1e-4,
    w_fric_dir=1.0, w_fric_mag=1.0, w_fric_cone=1.0,
    mu_init=0.3, learn_mu=True,

    # ---- validation ----
    val_horizon=20,                 # steps rolled forward per window (match the MPC horizon)
    val_stride=5,                   # start a validation window every 5 frames
    val_interval=10,                # validate every 10 epochs
)

run_name = "drone_two_stage"
save_model_path = os.path.join(script_dir, "models", run_name, run_name + ".pt")

if __name__ == "__main__":
    train, meta = build_drone_dataset(train_range, trajectory_folder, drone, rotor_speed_in_rpm, rotor_speed_hold)
    val, _ = build_drone_dataset(val_range, trajectory_folder, drone, rotor_speed_in_rpm, rotor_speed_hold)
    print(f"{len(train)} train / {len(val)} val trajectories")
    # To rerun only stage 2 from a saved stage-1 model:
    #   train_drone_force_model(drone, settings, train, val, save_model_path, stages=("contact",),
    #                           init_checkpoint=save_model_path.replace(".pt", "_aero_best.pt"))
    train_drone_force_model(drone, settings, train, val, save_model_path)
