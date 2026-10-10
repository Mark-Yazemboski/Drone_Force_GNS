"""Provide the shared data preparation and cube geometry used throughout the
force GNS workflow. This module defines the cube dimensions and recorded-step
timing constants, builds surface nodes and their nearest-neighbor graph,
handles position/velocity unscaling and relative-wind conversion, and supplies
random-walk noise for normalization. Its trajectory loader reads saved data
into center-of-mass positions, rotation matrices, wind vectors, and available
physics metadata, using quaternion conversion to prepare the rigid states
consumed by training and rollout.
"""

import os
import torch
import numpy as np

#The half width of the block used in the making of the dataset
BLOCK_HALF_WIDTH = 0.0524
BLOCK_WIDTH = 2.0 * BLOCK_HALF_WIDTH

#THIS NEEDS TO BE CHANGED WHEN WE MOVE AWAY FROM MOJOCO
TIMESTEP = 0.0001348
SUBSTEPS = 50
DT_RECORD = TIMESTEP * SUBSTEPS

def relative_wind(wind_ms, v_curr):
    """wind_ms (m/s) broadcastable to v_curr; v_curr is the most-recent FD velocity (m/step).
       Returns (u, ||u||) in meters-per-recorded-step."""
    u = wind_ms * DT_RECORD - v_curr
    return u, torch.norm(u, dim=-1, keepdim=True)

#takes in the raw data and converts it back to SI units using the conversion described in the paper
def unscale_position_velocity(scaled_tensor):
    unscaled_tensor = scaled_tensor.clone()
    unscaled_tensor[..., :3] *= BLOCK_HALF_WIDTH     # position
    unscaled_tensor[..., 7:10] *= BLOCK_HALF_WIDTH   # velocity
    return unscaled_tensor


#Creates a cube mesh surface centered at the origin with specified side length and number of nodes per edge
def mesh_cube_surface(side_length, nodes_per_edge):


    L = side_length / 2.0
    lin = np.linspace(-L, L, nodes_per_edge)

    nodes = []

    # ±X faces
    for x in [-L, L]:
        for y in lin:
            for z in lin:
                nodes.append([x, y, z])

    # ±Y faces
    for y in [-L, L]:
        for x in lin:
            for z in lin:
                nodes.append([x, y, z])

    # ±Z faces
    for z in [-L, L]:
        for x in lin:
            for y in lin:
                nodes.append([x, y, z])

    #removes all of the duplicates points, and returns the unique set of nodes
    return np.unique(np.array(nodes), axis=0)

#Finds the k-nearest neighbors of each node to create an adjacency list
def knn_adjacency(nodes, k):
    N = nodes.shape[0]
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=2)
    edge_list = []
    for i in range(N):
        knn_idx = np.argsort(dist[i])[1:k+1]  # exclude self
        for j in knn_idx:
            edge_list.append([i, j])
    edge_index = np.array(edge_list).T  # shape (2, num_edges)
    return edge_index

#Adds random walk noise to a sequence of positions, returning the new positions and the noise applied
def add_random_walk_noise(positions, noise_scale):
    velocities = positions[1:] - positions[:-1]            # (T, N, 3)
    T, N, _ = velocities.shape

    # Generate all noise in one shot
    noise = torch.randn(T, N, 3, device=positions.device) * noise_scale

    # Add noise to velocities (vectorized)
    noisy_velocities = velocities + noise

    # Reconstruct positions: cumulative sum of noisy velocities + initial position
    new_positions = torch.empty_like(positions)
    new_positions[0] = positions[0]
    new_positions[1:] = positions[0:1] + torch.cumsum(noisy_velocities, dim=0)

    return new_positions, noise


# function takes a quaternion in wxyz format and converts it to a rotation matrix
def quat_wxyz_to_R(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],     dim=-1),
        torch.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],     dim=-1),
        torch.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)
    return R


# ======================================================================
# Dataset: keep the rigid state, not just node positions
# ======================================================================

def build_force_dataset(traj_range, trajectory_folder, weights_only, unscale_data,
                        verbose_every=200):
    """
    Loads raw trajectory files, keeping per-frame COM and rotation matrix (the
    state the force model integrates) plus the wind vector. Also pulls the
    replica_physics dict (gravity, mu, ...) out of the first file that has one,
    so training can use the SAME gravity the data was generated with - a wrong
    g would be silently absorbed into the learned normal forces otherwise.

    Returns (dataset, meta) where dataset is a list of dicts
    {com (T,3), R (T,3,3), wind (3,), T} and meta holds replica_physics if found.
    """
    dataset, meta = [], {}

    # Iterate over the specified trajectory range and load each trajectory file.
    for n, throw_number in enumerate(traj_range):
        path = os.path.join(trajectory_folder, f"{throw_number}.pt")
        raw = torch.load(path, weights_only=weights_only)
        states = raw[0].float()

        # Unscale the position and velocity data if requested.
        # Used to revert any scaling applied to the raw trajectory data.
        # Only used when the trajectories were the papers trajectories
        if unscale_data:
            states = unscale_position_velocity(states)

        # Extract the center of mass (COM) and quaternion from the state.
        com = states[:, 0:3].contiguous()
        quat = states[:, 3:7]
        # Convert the quaternion to a rotation matrix.
        R = quat_wxyz_to_R(quat)


        # Extract the wind vector from the raw data if available.
        wind = torch.zeros(3)
        if len(raw) > 1:
            try:
                wind = torch.as_tensor(raw[1], dtype=torch.float32).reshape(3)
            except Exception:
                pass


        # Extract the replica_physics dictionary from the raw data if available.
        if not meta and len(raw) > 3 and isinstance(raw[3], dict):
            rp = raw[3].get("replica_physics", None)
            if isinstance(rp, dict):
                meta = dict(rp)

        # Append the processed trajectory data to the dataset list.
        dataset.append({"com": com, "R": R, "wind": wind, "T": com.shape[0]})

        # Print progress if verbose mode is enabled.
        if verbose_every and (n + 1) % verbose_every == 0:
            print(f"  loaded {n + 1} trajectories...", flush=True)

    # Return the complete dataset and any extracted metadata.
    return dataset, meta
