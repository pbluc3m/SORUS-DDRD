# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import tensorflow as tf
import torch
import os
import trimesh
import numpy as np
import tensorflow as tf
import torch

def extract_outdoor_loc_gain(coverage_location_list,
                             gain_dB,
                             min_UE_x_loc,
                             max_UE_x_loc,
                             min_UE_y_loc,
                             max_UE_y_loc):
    # Infer tile dimensions from gain_dB
    num_rows = gain_dB.shape[1]
    num_cols = gain_dB.shape[2]

    cell_centers = tf.stack(coverage_location_list)

    # Number of base stations
    num_basestation = gain_dB.shape[0]

    # Expand gain_dB to include a last dimension for concatenation
    expanded_gain_dB = tf.expand_dims(gain_dB, axis=-1)

    # Expand dimensions of cell_centers to match the desired shape
    expanded_cell_centers = tf.expand_dims(cell_centers, axis=1)
    expanded_cell_centers = tf.expand_dims(expanded_cell_centers, axis=2)

    # Tile to shape [num_basestation, num_rows, num_cols, ..., 3]
    expanded_cell_centers = tf.tile(
        expanded_cell_centers,
        [1, num_rows, num_cols, 1, 1, 1]
    )
    final_tensor = tf.concat([expanded_cell_centers, expanded_gain_dB], axis=-1)


    Coverage_location = final_tensor.numpy()
    Coverage_location_torch = torch.from_numpy(Coverage_location)

    x_coords_sionna = Coverage_location_torch[0, ..., 0]
    y_coords_sionna = Coverage_location_torch[0, ..., 1]

    # Use the input bounds instead of hard-coded numbers
    mask = (
        (x_coords_sionna >  min_UE_x_loc) &
        (x_coords_sionna <  max_UE_x_loc) &
        (y_coords_sionna >  min_UE_y_loc) &
        (y_coords_sionna <  max_UE_y_loc)
    )

    mask = mask.unsqueeze(0).expand(
        Coverage_location_torch.shape[0], -1, -1, -1, -1
    )

    filtered_locations_and_gains = Coverage_location_torch[mask].view(
        Coverage_location_torch.shape[0],
        Coverage_location_torch.shape[1],
        Coverage_location_torch.shape[2],
        -1,
        Coverage_location_torch.shape[-1]
    )

    pathgain_center_filterred_dB = filtered_locations_and_gains[..., 3].unsqueeze(-1)
    filtered_locations = filtered_locations_and_gains[..., 0:3]

    inf_mask = torch.isinf(pathgain_center_filterred_dB)
    user_invalid = inf_mask.squeeze(-1).all(dim=0)
    user_valid   = ~user_invalid

    user_valid_expanded = user_valid.unsqueeze(0).expand(
        pathgain_center_filterred_dB.shape[0], -1, -1, -1
    )

    pathgain_center_filterred_dB_outdoor = pathgain_center_filterred_dB[user_valid_expanded].view(
        pathgain_center_filterred_dB.shape[0],
        pathgain_center_filterred_dB.shape[1],
        pathgain_center_filterred_dB.shape[2],
        -1,
        pathgain_center_filterred_dB.shape[-1]
    )

    user_valid_expanded_locations = user_valid.unsqueeze(0).unsqueeze(-1).expand(
        filtered_locations.shape[0], -1, -1, -1, filtered_locations.shape[-1]
    )

    filtered_locations_outdoor = filtered_locations[user_valid_expanded_locations].view(
        filtered_locations.shape[0],
        filtered_locations.shape[1],
        filtered_locations.shape[2],
        -1,
        filtered_locations.shape[-1]
    )

    outdoor_loc_gain = tf.concat(
        [filtered_locations_outdoor, pathgain_center_filterred_dB_outdoor],
        axis=-1
    )

    return outdoor_loc_gain


def compute_signed_distance(points, mesh):
    distances = mesh.nearest.signed_distance(points)
    return distances


def check_points_in_buildings(points, ply_folder, threshold=0.0, verbose=True):
    num_points = points.shape[0]
    inside = np.zeros(num_points, dtype=bool)

    for ply_file in os.listdir(ply_folder):
        if ply_file.endswith('.ply') and ply_file != 'Plane.ply':
            mesh = trimesh.load_mesh(os.path.join(ply_folder, ply_file))
            if verbose:
                print(f"Checking building: {ply_file}")
            signed_distances = compute_signed_distance(points, mesh)
            inside |= (signed_distances > threshold)

    return inside


def get_outdoor_users(all_data_tensor, ply_folder, threshold=0.0, verbose=True):
    user_positions = all_data_tensor[0, 0, 0, :, :3].cpu().numpy()
    inside_mask = check_points_in_buildings(user_positions, ply_folder, threshold, verbose)
    outside_mask = ~inside_mask

    inside_outside_tensor = torch.from_numpy(inside_mask).unsqueeze(-1).bool()
    outside_mask_tensor = torch.from_numpy(outside_mask).bool().to(all_data_tensor.device)

    outdoor_data_tensor = all_data_tensor[:, :, :, outside_mask_tensor, :]
    return outdoor_data_tensor, inside_outside_tensor
