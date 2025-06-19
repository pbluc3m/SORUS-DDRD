# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import trimesh
import numpy as np
import torch
################# lat/long to Cartesian #######################################

def lat_lon_to_cartesian(lat, lon, lat_mid=51.5140, lon_mid=-0.1350):
    """
    Convert latitude and longitude to Cartesian coordinates.
    The origin (0,0) is at the midpoint of the specified bounds.
    
    Parameters:
    - lat: Latitude of the point to convert.
    - lon: Longitude of the point to convert.
    - lat_mid: Latitude of the midpoint (default is the midpoint of the specified bounds).
    - lon_mid: Longitude of the midpoint (default is the midpoint of the specified bounds).
    
    Returns:
    - A tuple (x, y) representing the Cartesian coordinates relative to the midpoint.
    """
    # Conversion factors: meters per degree
    lat_to_meters = 111000  # for latitude
    lon_to_meters = 69000   # for longitude at London's latitude
    
    # Calculate the difference from the midpoint
    delta_lat = lat - lat_mid
    delta_lon = lon - lon_mid
    
    # Convert to Cartesian coordinates (meters)
    x = delta_lon * lon_to_meters
    y = delta_lat * lat_to_meters
    
    return (x, y)


def construct_bs_dict(x_coords,
                      y_coords,
                      BS_height,
                      transmitter_power,
                      bearing_deg,
                      downtilt,
                      Beam_width_horizontal,
                      latitudes,
                      longitudes):
    """
    Construct a list of base-station parameter dictionaries.
    """
    

    unique_coords = {}
    siteID_counter = 0
    baseStations = []

    for (x, y, z, power, bearing, elevation, beam_width_horizontal, lat, lon) in zip(
            x_coords,
            y_coords,
            BS_height,
            transmitter_power,
            bearing_deg,
            downtilt,
            Beam_width_horizontal,
            latitudes,
            longitudes):

        coord_pair = (lat, lon)
        if coord_pair not in unique_coords:
            unique_coords[coord_pair] = siteID_counter
            siteID_counter += 1
        siteID = unique_coords[coord_pair]

        bs_params = {
            'x': x,
            'y': y,
            'z': z,
            'bearing_deg': bearing,
            'elevation_deg': elevation + 90,
            'siteID': siteID,
            'Tx_power': power,
            'max_antenna_gain_dB': 14.0,
            'horizontal_beamwidth_deg': beam_width_horizontal,
            'vertical_beamwidth_deg': 10.0,
            'antennaRatioFrontBack_dB': 30.0,
            'SLAv_dB': 30.0,
            'users_associated': [],
            'Precoder_associated': []
        }
        baseStations.append(bs_params)

    return baseStations


def construct_user_dict(Channel_large_RT_full):
    """
    Construct a list of user-parameter dictionaries from ray-tracing results.
    """
    

    # Extract unique user coordinates from first BS and first precoder
    unique_users = Channel_large_RT_full[0, 0, 0, :, :2].numpy()
    user_params_list = []

    for coords in unique_users:
        ue_params = {
            'x': coords[0],
            'y': coords[1],
            'z': 1.5,
            'serverID': -1,
            'Precoder_index': None,
            'RSS_dBm': -1,
            'SINR_dB': -100
        }
        user_params_list.append(ue_params)

    return user_params_list

def associate_users(rx_dB, baseStations, user_params_list):
    """
    Update user and BS dictionaries with association based on received power.

    Parameters:
    - rx_dB: torch.Tensor of shape [num_bs, num_rows, num_cols, num_users, 1]
    - baseStations: list of BS parameter dicts with 'users_associated' and 'Precoder_associated'
    - user_params_list: list of user parameter dicts

    Returns:
    - baseStations: updated list with associations
    - user_params_list: updated user list with RSS, serverID, and Precoder_index
    """
    

    # 1) Flatten the first three dimensions into one
    flat_rx_dB = rx_dB.view(-1, rx_dB.shape[-2], rx_dB.shape[-1])  # [(BS*rows*cols), num_users, 1]

    # 2) Find max RSS and flat indices
    bestRSSs_per_user, flat_indices = torch.max(flat_rx_dB, dim=0)  # [num_users, 1]
    flat_indices = flat_indices.squeeze(-1)
    bestRSSs_per_user = bestRSSs_per_user.squeeze(-1)

    bs_dim  = rx_dB.shape[0]
    row_dim = rx_dB.shape[1]
    col_dim = rx_dB.shape[2]

    # 3) Map flat index to BS, row, col
    bestServerID    = flat_indices // (row_dim * col_dim)
    bestRowPrecoder = (flat_indices % (row_dim * col_dim)) // col_dim
    bestColPrecoder = flat_indices % col_dim

    # 4) Update users
    for i, user in enumerate(user_params_list):
        user['RSS_dBm']       = bestRSSs_per_user[i].item()
        user['serverID']      = bestServerID[i].item()
        user['Precoder_index']= (
            int(bestRowPrecoder[i].item()),
            int(bestColPrecoder[i].item())
        )

    # 5) Update baseStations
    for user_index, (bs_index, row_index, col_index) in enumerate(
            zip(bestServerID, bestRowPrecoder, bestColPrecoder)):
        b  = int(bs_index.item())
        r  = int(row_index.item())
        c  = int(col_index.item())
        baseStations[b]['users_associated'].append(user_index)
        baseStations[b]['Precoder_associated'].append((r, c, user_index))

    return baseStations, user_params_list, bestRSSs_per_user

def create_user_params_list(users_location):
    """
    Create a list of user parameters (centroids) based on nearest tile locations.

    Parameters:
    - users_location (Tensor): Tensor of nearest tile locations with shape [..., 3].

    Returns:
    - list: A list of dictionaries containing user parameters.
    """
    # Extract x and y coordinates
    x_coords_centroids = users_location[..., 0]  
    y_coords_centroids = users_location[..., 1]  

    # Extract the unique users by considering only the x and y coordinates
    unique_centroids = users_location[:, :2].numpy()  # Shape: [num_users, 2]

    # Create the user parameters list
    users_params_list = []  # List to store user parameters

    for coords in unique_centroids:
        ue_params = {
            'x': coords[0],          # Set the x-coordinate
            'y': coords[1],          # Set the y-coordinate
            'z': 1.5,                # Set the z-coordinate (assuming z-coordinate is always 1.5m)
            'serverID': -1,          # Serving base station
            'Precoder_index': None,  # Initialize as None, to be updated later with (row_dim, col_dim)
            'RSS_dBm': -1,           # RSS from serving base station
            'SINR_dB': -100,         # SINR
        }
        users_params_list.append(ue_params)

    return users_params_list

def association_centroid(pathgain, users_params_list, baseStations):
    """
    Update user parameters and associate users with base stations based on RSS, server IDs, and precoder indices.

    Parameters:
    - pathgain (Tensor): Pathgain tensor with shape [num_base_stations, num_precoder_rows, num_precoder_cols, num_centroids].
    - users_params_list (list): List of user parameters (centroids).
    - baseStations (list): List of base station dictionaries.

    Returns:
    - users_params_list (list): Updated list of user parameters.
    - baseStations (list): Updated list of base station dictionaries with user and precoder associations.
    """
    # Step 1: Flatten the first three dimensions into one
    flat_pathgain = pathgain.reshape(-1, pathgain.shape[-1])  # Shape: [(num_base_stations * num_precoder_rows * num_precoder_cols), num_centroids]

    # Step 2: Find the maximum RSS values and flat indices across the flattened dimension
    bestRSSs_per_centroids, flat_indices = torch.max(flat_pathgain, dim=0)  # Shape: [num_centroids]

    # Step 3: Convert flat indices back to (Base Station, Row, Column)
    bs_dim = pathgain.shape[0]  # Base Stations
    row_dim = pathgain.shape[1]  # Precoder Rows
    col_dim = pathgain.shape[2]  # Precoder Columns

    bestServerID = flat_indices // (row_dim * col_dim)  # Base Station index
    bestRowPrecoder = (flat_indices % (row_dim * col_dim)) // col_dim  # Row precoder index
    bestColPrecoder = flat_indices % col_dim  # Column precoder index

    # Step 4: Update users_params_list with the best RSS, Base Station ID, and Precoder Index
    for i, user in enumerate(users_params_list):
        user['RSS_dBm'] = bestRSSs_per_centroids[i].item()
        user['serverID'] = bestServerID[i].item()
        user['Precoder_index'] = (int(bestRowPrecoder[i].item()), int(bestColPrecoder[i].item()))  # Update Precoder_index

    # Step 5: Associate users with Base Stations and their precoders
    for user_index, (bs_index, row_index, col_index) in enumerate(zip(bestServerID, bestRowPrecoder, bestColPrecoder)):
        bs_index_scalar = int(bs_index.item())
        row_index_scalar = int(row_index.item())
        col_index_scalar = int(col_index.item())

        # Append user index to users_associated list
        baseStations[bs_index_scalar]['users_associated'].append(user_index)

        # Append precoder information (row, column, user_index) to Precoder_associated list
        baseStations[bs_index_scalar]['Precoder_associated'].append((row_index_scalar, col_index_scalar, user_index))

    return users_params_list, baseStations



