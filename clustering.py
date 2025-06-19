# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
import os
import trimesh
import numpy as np
import torch
import tensorflow as tf

def perform_birch_clustering(input_dataframe, threshold=15, n_clusters=None, plot=True):
    """
    Perform BIRCH clustering on a DataFrame with user locations, calculate centroids for each cluster,
    and optionally generate a 2D scatter plot.

    Parameters:
    - input_dataframe (pd.DataFrame): The input DataFrame containing a 'location_users' column with user locations.
    - threshold (float): The threshold distance for BIRCH clustering.
    - n_clusters (int or None): The number of clusters to find. Use None for automatic determination.
    - plot (bool): Whether to generate a 2D scatter plot of users and centroids. Default is True.

    Returns:
    - pd.DataFrame: The input DataFrame with BIRCH clusters and centroid locations added.
    """

    # Ensure 'location_users' contains valid 3-element tuples or lists
    def validate_location(loc):
        if isinstance(loc, str):
            loc = eval(loc)
        if isinstance(loc, (list, tuple)) and len(loc) == 3:
            return loc
        return [0, 0, 0]

    input_dataframe['location_users'] = input_dataframe['location_users'].apply(validate_location)

    # Extract x, y, z coordinates
    locations = input_dataframe['location_users'].apply(pd.Series)
    locations.columns = ['x', 'y', 'z']

    # Apply BIRCH clustering on x, y
    birch = Birch(threshold=threshold, n_clusters=n_clusters)
    locations['cluster_birch'] = birch.fit_predict(locations[['x', 'y']])

    # Build clustered DataFrame
    clustered_dataframe = input_dataframe.copy()
    clustered_dataframe['cluster_birch'] = locations['cluster_birch']
    clustered_dataframe = clustered_dataframe.drop(columns=['cluster', 'centroid_location'], errors='ignore')

    # Calculate centroids
    centroids = (
        locations.groupby('cluster_birch')[['x', 'y']]
        .mean()
        .reset_index()
        .rename(columns={'x': 'centroid_x', 'y': 'centroid_y'})
    )

    # Map centroids back
    centroid_map = centroids.set_index('cluster_birch').to_dict('index')
    clustered_dataframe['centroid_location'] = clustered_dataframe['cluster_birch'].map(
        lambda c: (centroid_map[c]['centroid_x'], centroid_map[c]['centroid_y'])
    )

    # Report cluster count
    num_clusters = clustered_dataframe['cluster_birch'].nunique()
    print(f"Number of clusters formed by BIRCH: {num_clusters}")

    # Plot if requested
    if plot:
        plt.figure(figsize=(14, 10))
        # Plot users
        plt.scatter(locations['x'], locations['y'], c='red', s=20, label='Users')
        # Plot centroids
        plt.scatter(centroids['centroid_x'], centroids['centroid_y'], c='black', s=50, marker='X', label='Centroids')
        plt.legend(loc='upper left', fontsize=20)
        plt.xlabel('X Coordinate', fontsize=22)
        plt.ylabel('Y Coordinate', fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(False)
        plt.show()

    return clustered_dataframe


def extract_centroid_tiles(outdoor_users_numpy,
                           chosen_poor_coverage_Dataframe_BIRCH,
                           rx_dB):
    """
    Computing the 2D distance between each centroid point and the center location of 
    outdoor tiles to find the related tiles that centroids can be assigned to.
    """
    # Step 1: Convert the PyTorch tensor to TensorFlow
    outdoor_users_tensorflow = tf.convert_to_tensor(outdoor_users_numpy)

    # Step 2: Extract x, y coordinates
    centroids = chosen_poor_coverage_Dataframe_BIRCH.groupby('cluster_birch')['centroid_location'].first()
    centroids_array = np.array([list(loc) for loc in centroids])
    centroids_xy = tf.convert_to_tensor(centroids_array, dtype=tf.float32)
    outdoor_users_xy = outdoor_users_tensorflow[0,0,0, :, :2]
    outdoor_users_xy = tf.expand_dims(outdoor_users_xy, axis=0)

    # Step 3: Expand for broadcasting
    centroids_expanded = tf.expand_dims(tf.expand_dims(centroids_xy, axis=1), axis=0)
    outdoor_users_expanded = tf.expand_dims(outdoor_users_xy, axis=0)
    outdoor_users_expanded = tf.tile(
        outdoor_users_expanded,
        [1, centroids_xy.shape[0], 1, 1]
    )

    # Step 4: Compute 2D Euclidean distance
    squared_diff = tf.square(centroids_expanded - outdoor_users_expanded)
    squared_distance = tf.reduce_sum(squared_diff, axis=-1)
    distance_2d = tf.sqrt(squared_distance)

    # Step 5: Four nearest tiles via negative distance Top-K
    neg_distance_2d = -distance_2d
    values, indices = tf.math.top_k(neg_distance_2d, k=4)
    nearest_tile_indices = indices[0, :, 0]
    second_min_indices   = indices[0, :, 1]
    third_min_indices    = indices[0, :, 2]
    fourth_min_indices   = indices[0, :, 3]

    # Average users for gathering
    outdoor_users_result = tf.reduce_mean(outdoor_users_tensorflow, axis=[0, 1])

    # Gather tile locations
    nearest_tile_locations = tf.gather(outdoor_users_result[0], nearest_tile_indices)
    second_min_locations   = tf.gather(outdoor_users_result[0], second_min_indices)
    third_min_locations    = tf.gather(outdoor_users_result[0], third_min_indices)
    fourth_min_locations   = tf.gather(outdoor_users_result[0], fourth_min_indices)

    # Step 6: Pathgain extraction for each nearest tile
    batch_size, num_precoder_rows, num_precoder_cols, _, _ = rx_dB.shape

    def gather_pathgain(indices_tensor):
        idx_torch = torch.tensor(indices_tensor.numpy(), dtype=torch.long)
        idx_exp = idx_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        idx_exp = idx_exp.repeat(batch_size, num_precoder_rows, num_precoder_cols, 1)
        return torch.gather(rx_dB, 3, idx_exp.unsqueeze(-1)).squeeze(-1)

    pathgain_nearest_tiles  = gather_pathgain(nearest_tile_indices)
    pathgain_second_tiles   = gather_pathgain(second_min_indices)
    pathgain_third_tiles    = gather_pathgain(third_min_indices)
    pathgain_fourth_tiles   = gather_pathgain(fourth_min_indices)

    return (
        nearest_tile_indices, nearest_tile_locations, pathgain_nearest_tiles,
        second_min_indices, second_min_locations, pathgain_second_tiles,
        third_min_indices, third_min_locations, pathgain_third_tiles,
        fourth_min_indices, fourth_min_locations, pathgain_fourth_tiles
    )








