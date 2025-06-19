# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import torch
import tensorflow as tf
import numpy as np
import re
import ast
import pandas as pd

def choose_poor_coverage(baseStations,
                         user_params_list,
                         min_rss=-130,
                         max_rss=-100):
    """
    Select users with RSS_dBm in the specified range for each base station.

    Parameters:
    - baseStations: list of BS parameter dicts containing 'users_associated'
    - user_params_list: list of user parameter dicts containing 'RSS_dBm', 'serverID', etc.
    - min_rss: minimum RSS threshold (inclusive)
    - max_rss: maximum RSS threshold (inclusive)

    Returns:
    - Chosen_poor_coverage: list of dicts with selected poor-coverage user info
    """

    Chosen_poor_coverage = []

    for bs in baseStations:
        # Users associated to this BS
        associated_users = bs['users_associated']
        # Gather (index, RSS) for each associated user
        rss_values = [(idx, user_params_list[idx]['RSS_dBm']) for idx in associated_users]
        # Filter by RSS range
        filtered = [u for u in rss_values if min_rss <= u[1] <= max_rss]
        if not filtered:
            continue
        # Choose all filtered users
        chosen = filtered[:]
        # Append info
        for user_index, rss_value in chosen:
            info = user_params_list[user_index]
            Chosen_poor_coverage.append({
                'location_BS': (bs['x'], bs['y'], bs['z']),
                'serverID_users': info['serverID'],
                'location_users': (info['x'], info['y'], info['z']),
                'siteID_BS': bs['siteID'],
                'user_index': user_index,
                'RSS_dBm': rss_value,
                'Precoder_index': info['Precoder_index'],
            })

    return Chosen_poor_coverage


def combine_near_tiles(
    nearest_tile_indices,
    second_min_indices,
    third_min_indices,
    fourth_min_indices,
    nearest_tile_locations,
    second_min_locations,
    third_min_locations,
    fourth_min_locations,
    pathgain_nearest_tiles,
    pathgain_second_tiles,
    pathgain_third_tiles,
    pathgain_fourth_tiles
):
    # Indices
    combined_near_tiles_indices = tf.concat([
        nearest_tile_indices,
        second_min_indices,
        third_min_indices,
        fourth_min_indices
    ], axis=0)
    combined_near_tiles_indices = torch.tensor(
        combined_near_tiles_indices.numpy(), dtype=torch.long
    )

    # Locations
    combined_near_tiles_locations = tf.concat([
        nearest_tile_locations,
        second_min_locations,
        third_min_locations,
        fourth_min_locations
    ], axis=0)
    combined_near_tiles_locations = torch.tensor(
        combined_near_tiles_locations.numpy(), dtype=torch.float32
    )

    # Pathgains
    combined_near_tiles_pathgains = torch.cat([
        pathgain_nearest_tiles,
        pathgain_second_tiles,
        pathgain_third_tiles,
        pathgain_fourth_tiles
    ], dim=3)
    combined_near_tiles_pathgains = torch.tensor(
        combined_near_tiles_pathgains.numpy(), dtype=torch.float32
    )

    return (
        combined_near_tiles_indices,
        combined_near_tiles_locations,
        combined_near_tiles_pathgains
    )

def create_chosen_poor_coverage_centroid(baseStations, users_params_list):
    """
    Create a dictionary of chosen poor coverage centroids based on base station and user parameters.

    Parameters:
    - baseStations (list): List of base station dictionaries with associated users.
    - users_params_list (list): List of user parameter dictionaries.

    Returns:
    - Chosen_poor_coverage (list): List of dictionaries containing poor coverage user information.
    """
    # Initialize the Chosen_poor_coverage dictionary
    Chosen_poor_coverage = []

    # Iterate through each base station
    for bs in baseStations:
        # Get the list of users associated with the current base station
        associated_users_indices = bs['users_associated']

        # Get the RSS values for the associated users
        rss_values = [(index, users_params_list[index]['RSS_dBm']) for index in associated_users_indices]

        # Add the information to the Chosen_poor_coverage dictionary
        for user_index, rss_value in rss_values:
            user_info = users_params_list[user_index]
            Chosen_poor_coverage.append({
                'location_BS': (bs['x'], bs['y'], bs['z']),
                'serverID_users': user_info['serverID'],
                'location_users': (user_info['x'], user_info['y'], user_info['z']),
                'siteID_BS': bs['siteID'],
                'user_index': user_index,
                'RSS_dBm': rss_value,
                'Precoder_index': user_info['Precoder_index'],
            })

    return Chosen_poor_coverage


# post_processing.py


def ray_funcs(path_results_centroids_BIRCH, path_types_results_centroids_BIRCH):
    # Initialize empty lists to store the extracted values
    Centroids_all_a = []
    Centroids_all_tau = []
    Centroids_all_vertices = []
    Centroids_all_objects = []

    # Extract data from path_results_centroids
    for key in path_results_centroids_BIRCH.keys():
        if key.startswith('a'):
            Centroids_all_a.append(path_results_centroids_BIRCH[key])
        elif key.startswith('tau'):
            Centroids_all_tau.append(path_results_centroids_BIRCH[key])
        elif key.startswith('vertices'):
            Centroids_all_vertices.append(path_results_centroids_BIRCH[key])
        elif key.startswith('objects'):
            Centroids_all_objects.append(path_results_centroids_BIRCH[key])

    all_tx_path_types_centroids = {}

    # Iterate through all path types and organize them by Tx and Rx
    for key, path_type in path_types_results_centroids_BIRCH.items():
        if key.startswith('Tx') and '_Rx' in key:
            tx_index = key.split('_')[0]
            rx_index = key.split('_')[1]
            if tx_index not in all_tx_path_types_centroids:
                all_tx_path_types_centroids[tx_index] = {}
            all_tx_path_types_centroids[tx_index][rx_index] = path_type

    def merge_rx_for_tx(all_tx_path_types_centroids):
        merged_tx_path_types_centroids = {}
        for tx_index, rx_dict in all_tx_path_types_centroids.items():
            tx_rx_tensors = [tensor for tensor in rx_dict.values()]
            max_num_path = max(tensor.shape[-1] for tensor in tx_rx_tensors)
            padded = [
                tf.pad(t, [[0,0],[0, max_num_path - t.shape[-1]]], constant_values=-1)
                for t in tx_rx_tensors
            ]
            merged = tf.stack(padded, axis=1)
            merged_tx_path_types_centroids[tx_index] = merged
        return merged_tx_path_types_centroids

    merged_tx_path_types_centroids = merge_rx_for_tx(all_tx_path_types_centroids)

    merged_tx_path_types_centroids_list = []
    sorted_keys = sorted(merged_tx_path_types_centroids.keys(),
                         key=lambda x: int(x.replace('Tx','')))
    for key in sorted_keys:
        merged_tx_path_types_centroids_list.append(merged_tx_path_types_centroids[key])

    # Extracting diffraction path type and assign -2 in the objects tensor
    Centroids_all_objects_filtered = []
    num_txs = len(Centroids_all_objects)
    for tx_idx in range(num_txs):
        objects_tensor = Centroids_all_objects[tx_idx]
        path_types_tensor = merged_tx_path_types_centroids_list[tx_idx]
        if path_types_tensor.shape[0] == 1:
            path_types_tensor = tf.squeeze(path_types_tensor, axis=0)
        max_depth, num_targets_objects, num_sources, max_num_paths_objects = objects_tensor.shape
        num_targets_path_types, max_num_paths_path_types = path_types_tensor.shape
        if num_targets_objects != num_targets_path_types:
            raise ValueError(f"Mismatch in num_targets: {num_targets_objects} vs {num_targets_path_types}")
        if max_num_paths_objects > max_num_paths_path_types:
            pad_width = max_num_paths_objects - max_num_paths_path_types
            paddings = tf.constant([[0,0],[0,pad_width]])
            path_types_tensor_padded = tf.pad(path_types_tensor, paddings, mode='CONSTANT', constant_values=2)
        elif max_num_paths_objects < max_num_paths_path_types:
            path_types_tensor_padded = path_types_tensor[:, :max_num_paths_objects]
        else:
            path_types_tensor_padded = path_types_tensor
        mask = tf.equal(path_types_tensor_padded, 2)
        mask_expanded = tf.expand_dims(mask, axis=0)
        mask_expanded = tf.expand_dims(mask_expanded, axis=2)
        mask_broadcasted = tf.broadcast_to(mask_expanded, objects_tensor.shape)
        valid_mask = tf.not_equal(objects_tensor, -1)
        replacement_mask = tf.logical_and(mask_broadcasted, valid_mask)
        filtered = tf.where(replacement_mask, tf.constant(-2, dtype=objects_tensor.dtype), objects_tensor)
        Centroids_all_objects_filtered.append(filtered)

    # Finding best paths for each Centroids
    Centroids_all_a_db = []
    for a in Centroids_all_a:
        power = tf.abs(a)**2
        db = 10 * tf.math.log(power) / tf.math.log(10.0)
        Centroids_all_a_db.append(db)

    best_path_reflections_centroids = []
    for tx_index, a_tx_db in enumerate(Centroids_all_a_db):
        objects_tx = Centroids_all_objects_filtered[tx_index]
        num_receivers = a_tx_db.shape[1]
        for rx_index in range(num_receivers):
            path_gains_rx = a_tx_db[0, rx_index, 0, 0, 0, :, 0]
            if tf.reduce_all(tf.math.is_inf(path_gains_rx) & (path_gains_rx < 0)).numpy():
                best_path_reflections_centroids.append({
                    'Tx': tx_index,
                    'Rx': rx_index,
                    'best_path_index': None,
                    'num_reflections': 'No power received',
                    'path_type': 'No power received'
                })
            else:
                best_path_index = tf.argmax(path_gains_rx).numpy()
                objects_for_rx = objects_tx[:, rx_index, :, best_path_index]
                flat = objects_for_rx.numpy().flatten()
                if np.any(flat == -2):
                    path_type = 'Diffraction'
                    num_reflections = 'Diffraction'
                elif np.all(flat == -1):
                    path_type = 'LoS'
                    num_reflections = 0
                else:
                    path_type = 'Reflection'
                    num_reflections = np.count_nonzero((flat != -1) & (flat != -2))
                best_path_reflections_centroids.append({
                    'Tx': tx_index,
                    'Rx': rx_index,
                    'best_path_index': best_path_index,
                    'num_reflections': num_reflections,
                    'path_type': path_type
                })

    return best_path_reflections_centroids, Centroids_all_vertices, Centroids_all_objects_filtered, Centroids_all_a, Centroids_all_objects                                     
    
    
    Centroids_all_objects = [] 



def parse_RIS_orientation_deg(s):
    # Regular expression to find the TensorFlow tensor's numerical value
    tensor_pattern = r'<tf\.Tensor:.*?numpy=([-+]?\d*\.\d+|\d+)>'
    # Replace the tensor representation with its numerical value
    s = re.sub(tensor_pattern, r'\1', s)
    # Now, s should be a string like '[ -2.7236838, 0, 0 ]'
    # Remove any angle brackets and extra spaces
    s = s.replace('<', '').replace('>', '').strip()
    # Ensure the string is properly formatted as a list
    if not s.startswith('['):
        s = f'[{s}]'
    try:
        # Safely evaluate the string to a Python list
        result = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # Default to [0.0, 0.0, 0.0] if parsing fails
        result = [0.0, 0.0, 0.0]
    return result

import ast


# Define a function to convert array-like strings to tuple-like strings
def convert_bracket_to_parenthesis(point_a_str):
    if isinstance(point_a_str, str):
        point_a_str = point_a_str.strip()
        # Check if the string starts with '[' and ends with ']'
        if point_a_str.startswith('[') and point_a_str.endswith(']'):
            # Remove the square brackets
            point_a_str = point_a_str.strip('[]')
            # Replace multiple spaces with a single space
            point_a_str = ' '.join(point_a_str.split())
            # Split the string by whitespace
            point_a_values = point_a_str.split(' ')
            # Convert values to floats and format as a tuple string
            try:
                point_a_values = [float(value) for value in point_a_values]
                point_a_tuple_str = f'({point_a_values[0]}, {point_a_values[1]}, {point_a_values[2]})'
                return point_a_tuple_str
            except ValueError:
                # If conversion fails, return the original string
                return point_a_str
        else:
            # If not in the target format, return the original string
            return point_a_str
    else:
        # If the value is not a string, return it as is
        return point_a_str


def choose_rss(row):
    if row['RSS_with_RIS'] > row['RSS_without_RIS']:
        return row['RSS_with_RIS']
    elif row['RSS_with_RIS'] == -np.inf:
        return row['RSS_without_RIS']
    elif row['RSS_with_RIS'] < row['RSS_without_RIS']:
        return row['RSS_without_RIS']
    else:  # Covers the case where RSS_with_RIS == RSS_without_RIS
        return row['RSS_with_RIS']


# post_processing.py



def Fina_Dataframe(
    Coverage_user_phase1: pd.DataFrame,
    Coverage_user_phase1_all_path: pd.DataFrame,
    Coverage_user_phase2: pd.DataFrame,
    Coverage_user_phase2_all_path: pd.DataFrame,
    clusters_removed_phase1_npy: str
) -> pd.DataFrame:
    """
    Merge phase‐1 and phase‐2 direct and all‐path coverage DataFrames,
    compute final RSRP per user, track cluster counts, and assign final IDs.

    Inputs:
      - Coverage_user_phase1: pd.DataFrame from T15 direct‐path
      - Coverage_user_phase1_all_path: pd.DataFrame from T15 all‐path
      - Coverage_user_phase2: pd.DataFrame from T10 direct‐path
      - Coverage_user_phase2_all_path: pd.DataFrame from T10 all‐path
      - clusters_removed_phase1_npy: path to numpy file of removed clusters

    Returns:
      - Final_data: assembled pd.DataFrame
    """
    # Define RSRP selection logic
    def choose_rss(row):
        if row['RSS_with_RIS'] > row['RSS_without_RIS']:
            return row['RSS_with_RIS']
        elif row['RSS_with_RIS'] == -np.inf:
            return row['RSS_without_RIS']
        else:
            return row['RSS_without_RIS']

    # Compute Final_RSS for each phase/DataFrame
    for df in (
        Coverage_user_phase1,
        Coverage_user_phase1_all_path,
        Coverage_user_phase2,
        Coverage_user_phase2_all_path
    ):
        df['Final_RSS'] = df.apply(choose_rss, axis=1)

    # Count unique clusters
    n1 = Coverage_user_phase1['Cluster_ID'].nunique()
    print(f"Number of unique clusters in phase1: {n1}")
    n2 = Coverage_user_phase2['Cluster_ID'].nunique()
    print(f"Number of unique clusters in phase2: {n2}")

    # Phase1 direct
    Final_data = Coverage_user_phase1.copy()
    Final_data['phase1'] = Coverage_user_phase1['Final_RSS']

    # Phase1 all‐path
    p1_dict         = dict(zip(Coverage_user_phase1_all_path['Rx_index'], Coverage_user_phase1_all_path['Final_RSS']))
    tx1_dict        = dict(zip(Coverage_user_phase1_all_path['Rx_index'], Coverage_user_phase1_all_path['Tx_index']))
    prec1_dict      = dict(zip(Coverage_user_phase1_all_path['Rx_index'], Coverage_user_phase1_all_path['Precoder_index']))
    pos1_dict       = dict(zip(Coverage_user_phase1_all_path['Rx_index'], Coverage_user_phase1_all_path['RIS_position']))
    ori1_dict       = dict(zip(Coverage_user_phase1_all_path['Rx_index'], Coverage_user_phase1_all_path['RIS_orientation']))
    ptA1_dict       = dict(zip(Coverage_user_phase1_all_path['Rx_index'], Coverage_user_phase1_all_path['Point_A']))

    Final_data['phase1_all_path']     = Final_data['Rx_index'].map(p1_dict).fillna(Final_data['phase1'])
    Final_data['Tx_index']            = Final_data['Rx_index'].map(tx1_dict).fillna(Final_data['Tx_index'])
    Final_data['Precoder_index']      = Final_data['Rx_index'].map(prec1_dict).fillna(Final_data['Precoder_index'])
    Final_data['RIS_position']        = Final_data['Rx_index'].map(pos1_dict).fillna(Final_data['RIS_position'])
    Final_data['RIS_orientation']     = Final_data['Rx_index'].map(ori1_dict).fillna(Final_data['RIS_orientation'])
    Final_data['Point_A']             = Final_data['Rx_index'].map(ptA1_dict).fillna(Final_data.get('Point_A', None))

    # Phase2 direct
    p2_dict         = dict(zip(Coverage_user_phase2['Rx_index'], Coverage_user_phase2['Final_RSS']))
    tx2_dict        = dict(zip(Coverage_user_phase2['Rx_index'], Coverage_user_phase2['Tx_index']))
    prec2_dict      = dict(zip(Coverage_user_phase2['Rx_index'], Coverage_user_phase2['Precoder_index']))
    pos2_dict       = dict(zip(Coverage_user_phase2['Rx_index'], Coverage_user_phase2['RIS_position']))
    ori2_dict       = dict(zip(Coverage_user_phase2['Rx_index'], Coverage_user_phase2['RIS_orientation']))
    ptA2_dict       = dict(zip(Coverage_user_phase2['Rx_index'], Coverage_user_phase2['Point_A']))

    Final_data['phase2']               = Final_data['Rx_index'].map(p2_dict).fillna(Final_data['phase1_all_path'])
    Final_data['Tx_index']            = Final_data['Rx_index'].map(tx2_dict).fillna(Final_data['Tx_index'])
    Final_data['Precoder_index']      = Final_data['Rx_index'].map(prec2_dict).fillna(Final_data['Precoder_index'])
    Final_data['RIS_position']        = Final_data['Rx_index'].map(pos2_dict).fillna(Final_data['RIS_position'])
    Final_data['RIS_orientation']     = Final_data['Rx_index'].map(ori2_dict).fillna(Final_data['RIS_orientation'])
    Final_data['Point_A']             = Final_data['Rx_index'].map(ptA2_dict).fillna(Final_data['Point_A'])

    # Phase2 all‐path
    p2ap_dict      = dict(zip(Coverage_user_phase2_all_path['Rx_index'], Coverage_user_phase2_all_path['Final_RSS']))
    tx2ap_dict     = dict(zip(Coverage_user_phase2_all_path['Rx_index'], Coverage_user_phase2_all_path['Tx_index']))
    prec2ap_dict   = dict(zip(Coverage_user_phase2_all_path['Rx_index'], Coverage_user_phase2_all_path['Precoder_index']))
    pos2ap_dict    = dict(zip(Coverage_user_phase2_all_path['Rx_index'], Coverage_user_phase2_all_path['RIS_position']))
    ori2ap_dict    = dict(zip(Coverage_user_phase2_all_path['Rx_index'], Coverage_user_phase2_all_path['RIS_orientation']))
    ptA2ap_dict    = dict(zip(Coverage_user_phase2_all_path['Rx_index'], Coverage_user_phase2_all_path['Point_A']))

    Final_data['phase2_all_path']     = Final_data['Rx_index'].map(p2ap_dict).fillna(Final_data['phase2'])
    Final_data['Tx_index']            = Final_data['Rx_index'].map(tx2ap_dict).fillna(Final_data['Tx_index'])
    Final_data['Precoder_index']      = Final_data['Rx_index'].map(prec2ap_dict).fillna(Final_data['Precoder_index'])
    Final_data['RIS_position']        = Final_data['Rx_index'].map(pos2ap_dict).fillna(Final_data['RIS_position'])
    Final_data['RIS_orientation']     = Final_data['Rx_index'].map(ori2ap_dict).fillna(Final_data['RIS_orientation'])
    Final_data['Point_A']             = Final_data['Rx_index'].map(ptA2ap_dict).fillna(Final_data['Point_A'])

    # Improvement percentage
    total_users = len(Final_data)
    improved    = (Final_data['phase2_all_path'] > Final_data['RSS_without_RIS']).sum()
    pct         = improved / total_users * 100
    print(f"Percentage of Improved Users: {pct:.2f}%")

    # Load removed clusters and count remaining
    removed = np.load(clusters_removed_phase1_npy, allow_pickle=True)
    unique_clusters = set(Final_data['Cluster_ID'].unique())
    filtered_clusters = unique_clusters - set(removed)
    cnt1 = len(filtered_clusters)
    print(f"Remaining Unique Clusters Count phase1: {cnt1}")

    # Phase2 cluster removal
    not_improved_phase2 = Coverage_user_phase2_all_path.groupby('Cluster_ID').filter(
        lambda x: (x['Final_RSS'] <= x['RSS_without_RIS']).all()
    )['Cluster_ID'].unique()
    remain2 = set(Coverage_user_phase2['Cluster_ID'].unique()) - set(not_improved_phase2)
    cnt2 = len(remain2)
    print(f"Remaining Unique Clusters Count phase2 after removal: {cnt2}")

    # Assign final cluster IDs
    id_map = dict(zip(Coverage_user_phase2['Rx_index'], Coverage_user_phase2['Cluster_ID']))
    Final_data['Cluster_ID_phase2'] = Final_data['Rx_index'].map(id_map).fillna(np.nan)

    def generate_final_cluster_id(row):
        if not pd.isna(row['Cluster_ID_phase2']):
            return f"{int(row['Cluster_ID_phase2'])}-2"
        elif not pd.isna(row['Cluster_ID']):
            return int(row['Cluster_ID'])
        else:
            return np.nan

    Final_data['Final_Cluster_ID'] = Final_data.apply(generate_final_cluster_id, axis=1)
    
    Worst_UEs_Data = Final_data[Final_data['phase2_all_path'] == Final_data['RSS_without_RIS']].copy()


    unique_RIS_data = Final_data[['RIS_position', 'RIS_orientation']].drop_duplicates()
    
    # Convert to a list of tuples for easy access
    unique_RIS_list = unique_RIS_data.values.tolist()
    
    
    
    return Final_data, unique_RIS_list, Worst_UEs_Data











