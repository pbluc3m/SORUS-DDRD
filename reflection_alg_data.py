# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import pandas as pd
import numpy as np
import tensorflow as tf
from RIS_orientation import compute_azimuth_elevation
import re
import ast
import post_processing
###############################################################################


#--------- Reflection-based data --> Strongest Ray Selection Algorithm --------

def Data_Strong_ray(Chosen_poor_centroids_BIRCH,
                    best_path_reflections_centroids,
                    Centroids_all_vertices):
    """
    Prepare DataFrame of centroids with reflection and RIS data.

    Inputs:
    - Chosen_poor_centroids_BIRCH: list of dicts with keys
        'serverID_users','user_index','location_BS','location_users','RSS_dBm','Precoder_index'
    - best_path_reflections_centroids: list of dicts with keys
        'Tx','Rx','best_path_index','num_reflections','path_type'
    - Centroids_all_vertices: list of tf.Tensor, one per Tx, shape
        [max_depth, num_targets, num_sources, max_num_paths, 3]

    Output:
    - df_centroids_BIRCH: pd.DataFrame with columns
        Tx_indice, Rx_indice, Tx_location, Rx_location,
        RSS_rx, Precoder, All_reflection_locations,
        Last_reflection_location,
        Second_to_last_reflection_location,
        Third_to_last_reflection_location,
        Fourth_to_last_reflection_location,
        RIS_deployment_location, RIS_orientation,
        point_A, point_B
    """
    # 1) build base DataFrame
    tx_indices = []
    rx_indices = []
    tx_locations = []
    rx_locations = []
    RSS_rx = []
    Precoder_index = []

    for e in Chosen_poor_centroids_BIRCH:
        tx_indices.append(e['serverID_users'])
        rx_indices.append(e['user_index'])
        tx_locations.append(e['location_BS'])
        rx_locations.append(e['location_users'])
        RSS_rx.append(e['RSS_dBm'])
        Precoder_index.append(e['Precoder_index'])

    df = pd.DataFrame({
        'Tx_indice': tx_indices,
        'Rx_indice': rx_indices,
        'Tx_location': tx_locations,
        'Rx_location': rx_locations,
        'RSS_rx': RSS_rx,
        'Precoder': Precoder_index
    })

    # 2) reflection locations
    all_reflections_list = []
    last_reflection_list = []

    for entry in best_path_reflections_centroids:
        tx_i = entry['Tx']
        rx_i = entry['Rx']
        best_idx = entry['best_path_index']
        n_ref = entry['num_reflections']

        locs = []
        last_loc = None
        if best_idx is not None and isinstance(n_ref, int):
            verts = Centroids_all_vertices[tx_i]
            for d in range(n_ref):
                p = verts[d, rx_i, 0, best_idx, :].numpy()
                locs.append(p)
            if locs:
                last_loc = locs[-1]
        # else locs stays [] and last_loc stays None

        all_reflections_list.append(locs)
        last_reflection_list.append(last_loc)

    df['All_reflection_locations'] = all_reflections_list
    df['Last_reflection_location'] = last_reflection_list

    # 3) 2nd/3rd/4th to last
    sec_list = []
    third_list = []
    fourth_list = []
    for idx, row in df.iterrows():
        refl = row['All_reflection_locations']
        tx_loc = row['Tx_location']
        # 2nd to last
        sec_list.append(refl[-2] if len(refl) > 1 else tx_loc)
        third_list.append(refl[-3] if len(refl) > 2 else tx_loc)
        fourth_list.append(refl[-4] if len(refl) > 3 else tx_loc)

    df['Second_to_last_reflection_location'] = sec_list
    df['Third_to_last_reflection_location']  = third_list
    df['Fourth_to_last_reflection_location'] = fourth_list

    # 4) RIS deployment & orientation
    ris_locs = []
    ris_orients = []
    for idx, row in df.iterrows():
        last     = row['Last_reflection_location']
        sec      = row['Second_to_last_reflection_location']
        third    = row['Third_to_last_reflection_location']
        rx_loc   = row['Rx_location']

        if last is not None and sec is not None and rx_loc is not None:
            if last[2] > 1:
                deploy = last
                A_tf = tf.constant(last, dtype=tf.float32)
                B_tf = tf.constant(sec,  dtype=tf.float32)
                C_tf = tf.constant(rx_loc, dtype=tf.float32)
            else:
                deploy = sec
                A_tf = tf.constant(sec,   dtype=tf.float32)
                B_tf = tf.constant(third, dtype=tf.float32)
                C_tf = tf.constant(rx_loc,dtype=tf.float32)

            az1, el1 = compute_azimuth_elevation(A_tf, B_tf)
            az2, el2 = compute_azimuth_elevation(A_tf, C_tf)
            final_az = (az1 + az2) / 2
            if abs(final_az - az1) > 90:
                final_az += 180
        else:
            deploy = None
            final_az = None

        ris_locs.append(deploy)
        ris_orients.append(final_az)

    df['RIS_deployment_location'] = ris_locs
    df['RIS_orientation']         = ris_orients

    # 5) points A and B for phase gradient
    ptA = []
    ptB = []
    for idx, row in df.iterrows():
        dep = row['RIS_deployment_location']
        last = row['Last_reflection_location']
        sec  = row['Second_to_last_reflection_location']
        third = row['Third_to_last_reflection_location']
        rx    = row['Rx_location']

        if dep is not None and last is not None and np.array_equal(dep, last):
            A = sec
        elif dep is not None and sec is not None and np.array_equal(dep, sec):
            A = third
        else:
            A = None
        B = rx
        ptA.append(A)
        ptB.append(B)

    df['point_A'] = ptA
    df['point_B'] = ptB

    return df



def update_rx_indice(df, nearest_tile_indices):
    """
    Update the Rx_indice or Rx_index column in the given DataFrame based on nearest_tile_indices.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing either 'Rx_indice' or 'Rx_index' column.
    - nearest_tile_indices (sequence): The indices array (length N) used to wrap Rx values.

    Returns:
    - pd.DataFrame: The updated DataFrame.
    """
    def transform_rx(rx_value):
        N = len(nearest_tile_indices)
        if 0 <= rx_value <=     N - 1:
            return rx_value
        elif     N <= rx_value <= 2*N - 1:
            return rx_value -     N
        elif 2*N <= rx_value <= 3*N - 1:
            return rx_value - 2 * N
        elif 3*N <= rx_value <= 4*N - 1:
            return rx_value - 3 * N
        else:
            return rx_value

    if 'Rx_indice' in df.columns:
        df['Rx_indice'] = df['Rx_indice'].apply(transform_rx)
    elif 'Rx_index' in df.columns:
        df['Rx_index']  = df['Rx_index'].apply(transform_rx)
    else:
        raise KeyError("Neither 'Rx_indice' nor 'Rx_index' column found in the DataFrame.")
    return df


def Choose_centroid_with_ray(df):
    """
    Filters the DataFrame by Rx_indice to retain only one user for each unique Rx_indice,
    choosing the first entry with a non-None Last_reflection_location, then sorts by Tx_indice.
    """
    filtered_rows = []

    for rx_indice, group in df.groupby('Rx_indice'):
        non_none_users = group[group['Last_reflection_location'].notna()]
        if not non_none_users.empty:
            filtered_rows.append(non_none_users.iloc[0])

    filtered_df = pd.DataFrame(filtered_rows)
    sorted_df = filtered_df.sort_values(by='Tx_indice').reset_index(drop=True)
    return sorted_df

def Data2_Strong_ray(df_filtered):
    """
    Create df2 with updated RIS deployment criteria from df_filtered.
    """
    df2 = df_filtered.copy()

    ris_locs = []
    ris_orients = []
    for _, row in df2.iterrows():
        sec = row['Second_to_last_reflection_location']
        third = row['Third_to_last_reflection_location']
        last = row['Last_reflection_location']
        tx_loc = row['Tx_location']
        rx_loc = row['Rx_location']

        if sec is not None and third is not None and last is not None and rx_loc is not None:
            if np.array_equal(sec, tx_loc):
                deploy = last
                A_tf = tf.constant(last, dtype=tf.float32)
                B_tf = tf.constant(tx_loc, dtype=tf.float32)
                C_tf = tf.constant(rx_loc, dtype=tf.float32)
                az1, _ = compute_azimuth_elevation(A_tf, B_tf)
                az2, _ = compute_azimuth_elevation(A_tf, C_tf)
            else:
                deploy = sec
                A_tf = tf.constant(sec,   dtype=tf.float32)
                B_tf = tf.constant(third, dtype=tf.float32)
                C_tf = tf.constant(rx_loc,dtype=tf.float32)
                az1, _ = compute_azimuth_elevation(A_tf, B_tf)
                az2, _ = compute_azimuth_elevation(A_tf, C_tf)

            final_az = (az1 + az2) / 2
            if abs(final_az - az1) > 90:
                final_az += 180
        else:
            deploy = None
            final_az = None

        ris_locs.append(deploy)
        ris_orients.append(final_az)

    df2['RIS_deployment_location'] = ris_locs
    df2['RIS_orientation']         = ris_orients

    # point_A always third-to-last, point_B always Rx_location
    df2['point_A'] = df2['Third_to_last_reflection_location']
    df2['point_B'] = df2['Rx_location']

    return df2

def Data3_Strong_ray(df_centroids_BIRCH_fitered2):
    # 1) copy df2 to df3
    df3 = df_centroids_BIRCH_fitered2.copy()

    # 2) fourth-to-last reflection
    fourth_list = []
    for _, row in df3.iterrows():
        all_ref = row['All_reflection_locations']
        if len(all_ref) > 3:
            fourth = all_ref[-4]
        else:
            fourth = row['Tx_location']
        fourth_list.append(fourth)
    df3['Fourth_to_last_reflection_location'] = fourth_list

    # 3) RIS deploy & orientation
    ris_locs = []
    ris_orients = []
    for _, row in df3.iterrows():
        third = row['Third_to_last_reflection_location']
        fourth = row['Fourth_to_last_reflection_location']
        last = row['Last_reflection_location']
        tx_loc = row['Tx_location']
        rx_loc = row['Rx_location']

        if third is not None and rx_loc is not None and last is not None:
            if np.array_equal(third, tx_loc):
                deploy = last
                A_tf = tf.constant(last, dtype=tf.float32)
                B_tf = tf.constant(tx_loc, dtype=tf.float32)
                C_tf = tf.constant(rx_loc, dtype=tf.float32)
                az1, _ = compute_azimuth_elevation(A_tf, B_tf)
                az2, _ = compute_azimuth_elevation(A_tf, C_tf)
            else:
                deploy = third
                A_tf = tf.constant(third, dtype=tf.float32)
                C_tf = tf.constant(rx_loc, dtype=tf.float32)
                if fourth is not None:
                    B_tf = tf.constant(fourth, dtype=tf.float32)
                    az1, _ = compute_azimuth_elevation(A_tf, B_tf)
                else:
                    B_tf = tf.constant(tx_loc, dtype=tf.float32)
                    az1, _ = compute_azimuth_elevation(A_tf, B_tf)
                az2, _ = compute_azimuth_elevation(A_tf, C_tf)

            final_az = (az1 + az2) / 2
            if abs(final_az - az1) > 90:
                final_az += 180
        else:
            deploy = None
            final_az = None

        ris_locs.append(deploy)
        ris_orients.append(final_az)

    df3['RIS_deployment_location'] = ris_locs
    df3['RIS_orientation']         = ris_orients

    # 4) points A & B
    ptA = df3['Fourth_to_last_reflection_location']
    ptB = df3['Rx_location']
    df3['point_A'] = ptA
    df3['point_B'] = ptB

    return df3


def Data4_Strong_ray(df_centroids_BIRCH_fitered3):
    df4 = df_centroids_BIRCH_fitered3.copy()

    ris_deployment_locations_df4 = []
    ris_orientations_df4 = []

    for _, row in df4.iterrows():
        fourth_to_last_reflection_location = row['Fourth_to_last_reflection_location']
        last_reflection_location = row['Last_reflection_location']
        second_to_last_reflection_location = row['Second_to_last_reflection_location']
        tx_location = row['Tx_location']
        rx_location = row['Rx_location']

        if fourth_to_last_reflection_location is not None and rx_location is not None and last_reflection_location is not None:
            if np.array_equal(fourth_to_last_reflection_location, tx_location):
                ris_deployment_location = last_reflection_location
                last_reflection_location_tf = tf.constant(last_reflection_location, dtype=tf.float32)
                second_to_last_reflection_location_tf = tf.constant(second_to_last_reflection_location, dtype=tf.float32)
                rx_location_tf = tf.constant(rx_location, dtype=tf.float32)

                azimuth1_deg, _ = compute_azimuth_elevation(
                    last_reflection_location_tf, second_to_last_reflection_location_tf
                )
                azimuth2_deg, _ = compute_azimuth_elevation(
                    last_reflection_location_tf, rx_location_tf
                )
            else:
                ris_deployment_location = fourth_to_last_reflection_location
                fourth_to_last_reflection_location_tf = tf.constant(fourth_to_last_reflection_location, dtype=tf.float32)
                tx_location_tf = tf.constant(tx_location, dtype=tf.float32)
                rx_location_tf = tf.constant(rx_location, dtype=tf.float32)

                azimuth1_deg, _ = compute_azimuth_elevation(
                    fourth_to_last_reflection_location_tf, tx_location_tf
                )
                azimuth2_deg, _ = compute_azimuth_elevation(
                    fourth_to_last_reflection_location_tf, rx_location_tf
                )

            final_orientation_azimuth = (azimuth1_deg + azimuth2_deg) / 2
            if abs(final_orientation_azimuth - azimuth1_deg) > 90:
                final_orientation_azimuth += 180

            ris_deployment_locations_df4.append(ris_deployment_location)
            ris_orientations_df4.append(final_orientation_azimuth)
        else:
            ris_deployment_locations_df4.append(None)
            ris_orientations_df4.append(None)

    df4['RIS_deployment_location'] = ris_deployment_locations_df4
    df4['RIS_orientation']         = ris_orientations_df4

    point_A_list_df4 = []
    point_B_list_df4 = []

    for _, row in df4.iterrows():
        tx_location = row['Tx_location']
        rx_location = row['Rx_location']
        point_A_list_df4.append(tx_location)
        point_B_list_df4.append(rx_location)

    df4['point_A'] = point_A_list_df4
    df4['point_B'] = point_B_list_df4

    return df4

def Data_prep_precoder(final_results_df_centroids, df_centroids_BIRCH_fitered):
    """
    Merge in Precoder info and select the strongest-RIS configuration per user.

    Inputs:
      - final_results_df_centroids: pd.DataFrame with coverage results and
          columns ['Rx_index','RSS_without_RIS','RSS_dB_RIS',
                    'RSS_RIS_df2','RSS_RIS_df3','RSS_RIS_df4',
                    'RIS_position','RIS_orientation',
                    'RIS_position_df2','RIS_orientation_df2',
                    'RIS_position_df3','RIS_orientation_df3',
                    'RIS_position_df4','RIS_orientation_df4',
                    'point_A_x','point_A_y','point_A_left','point_A_right', ...]
      - df_centroids_BIRCH_fitered: pd.DataFrame with columns ['Rx_indice','Precoder']

    Returns:
      - Best_RIS_RSS_Configuration_BIRCH_Centroids: pd.DataFrame with columns
          ['Tx_indice','Rx_indice','Tx_location','Rx_location','Precoder',
           'RSS_without_RIS','Best_RSS','RIS_deployment_location',
           'RIS_orientation','Point_A']
    """
    # Ensure both DataFrames have Rx_indice and select the relevant columns
    df_temp = df_centroids_BIRCH_fitered[['Rx_indice', 'Precoder']].copy()

    # Merge the DataFrames on the Rx_indice column
    merged = final_results_df_centroids.merge(
        df_temp,
        left_on='Rx_index',
        right_on='Rx_indice',
        how='left'
    )
    # Drop the extra Rx_indice column from the merge
    merged.drop(columns=['Rx_indice'], inplace=True)

    # Initialize lists to store the best LSG configuration data
    tx_indices = []
    rx_indices = []
    tx_locations = []
    rx_locations = []
    rss_without_ris_list = []
    best_rss_list = []
    ris_deployment_locations = []
    ris_orientations = []
    point_A_list = []
    Precoder_list = []

    # Iterate over each row to determine the best LSG configuration
    for _, row in merged.iterrows():
        rss_without_ris = row['RSS_without_RIS']
        rss_dB_ris      = row.get('RSS_dB_RIS',     float('-inf'))
        rss_ris_df2     = row.get('RSS_RIS_df2',    float('-inf'))
        rss_ris_df3     = row.get('RSS_RIS_df3',    float('-inf'))
        rss_ris_df4     = row.get('RSS_RIS_df4',    float('-inf'))

        rss_values = [rss_dB_ris, rss_ris_df2, rss_ris_df3, rss_ris_df4]
        best_rss   = max(rss_values)

        if best_rss <= rss_without_ris:
            best_rss   = rss_without_ris
            best_index = -1
        else:
            best_index = rss_values.index(best_rss)

        # Select corresponding RIS config and point_A
        if   best_index == 0:
            ris_deployment_location = row['RIS_position']
            ris_orientation         = row['RIS_orientation']
            point_A                 = row['point_A_x']
        elif best_index == 1:
            ris_deployment_location = row['RIS_position_df2']
            ris_orientation         = row['RIS_orientation_df2']
            point_A                 = row['point_A_y']
        elif best_index == 2:
            ris_deployment_location = row['RIS_position_df3']
            ris_orientation         = row['RIS_orientation_df3']
            point_A                 = row['point_A_left']
        elif best_index == 3:
            ris_deployment_location = row['RIS_position_df4']
            ris_orientation         = row['RIS_orientation_df4']
            point_A                 = row['point_A_right']
        else:
            ris_deployment_location = row['RIS_position']
            ris_orientation         = row['RIS_orientation']
            point_A                 = row['point_A_x']

        tx_indices.append(row['Tx_index'])
        rx_indices.append(row['Rx_index'])
        tx_locations.append(row['Tx_location'])
        rx_locations.append(row['Rx_location'])
        Precoder_list.append(row['Precoder'])
        rss_without_ris_list.append(rss_without_ris)
        best_rss_list.append(best_rss)
        ris_deployment_locations.append(ris_deployment_location)
        ris_orientations.append(ris_orientation)
        point_A_list.append(point_A)

    # Create the new DataFrame
    Best_RIS_RSS_Configuration_BIRCH_Centroids = pd.DataFrame({
        'Tx_indice': tx_indices,
        'Rx_indice': rx_indices,
        'Tx_location': tx_locations,
        'Rx_location': rx_locations,
        'Precoder': Precoder_list,
        'RSS_without_RIS': rss_without_ris_list,
        'Best_RSS': best_rss_list,
        'RIS_deployment_location': ris_deployment_locations,
        'RIS_orientation': ris_orientations,
        'Point_A': point_A_list
    })

    return Best_RIS_RSS_Configuration_BIRCH_Centroids
###############################################################################

#--------------------  Reflection-based data --> All Ray Algorithm ------------

def PreData_All_Ray(
    Centroids_all_vertices,
    Centroids_all_objects,
    Centroids_all_a,
    Centroids_all_objects_filtered
):
    """
    Process raw ray-tracing outputs into path-detail data.

    Inputs:
      - Centroids_all_vertices: list of tf.Tensor, each shape
          [max_depth, num_targets, num_sources, max_num_paths, 3]
      - Centroids_all_objects:  list of tf.Tensor, each shape
          [max_depth, num_targets, num_sources, max_num_paths]
      - Centroids_all_a:        list of tf.Tensor of complex path gains
      - Centroids_all_objects_filtered: list of tf.Tensor, same shape as Centroids_all_objects,
          but with diffraction paths marked as -2

    Returns:
      - processed_Centroids_all_vertices: list of tf.Tensor, truncated to first 2 depths
      - processed_Centroids_all_objects:  list of tf.Tensor, truncated to first 2 depths
      - updated_vertices:                list of tf.Tensor, vertices zeroed where objects == -1
      - Centroids_all_a_db:              list of tf.Tensor, path gains in dB
      - all_path_details_centroids:      list of dict, one per Rx, containing path_index, power_dB,
                                         num_reflections, path_type
    """
    # 1) truncate to first two depths
    processed_Centroids_all_vertices = []
    for tensor in Centroids_all_vertices:
        processed_Centroids_all_vertices.append(tensor[:2, ...])

    processed_Centroids_all_objects = []
    for tensor in Centroids_all_objects:
        processed_Centroids_all_objects.append(tensor[:2, ...])

    # 2) zero out invalid vertices
    updated_vertices = []
    for vertices_tensor, objects_tensor in zip(processed_Centroids_all_vertices, processed_Centroids_all_objects):
        expanded_objects = tf.expand_dims(objects_tensor, axis=-1)
        valid_mask = tf.cast(expanded_objects != -1, tf.float32)
        updated_vertices.append(vertices_tensor * valid_mask)

    # 3) convert complex gains to dB
    Centroids_all_a_db = []
    for a in Centroids_all_a:
        path_gain_power = tf.abs(a)**2
        path_gain_db = 10 * tf.math.log(path_gain_power) / tf.math.log(10.0)
        Centroids_all_a_db.append(path_gain_db)

    # 4) assemble per‐path details
    all_path_details_centroids = []
    for tx_index, a_tx_db in enumerate(Centroids_all_a_db):
        objects_tx = Centroids_all_objects_filtered[tx_index]
        num_receivers = a_tx_db.shape[1]
        for rx_index in range(num_receivers):
            path_gains_rx = a_tx_db[0, rx_index, 0, 0, 0, :, 0]
            if tf.reduce_all(tf.math.is_inf(path_gains_rx) & (path_gains_rx < 0)).numpy():
                all_path_details_centroids.append({
                    'Tx': tx_index,
                    'Rx': rx_index,
                    'paths': [{
                        'path_index': None,
                        'power_dB': None,
                        'num_reflections': 'No power received',
                        'path_type': 'No power received'
                    }]
                })
            else:
                paths_details = []
                for path_index in range(path_gains_rx.shape[0]):
                    power_dB = path_gains_rx[path_index].numpy()
                    objects_for_rx = objects_tx[:, rx_index, :, path_index]
                    flat = objects_for_rx.numpy().flatten()
                    if np.any(flat == -2):
                        path_type = 'Diffraction'
                        num_reflections = 'Diffraction'
                    elif np.all(flat == -1):
                        path_type = 'LoS'
                        num_reflections = 0
                    else:
                        path_type = 'Reflection'
                        num_reflections = int(np.count_nonzero((flat != -1) & (flat != -2)))
                    paths_details.append({
                        'path_index': path_index,
                        'power_dB': power_dB,
                        'num_reflections': num_reflections,
                        'path_type': path_type
                    })
                all_path_details_centroids.append({
                    'Tx': tx_index,
                    'Rx': rx_index,
                    'paths': paths_details
                })

    return (
        processed_Centroids_all_vertices,
        processed_Centroids_all_objects,
        updated_vertices,
        Centroids_all_a_db,
        all_path_details_centroids
    )


def Data_All_Ray(
    Chosen_poor_centroids_BIRCH: list,
    all_path_details_centroids: list,
    Centroids_all_vertices: list
) -> pd.DataFrame:
    """
    Build a per‐path DataFrame for centroid users.

    Inputs:
      - Chosen_poor_centroids_BIRICH: list of dicts with keys
          ['serverID_users','user_index','location_BS','location_users','RSS_dBm','Precoder_index']
      - all_path_details_centroids: list of dicts with keys
          ['Tx','Rx','paths'] where paths is list of dicts
          ['path_index','power_dB','num_reflections','path_type']
      - Centroids_all_vertices: list of tf.Tensor, each shape
          [max_depth, num_targets, num_sources, max_num_paths, 3]

    Returns:
      - df_paths_centroids_BIRCH: pd.DataFrame
    """
    # 1) base user table
    tx_indices = []
    rx_indices = []
    tx_locations = []
    rx_locations = []
    RSS_rx = []
    Precoder_index = []

    for entry in Chosen_poor_centroids_BIRCH:
        tx_indices.append(entry['serverID_users'])
        rx_indices.append(entry['user_index'])
        tx_locations.append(entry['location_BS'])
        rx_locations.append(entry['location_users'])
        RSS_rx.append(entry['RSS_dBm'])
        Precoder_index.append(entry['Precoder_index'])

    df_all_path_centroids_BIRCH = pd.DataFrame({
        'Tx_indice': tx_indices,
        'Rx_indice': rx_indices,
        'Tx_location': tx_locations,
        'Rx_location': rx_locations,
        'RSS_rx': RSS_rx,
        'Precoder': Precoder_index,
    })

    # 2) extract per‐path reflection info
    all_reflection_locations_list = []
    last_reflection_locations_list  = []
    path_indices_list               = []
    rx_global_index = 0

    for entry in all_path_details_centroids:
        tx_index = entry['Tx']
        rx_index = entry['Rx']
        paths    = entry['paths']
        reflection_found = False

        for path in paths:
            if path['path_type'] == 'Reflection':
                pi = path['path_index']
                nr = path['num_reflections']
                all_ref_locs  = []
                last_ref_loc  = None
                if isinstance(nr, int) and nr > 0:
                    verts = Centroids_all_vertices[tx_index]
                    for depth in range(nr):
                        rl = verts[depth, rx_index, 0, pi, :].numpy()
                        all_ref_locs.append(rl)
                    if all_ref_locs:
                        last_ref_loc = all_ref_locs[-1]
                path_indices_list.append((rx_global_index, pi))
                all_reflection_locations_list.append((rx_global_index, all_ref_locs))
                last_reflection_locations_list.append((rx_global_index, last_ref_loc))
                reflection_found = True

        if not reflection_found:
            path_indices_list.append((rx_global_index, None))
            all_reflection_locations_list.append((rx_global_index, None))
            last_reflection_locations_list.append((rx_global_index, None))

        rx_global_index += 1

    df_all_path_info = pd.DataFrame({
        'Rx_Global_Index': [p[0] for p in path_indices_list],
        'Path_index':       [p[1] for p in path_indices_list],
        'All_reflection_locations': [p[1] for p in all_reflection_locations_list],
        'Last_reflection_location': [p[1] for p in last_reflection_locations_list],
    })

    # 3) merge with user table
    df_paths_centroids_BIRCH = df_all_path_info.merge(
        df_all_path_centroids_BIRCH,
        left_on='Rx_Global_Index',
        right_index=True,
        how='left'
    ).reset_index(drop=True)

    # empty‐list → None for convenience
    df_paths_centroids_BIRCH['All_reflection_locations'] = (
        df_paths_centroids_BIRCH['All_reflection_locations']
        .apply(lambda x: [] if x is None else x)
    )

    # 4) pull out 1st/2nd/3rd reflections
    first_locs  = []
    second_locs = []
    third_locs  = []
    for _, row in df_paths_centroids_BIRCH.iterrows():
        all_rl = row['All_reflection_locations']
        first_locs.append( all_rl[0] if len(all_rl)>0 else [] )
        second_locs.append(all_rl[1] if len(all_rl)>1 else [] )
        third_locs.append( all_rl[2] if len(all_rl)>2 else [] )

    df_paths_centroids_BIRCH['First_reflection_location']  = first_locs
    df_paths_centroids_BIRCH['Second_reflection_location'] = second_locs
    df_paths_centroids_BIRCH['Third_reflection_location']  = third_locs

    # convert empty‐list to None
    for col in ['First_reflection_location','Second_reflection_location','Third_reflection_location']:
        df_paths_centroids_BIRCH[col] = (
            df_paths_centroids_BIRCH[col]
            .apply(lambda x: None if isinstance(x,list) and not x else x)
        )

    # 5) RIS deployment & orientation at first reflection
    ris_locs = []
    ris_oris = []
    for _, row in df_paths_centroids_BIRCH.iterrows():
        fr = row['First_reflection_location']
        tx = row['Tx_location']
        rx = row['Rx_location']
        if fr is not None:
            A_tf = tf.constant(fr, dtype=tf.float32)
            tx_tf= tf.constant(tx, dtype=tf.float32)
            rx_tf= tf.constant(rx, dtype=tf.float32)
            az1, _ = compute_azimuth_elevation(A_tf, tx_tf)
            az2, _ = compute_azimuth_elevation(A_tf, rx_tf)
            final_az = (az1+az2)/2
            if abs(final_az-az1)>90: final_az += 180
            ris_locs.append(fr)
            ris_oris.append(final_az)
        else:
            ris_locs.append(None)
            ris_oris.append(None)

    df_paths_centroids_BIRCH['RIS_deployment_location'] = ris_locs
    df_paths_centroids_BIRCH['RIS_orientation']         = ris_oris

    # 6) point_A & point_B
    pA = []
    pB = []
    for _, row in df_paths_centroids_BIRCH.iterrows():
        fr = row['First_reflection_location']
        tx = row['Tx_location']
        rx = row['Rx_location']
        if fr is not None and np.array_equal(row['RIS_deployment_location'],fr):
            pA.append(tx)
        else:
            pA.append(None)
        pB.append(rx)

    df_paths_centroids_BIRCH['point_A'] = pA
    df_paths_centroids_BIRCH['point_B'] = pB

    return df_paths_centroids_BIRCH


def Data2_All_Ray(df_paths_centroids_BIRCH):
    """
    Given df_paths_centroids_BIRCH, compute RIS deployment/orientation based on
    the second reflection, then set point_A to the first reflection and point_B to Rx.
    """
    df2 = df_paths_centroids_BIRCH.copy()

    # 1) Compute new RIS deployment location & orientation based on second reflection
    ris_deployment_locations_df2 = []
    ris_orientations_df2 = []
    for _, row in df2.iterrows():
        second_reflection_location = row['Second_reflection_location']
        first_reflection_location  = row['First_reflection_location']
        tx_location                = row['Tx_location']
        rx_location                = row['Rx_location']

        if second_reflection_location is not None:
            ris_deployment_location = second_reflection_location

            # TF constants for azimuth/elevation
            a_tf = tf.constant(second_reflection_location, dtype=tf.float32)
            b_tf = tf.constant(first_reflection_location,  dtype=tf.float32)
            c_tf = tf.constant(rx_location,                dtype=tf.float32)

            az1_deg, _ = compute_azimuth_elevation(a_tf, b_tf)
            az2_deg, _ = compute_azimuth_elevation(a_tf, c_tf)

            final_az = (az1_deg + az2_deg) / 2
            if abs(final_az - az1_deg) > 90:
                final_az += 180

            ris_deployment_locations_df2.append(ris_deployment_location)
            ris_orientations_df2.append(final_az)
        else:
            ris_deployment_locations_df2.append(None)
            ris_orientations_df2.append(None)

    df2['RIS_deployment_location'] = ris_deployment_locations_df2
    df2['RIS_orientation']         = ris_orientations_df2

    # 2) Set point_A = first reflection, point_B = Rx location
    point_A_list_df2 = []
    point_B_list_df2 = []
    for _, row in df2.iterrows():
        first_reflection_location = row['First_reflection_location']
        rx_location               = row['Rx_location']
        point_A_list_df2.append(first_reflection_location)
        point_B_list_df2.append(rx_location)

    df2['point_A'] = point_A_list_df2
    df2['point_B'] = point_B_list_df2

    return df2



def All_ray_filtering_data(
    results_cluster_df_centroids: pd.DataFrame,
    final_results_df_centroids_all_paths: pd.DataFrame
):
    """
    Filter cluster and full‐path results into two DataFrames:
      - clusters whose steering coverage < 40% improved users
      - within those, split into centroid‐improved vs. not‐improved groups
    """
    # Phase 1: drop clusters where ≥40% of users improved in steering stage
    grouped = results_cluster_df_centroids.groupby('Cluster_ID')
    clusters_to_keep = []
    for cluster_id, group in grouped:
        improved_ratio = (group['RSS_with_RIS'] > group['RSS_without_RIS']).mean()
        if improved_ratio < 0.4:
            clusters_to_keep.append(cluster_id)
    No_improved_users_phase1 = results_cluster_df_centroids[
        results_cluster_df_centroids['Cluster_ID'].isin(clusters_to_keep)
    ]

    # Phase 2: restrict full-path results to those clusters
    clusters_to_keep_phase1 = No_improved_users_phase1['Cluster_ID'].unique()
    No_improved_centroids_all_path_results = final_results_df_centroids_all_paths[
        final_results_df_centroids_all_paths['Rx_index'].isin(clusters_to_keep_phase1)
    ]

    # Phase 3: within those, split by ≥3 dB gain under any path method
    grouped2 = No_improved_centroids_all_path_results.groupby('Rx_index')
    indices_improved     = []
    indices_not_improved = []
    for rx_index, group in grouped2:
        cond = (
            (group['RSS_dB_RIS']   > group['RSS_without_RIS'] + 3) |
            (group['RSS_RIS_df2']  > group['RSS_without_RIS'] + 3)
        ).any()
        if cond:
            indices_improved.extend(group.index)
        else:
            indices_not_improved.extend(group.index)

    filtered_improved_centroids_all_path_results = No_improved_centroids_all_path_results.loc[indices_improved]
    filtered_No_improved_centroids_all_path_results = No_improved_centroids_all_path_results.loc[indices_not_improved]

    return filtered_improved_centroids_all_path_results, filtered_No_improved_centroids_all_path_results, clusters_to_keep_phase1


def Data_prep_precoder_all_ray(filtered_improved_centroids_all_path_results: pd.DataFrame) -> pd.DataFrame:
    """
    From the filtered improved results, choose for each user the best RIS deployment
    (centroid vs. full‐path) and corresponding precoder, build a new DataFrame.
    """
    tx_indices = []
    rx_indices = []
    tx_locations = []
    rx_locations = []
    rss_without_ris_list = []
    best_rss_list = []
    ris_deployment_locations = []
    ris_orientations = []
    point_A_list = []
    Precoder = []

    for _, row in filtered_improved_centroids_all_path_results.iterrows():
        # base RSS
        rss_without_ris = row['RSS_without_RIS']
        # two RIS options
        rss_dB_ris   = row.get('RSS_dB_RIS', float('-inf'))
        rss_ris_df2  = row.get('RSS_RIS_df2', float('-inf'))

        rss_values = [rss_dB_ris, rss_ris_df2]
        best_rss   = max(rss_values)
        best_index = rss_values.index(best_rss)

        if best_index == 0:
            ris_deployment_location = row['RIS_position']
            ris_orientation         = row['RIS_orientation']
            point_A                 = row['point_A_x']
            precoder_value          = row['Precoder_x']
        else:
            ris_deployment_location = row['RIS_position_df2']
            ris_orientation         = row['RIS_orientation_df2']
            point_A                 = row['point_A_y']
            precoder_value          = row['Precoder_y']

        # convert precoder string → tuple
        if isinstance(precoder_value, str):
            precoder_value = ast.literal_eval(precoder_value)
        elif not isinstance(precoder_value, tuple):
            raise TypeError(f"Invalid type for precoder: {type(precoder_value)}")

        # parse Rx_location
        Rx_loc = row['Rx_location']
        if isinstance(Rx_loc, str):
            Rx_loc = ast.literal_eval(Rx_loc)
        Rx_loc = np.array(Rx_loc, dtype=np.float32)

        # parse RIS deployment location
        try:
            clean_location = re.sub(r'[\[\]]', '', ris_deployment_location).split()
            ris_deployment_location = np.array([float(num) for num in clean_location])
        except (ValueError, TypeError):
            continue

        tx_indices.append(row['Tx_index'])
        rx_indices.append(row['Rx_index'])
        tx_locations.append(row['Tx_location'])
        rx_locations.append(Rx_loc)
        rss_without_ris_list.append(rss_without_ris)
        best_rss_list.append(best_rss)
        ris_deployment_locations.append(ris_deployment_location)
        ris_orientations.append(ris_orientation)
        point_A_list.append(point_A)
        Precoder.append(precoder_value)

    Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path = pd.DataFrame({
        'Tx_indice': tx_indices,
        'Rx_indice': rx_indices,
        'Tx_location': tx_locations,
        'Rx_location': rx_locations,
        'RSS_without_RIS': rss_without_ris_list,
        'Best_RSS': best_rss_list,
        'RIS_deployment_location': ris_deployment_locations,
        'RIS_orientation': ris_orientations,
        'Point_A': point_A_list,
        'Precoder': Precoder
    })

    return Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path


def Select_RIS_All_Ray(
    Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path: pd.DataFrame,
    clusters_to_keep_phase1: list,
    output_csv_path: str
) -> (pd.DataFrame, list):
    """
    From the best‐RSRP configuration DataFrame and initial cluster list,
    compute which clusters remain, then pick for each Rx_index the single
    RIS deployment with maximum improvement (ties broken by minimum distance).

    Inputs:
      - Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path: DataFrame with columns
          ['Rx_indice','Best_RSS','RSS_without_RIS',
           'RIS_deployment_location','Rx_location',...]
      - clusters_to_keep_phase1: list of cluster IDs from phase‐1 filtering
      - output_csv_path: path to write the final selected‐RIS CSV

    Returns:
      - df_selected: DataFrame after selection
      - filtered_clusters_to_keep_phase1: list of clusters not present in Rx_indice
    """
    # 1) Count final clusters vs. initial list
    unique_rx_indices = set(
        Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path['Rx_indice'].unique()
    )
    filtered_clusters_to_keep_phase1 = [
        x for x in clusters_to_keep_phase1
        if x not in unique_rx_indices
    ]

    # 2) Define per‐Rx selection
    def select_most_improved_user(group):
        improved = group[group['Best_RSS'] > group['RSS_without_RIS']]
        if improved.empty:
            return group.iloc[0:0]

        improved = improved.copy()
        improved['RSS_improvement'] = (
            improved['Best_RSS'] - improved['RSS_without_RIS']
        )
        max_imp = improved['RSS_improvement'].max()
        top = improved[improved['RSS_improvement'] == max_imp]

        if len(top) > 1:
            top['2D_distance'] = top.apply(
                lambda r: np.sqrt(
                    (r['RIS_deployment_location'][0] - r['Rx_location'][0])**2 +
                    (r['RIS_deployment_location'][1] - r['Rx_location'][1])**2
                ),
                axis=1
            )
            selected = top.nsmallest(1, '2D_distance')
        else:
            selected = top

        return selected

    # 3) Apply grouping & selection
    df_selected = (
        Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path
        .groupby('Rx_indice', group_keys=False)
        .apply(select_most_improved_user)
    )

    # 4) Clean up
    drop_cols = ['RSS_improvement', '2D_distance']
    df_selected = df_selected.drop(
        columns=[c for c in drop_cols if c in df_selected.columns]
    )

    # 5) Save and return
    df_selected.to_csv(output_csv_path, index=False)
    return df_selected, filtered_clusters_to_keep_phase1

###############################################################################

def Data_Recluster(
    results_cluster_df_centroids: pd.DataFrame,
    final_results_df_centroids_all_paths: pd.DataFrame,
    users_all_path_csv: str
) -> pd.DataFrame:
    """
    Recluster filtering and prepare data for reclustering.

    Inputs:
      - results_cluster_df_centroids: DataFrame from initial cluster steering stage
      - final_results_df_centroids_all_paths: DataFrame from full‐path RIS coverage
      - users_all_path_csv: path to CSV with all‐path user results

    Output:
      - Data_algh3_clustering saved to 'Dataframe_prepared_10GHz_T15_clustering_again.csv'
      - Returned Data_algh3_clustering DataFrame
    """
    # Phase 1: keep clusters with <40% improved in steering
    grouped = results_cluster_df_centroids.groupby('Cluster_ID')
    clusters_to_keep = []
    for cluster_id, group in grouped:
        improved_ratio = (group['RSS_with_RIS'] > group['RSS_without_RIS']).mean()
        if improved_ratio < 0.4:
            clusters_to_keep.append(cluster_id)
    No_improved_users_phase1 = results_cluster_df_centroids[
        results_cluster_df_centroids['Cluster_ID'].isin(clusters_to_keep)
    ]

    # Phase 2: restrict full‐path results to those clusters
    clusters_to_keep_phase1 = No_improved_users_phase1['Cluster_ID'].unique()
    No_improved_centroids_all_path_results = final_results_df_centroids_all_paths[
        final_results_df_centroids_all_paths['Rx_index'].isin(clusters_to_keep_phase1)
    ]

    # Phase 3: within those, split by ≥3 dB improvement under any path
    grouped2 = No_improved_centroids_all_path_results.groupby('Rx_index')
    indices_improved     = []
    indices_not_improved = []
    for rx_index, group in grouped2:
        cond = (
            (group['RSS_dB_RIS']  > group['RSS_without_RIS'] + 3) |
            (group['RSS_RIS_df2'] > group['RSS_without_RIS'] + 3)
        ).any()
        if cond:
            indices_improved.extend(group.index)
        else:
            indices_not_improved.extend(group.index)
    filtered_improved = No_improved_centroids_all_path_results.loc[indices_improved]
    filtered_not     = No_improved_centroids_all_path_results.loc[indices_not_improved]

    cluster_ids_to_keep = filtered_not['Rx_index'].unique()

    Rest_users_clustering_again = No_improved_users_phase1[
        No_improved_users_phase1['Cluster_ID'].isin(cluster_ids_to_keep)
    ]

    # Phase 4: load all‐path user results and filter those unchanged
    Final_results_users_all_path = pd.read_csv(users_all_path_csv)
    Final_results_users_all_path['Final_RSS'] = Final_results_users_all_path.apply(
        post_processing.choose_rss, axis=1
    )
    No_improved_Final = Final_results_users_all_path[
        Final_results_users_all_path['RSS_without_RIS'] == Final_results_users_all_path['Final_RSS']
    ]

    # Phase 5: merge remaining users for reclustering
    Data_algh3_clustering = pd.concat(
        [Rest_users_clustering_again, No_improved_Final],
        ignore_index=True
    )
    # cleanup columns
    cols_to_drop = [
        'RIS_position','RIS_orientation','RIS_name',
        'RSS_with_RIS','Final_RSS','Precoder',
        'Precoder_index','Cluster_ID'
    ]
    Data_algh3_clustering = Data_algh3_clustering.drop(columns=cols_to_drop, errors='ignore')
    Data_algh3_clustering = Data_algh3_clustering.rename(columns={'Rx_location':'Location_users'})

    # save output
    output_path = 'Dataframe_prepared_3_5GHz_T15_clustering_again.csv'
    Data_algh3_clustering.to_csv(output_path, index=False)

    return Data_algh3_clustering









