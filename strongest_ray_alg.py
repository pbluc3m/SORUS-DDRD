# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

# strongest_ray_alg.py

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from tqdm import tqdm
from sionna.rt import Transmitter, RIS
import re
from utils import lat_lon_to_cartesian
from post_processing import parse_RIS_orientation_deg
import ast
from sionna.rt import Transmitter, RIS
from post_processing import convert_bracket_to_parenthesis



def Strongest_Ray_RIS_Opt(
    scene,
    df1,
    df2,
    df3,
    df4,
    filtered_data,
    gob,
    txPower_dBm,
    ris_row_elem,
    ris_col_elem,
    ray_reflection,
    ray_diffraction,
    ray_scattering,
    UE_grid,
    num_ray_shooting,
    max_ray_bouncing
    
):
    
    """
    Compute coverage with RIS for four centroid-based DataFrames and return merged results.

    Inputs:
      - scene: Sionna scene object
      - df1, df2, df3, df4: pd.DataFrame, each with columns
          ['Tx_indice','Rx_indice','Tx_location','Rx_location',
           'Precoder','RIS_deployment_location','RIS_orientation',
           'point_A','point_B','RSS_rx']
      - filtered_data: full BS dataframe from which we take bearing (col 45) and downtilt (col 43)
      - gob: array of shape [num_rows, num_cols, N] of precoding vectors
      - txPower_dBm: scalar to add to RIS gain

    Returns:
      - final_results_df_centroids: merged pd.DataFrame of coverage results
    """
    def to_db(x):
        return 10 * tf.math.log(x) / tf.math.log(10.)

    # Prepare bearing/downtilt lists
    bearing_deg_list = list(filtered_data.iloc[:,45])
    downtilt_list    = list(filtered_data.iloc[:,43])

    # Storage
    results1 = []
    results2 = []
    results3 = []
    results4 = []

    # Helper to remove all Tx and RIS
    def clear_scene():
        for tx_name in list(scene.transmitters.keys()):
            scene.remove(tx_name)
        for ris_name in list(scene.ris.keys()):
            scene.remove(ris_name)

    # Core loop for one DataFrame
    def process_df(df, results_list, desc):
        clear_scene()
        tx_indices = df['Tx_indice'].unique()
        for tx_index in tqdm(tx_indices, desc=desc):
            df_tx = df[df['Tx_indice']==tx_index].sort_values('Rx_indice')
            for _, row in df_tx.iterrows():
                loc = row['RIS_deployment_location']
                if (loc is not None
                    and not (isinstance(loc,float) and np.isnan(loc))
                    and isinstance(loc, (list,tuple,np.ndarray))
                    and len(loc)==3
                    and not np.isnan(loc).any()):
                    # extract
                    Tx_i = row['Tx_indice']
                    Rx_i = row['Rx_indice']
                    Tx_loc = row['Tx_location']
                    Rx_loc = row['Rx_location']
                    r, c = row['Precoder']
                    prec = gob[r,c,:]
                    # antenna orientation
                    try:
                        bd = bearing_deg_list[Tx_i]
                        dt = downtilt_list[Tx_i]
                    except IndexError:
                        continue
                    tx_or = [bd*math.pi/180, dt*math.pi/180, 0]
                    # add Tx
                    tx_name = f"tx{Tx_i}"
                    tx = Transmitter(name=tx_name, position=Tx_loc, orientation=tx_or)
                    scene.add(tx)
                    # add RIS
                    ris_name = f"ris{Tx_i}"
                    ris_or = [row['RIS_orientation']*math.pi/180, 0, 0]
                    ris = RIS(name=ris_name,
                              position=tf.Variable(loc, dtype=tf.float32),
                              orientation=ris_or,
                              num_rows=ris_row_elem, num_cols=ris_col_elem, num_modes=1)
                    scene.add(ris)
                    # set RIS phase
                    scene.ris[ris_name].phase_gradient_reflector(
                        row['point_A'], row['point_B'])
                    # coverage_map
                    cm = scene.coverage_map(
                        num_samples=num_ray_shooting,
                        max_depth=max_ray_bouncing,
                        los=True, 
                        reflection=ray_reflection,
                        diffraction=ray_diffraction, 
                        scattering=ray_scattering,
                        ris=True,
                        combining_vec=None,
                        precoding_vec=prec,
                        cm_cell_size=UE_grid,
                        cm_orientation=[0,0,0],
                        cm_center=Rx_loc,
                        cm_size=[2,2]
                    )
                    

                    
                    p = cm.path_gain[0,0,0]
                    p_db = to_db(p) + txPower_dBm
                    results_list.append({
                        'Tx_index': Tx_i,
                        'Rx_index': Rx_i,
                        'Tx_location': Tx_loc,
                        'Rx_location': Rx_loc,
                        'RIS_name': ris_name,
                        'RIS_position': loc,
                        'RIS_orientation': ris_or,
                        'RSS_without_RIS': row['RSS_rx'],
                        'RSS_dB_RIS': p_db.numpy(),
                        'point_A': row['point_A'],
                        'point_B': row['point_B']
                    })
                    scene.remove(tx_name)
                    scene.remove(ris_name)

    # Process each dataframe
    process_df(df1, results1,  "Coverage df1")
    df_r1 = pd.DataFrame(results1)

    process_df(df2, results2, "Coverage df2")
    df_r2 = pd.DataFrame(results2)

    # Merge df1 & df2
    merged12 = pd.merge(df_r1, df_r2,
                       on=['Tx_index','Rx_index'],
                       how='outer',
                       suffixes=('_df1','_df2'))

    process_df(df3, results3, "Coverage df3")
    df_r3 = pd.DataFrame(results3)
    merged123 = pd.merge(merged12, df_r3,
                        on=['Tx_index','Rx_index'],
                        how='outer')

    process_df(df4, results4, "Coverage df4")
    df_r4 = pd.DataFrame(results4)
    final_results = pd.merge(merged123, df_r4,
                             on=['Tx_index','Rx_index'],
                             how='outer')

    return final_results



def BS_Beam_Opt(
    scene,
    csv_input_path,
    latitudes,
    longitudes,
    BS_height,
    bearing_deg,
    downtilt,
    num_rows,
    num_cols,
    gob,
    txPower_dBm,
    csv_output_path
):
    """
    Update precoder selection for centroids and save results to CSV.
    """
    # Load DataFrame
    Centroids_precoder_update = pd.read_csv(csv_input_path)

    # Remove existing transmitters
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)
       
    BS_coverage_dB = []

    # Loop over each row to update
    for idx, row in tqdm(Centroids_precoder_update.iterrows(), 
                     total=len(Centroids_precoder_update), 
                     desc="Precoder Optimization"):

        # Extract RIS deployment location and orientation
        ris_deployment_location = row['RIS_deployment_location']
        RIS_orientation_deg      = row['RIS_orientation']
        Tx_index                 = row['Tx_indice']

        # Clean up location string → numpy array
        try:
            clean_location = re.sub(r'[\[\]]', '', ris_deployment_location).split()
            ris_deployment_location = np.array([float(num) for num in clean_location])
        except (ValueError, TypeError):
            continue

        if (isinstance(ris_deployment_location, np.ndarray)
            and ris_deployment_location.shape == (3,)
            and not np.isnan(ris_deployment_location).any()):
            row['RIS_deployment_location'] = ris_deployment_location

        # Parse orientation
        if isinstance(RIS_orientation_deg, str):
            RIS_orientation_deg = parse_RIS_orientation_deg(RIS_orientation_deg)
        elif isinstance(RIS_orientation_deg, list):
            parsed = []
            for elem in RIS_orientation_deg:
                if isinstance(elem, tf.Tensor):
                    parsed.append(elem.numpy())
                else:
                    parsed.append(float(elem))
            RIS_orientation_deg = parsed

        ris_orientation_rad = RIS_orientation_deg
        ris_orientation_rad[1] -= math.radians(90)

        BS_coverage_dB = []

        # Transmitter parameters
        lat    = latitudes[Tx_index]
        lon    = longitudes[Tx_index]
        height = BS_height[Tx_index]
        bearing= bearing_deg[Tx_index]
        tilt   = downtilt[Tx_index]

        x, y = lat_lon_to_cartesian(lat, lon)
        tx_position    = [x, y, height]
        tx_name        = f"tx{Tx_index}"
        tx_orientation = [bearing*math.pi/180, tilt*math.pi/180, 0]

        tx = Transmitter(name=tx_name, position=tx_position, orientation=tx_orientation)
        scene.add(tx)

        if ris_deployment_location[2] < 3:
            ris_deployment_location[2] += 4

        # Compute coverage for each precoder
        for r in range(num_rows):
            for c in range(num_cols):
                precoding_vec = gob[r, c, :]
                cm = scene.coverage_map(
                    num_samples=int(4e6),
                    max_depth=4,
                    los=True,
                    reflection=True,
                    diffraction=False,
                    scattering=False,
                    ris=False,
                    combining_vec=None,
                    precoding_vec=precoding_vec,
                    cm_cell_size=[4,4],
                    cm_orientation=ris_orientation_rad,
                    cm_center=ris_deployment_location,
                    cm_size=[4,4]
                )
                Coverage_tensor = cm.path_gain
                gain_dB = 10 * tf.math.log(Coverage_tensor) / tf.math.log(10.0)
                BS_coverage_dB.append(gain_dB)

        # Stack or fill empty
        if not BS_coverage_dB:
            gain_dB_single_BS = tf.fill([1, num_rows, num_cols], float('-inf'))
        else:
            gain_dB_single_BS = tf.stack(BS_coverage_dB, axis=0)
            gain_dB_single_BS = tf.reshape(gain_dB_single_BS, [1, num_rows, num_cols])

        scene.remove(tx_name)

        # Concatenate across BS
        if not BS_coverage_dB:
            gain_dB = tf.fill([1, num_rows, num_cols], float('-inf'))
        else:
            gain_dB = tf.concat([gain_dB_single_BS], axis=0)

        flat = tf.reshape(gain_dB, [-1])
        max_gain = tf.reduce_max(flat)
        max_idx  = tf.argmax(flat).numpy()

        total_bs = gain_dB.shape[0]
        pr = gain_dB.shape[1]
        pc = gain_dB.shape[2]

        best_BS  = max_idx // (pr * pc)
        best_row = (max_idx % (pr * pc)) // pc
        best_col = max_idx % pc
        best_server = (int(best_row), int(best_col))

        Centroids_precoder_update.loc[idx, 'Best_RIS_power']  = max_gain.numpy()
        Centroids_precoder_update.loc[idx, 'Best_RIS_server'] = f"({best_server[0]}, {best_server[1]})"

    # Save to CSV
    Centroids_precoder_update.to_csv(csv_output_path, index=False)

    return Centroids_precoder_update

###############################################################################



def RSRP_RIS_UE(
    scene,
    Best_RIS_RSS_Configuration_BIRCH_Centroids,
    filtered_data,
    gob,
    chosen_poor_coverage_Dataframe_BIRCH,
    txPower_dBm,
    output_csv_path
):
    """
    Steered RIS coverage per cluster user, saving intermediate and final CSV.
    """
    def to_db(x):
        return 10 * tf.math.log(x) / tf.math.log(10.)

    results_cluster_df_centroids = []

    # Clear scene
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)
    for ris_name in list(scene.ris.keys()):
        scene.remove(ris_name)

    tx_indices = Best_RIS_RSS_Configuration_BIRCH_Centroids['Tx_indice'].unique()
    bearing_deg_list = list(filtered_data.iloc[:, 45])
    downtilt_list    = list(filtered_data.iloc[:, 43])

    for tx_index in tqdm(tx_indices, desc="Computing RSRP for outage UEs with RIS"):
        df_tx = Best_RIS_RSS_Configuration_BIRCH_Centroids[
            Best_RIS_RSS_Configuration_BIRCH_Centroids['Tx_indice'] == tx_index
        ].sort_values('Rx_indice')

        for _, row in df_tx.iterrows():
            ris_deployment_location = row['RIS_deployment_location']
            try:
                clean_location = re.sub(r'[\[\]]', '', ris_deployment_location).split()
                ris_deployment_location = np.array([float(num) for num in clean_location])
            except (ValueError, TypeError):
                continue

            if not (isinstance(ris_deployment_location, np.ndarray) and
                    len(ris_deployment_location) == 3 and
                    not np.isnan(ris_deployment_location).any()):
                continue

            selected_row = row
            Tx_index = selected_row['Tx_indice']
            Rx_index = selected_row['Rx_indice']

            Tx_location = ast.literal_eval(selected_row['Tx_location'])
            Tx_location = np.array(Tx_location, dtype=np.float32)
            Rx_location = selected_row['Rx_location']

            precoder_index = eval(selected_row['Precoder'])
            row_precoder, col_precoder = precoder_index
            precoding_vec = gob[row_precoder, col_precoder, :]

            try:
                bearing_deg = bearing_deg_list[Tx_index]
                downtilt_deg = downtilt_list[Tx_index]
            except IndexError:
                continue

            tx_orientation = [bearing_deg * (math.pi/180), downtilt_deg * (math.pi/180), 0]
            tx_name = f"tx{Tx_index}"
            tx = Transmitter(name=tx_name, position=Tx_location.tolist(), orientation=tx_orientation)
            scene.add(tx)

            ris_name = f"ris{Tx_index}"
            RIS_orientation_deg = selected_row['RIS_orientation']
            if isinstance(RIS_orientation_deg, str):
                RIS_orientation_deg = parse_RIS_orientation_deg(RIS_orientation_deg)
            elif isinstance(RIS_orientation_deg, list):
                parsed = []
                for elem in RIS_orientation_deg:
                    if isinstance(elem, tf.Tensor):
                        parsed.append(elem.numpy())
                    else:
                        parsed.append(float(elem))
                RIS_orientation_deg = parsed

            ris_orientation_rad = RIS_orientation_deg
            if ris_deployment_location[2] < 3:
                ris_deployment_location[2] += 4

            cluster_birch = Rx_index
            cluster_data = chosen_poor_coverage_Dataframe_BIRCH[
                chosen_poor_coverage_Dataframe_BIRCH['cluster_birch'] == cluster_birch
            ]

            point_A = ast.literal_eval(selected_row['Point_A'])
            point_A = np.array(point_A, dtype=np.float32)

            for _, cluster_row in cluster_data.iterrows():
                cluster_rx_index = cluster_row.get('user_index') or cluster_row.get('Rx_index')
                cluster_rx_location = cluster_row['location_users']

                ris = RIS(
                    name=ris_name,
                    position=tf.Variable(ris_deployment_location, dtype=tf.float32),
                    orientation=ris_orientation_rad,
                    num_rows=260,
                    num_cols=260,
                    num_modes=1
                )
                scene.add(ris)
                ris.phase_gradient_reflector(point_A, cluster_rx_location)

                cm_ris_cluster = scene.coverage_map(
                    num_samples=int(10e6),
                    max_depth=4,
                    los=True,
                    reflection=True,
                    diffraction=False,
                    scattering=False,
                    ris=True,
                    combining_vec=None,
                    precoding_vec=precoding_vec,
                    cm_cell_size=[2,2],
                    cm_orientation=[0,0,0],
                    cm_center=cluster_rx_location,
                    cm_size=[2,2]
                )

                Coverage_tensor_RIS_cluster = cm_ris_cluster.path_gain
                power_cluster = Coverage_tensor_RIS_cluster[0,0,0]
                power_db_cluster = to_db(power_cluster) + txPower_dBm

                RSS_rx_without_RIS = cluster_row.get('RSS_dBm') or cluster_row.get('RSS_without_RIS')

                cluster_result = {
                    'Cluster_ID': cluster_birch,
                    'Tx_index': Tx_index,
                    'Tx_location': Tx_location.tolist(),
                    'Rx_index': cluster_rx_index,
                    'Rx_location': cluster_rx_location,
                    'Precoder_index': precoder_index,
                    'Point_A': point_A.tolist(),
                    'RIS_name': ris_name,
                    'RIS_position': ris_deployment_location.tolist(),
                    'RIS_orientation': ris_orientation_rad,
                    'RSS_without_RIS': RSS_rx_without_RIS,
                    'RSS_with_RIS': power_db_cluster.numpy(),
                }
                results_cluster_df_centroids.append(cluster_result)

                scene.remove(ris_name)

            scene.remove(tx_name)

        # save intermediate results
        pd.DataFrame(results_cluster_df_centroids).to_csv(output_csv_path, index=False)

    results_cluster_df_centroids = pd.DataFrame(results_cluster_df_centroids)
    results_cluster_df_centroids.to_csv(output_csv_path, index=False)
    return results_cluster_df_centroids











