# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import trimesh
import numpy as np
import pandas as pd
import ast
import math
import tensorflow as tf
from tqdm import tqdm
from sionna.rt import Transmitter, RIS
import re


def LoS_UE_RIS(
    Worst_UEs_Data: pd.DataFrame,
    unique_RIS_list: list,
    ply_folder: str
) -> pd.DataFrame:
    """
    For each user in Worst_UEs_Data, find the first RIS in unique_RIS_list
    that has line-of-sight (LoS) to the user, using the 3D meshes in ply_folder.
    Returns only those user-RIS pairs that are LoS.
    """

    def load_meshes(folder: str):
        meshes = []
        for f in os.listdir(folder):
            if f.endswith(".ply"):
                meshes.append(trimesh.load(os.path.join(folder, f)))
        return meshes

    def parse_location(loc):
        if isinstance(loc, str):
            loc = loc.strip()
            try:
                return tuple(map(float, ast.literal_eval(loc)))
            except (ValueError, SyntaxError):
                parts = loc.replace('[','').replace(']','').split()
                return tuple(map(float, parts))
        return loc

    def is_in_los(ris_pos, user_pos, meshes):
        origin = np.array(ris_pos, dtype=np.float64)
        target = np.array(user_pos, dtype=np.float64)
        direction = target - origin
        length = np.linalg.norm(direction)
        if length == 0:
            return False
        direction /= length
        for mesh in meshes:
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=[origin],
                ray_directions=[direction]
            )
            if len(locs) > 0:
                dists = np.linalg.norm(locs - origin, axis=1)
                if np.any(dists < length):
                    return False
        return True

    meshes = load_meshes(ply_folder)
    los_results = []

    for _, user_row in Worst_UEs_Data.iterrows():
        user_loc = parse_location(user_row['Rx_location'])
        best_loc = None
        best_ori = None

        for ris_loc, ris_ori in unique_RIS_list:
            rl = parse_location(ris_loc)
            if is_in_los(rl, user_loc, meshes):
                best_loc = rl
                best_ori = ris_ori
                break

        los_results.append({
            'Rx_index':       user_row['Rx_index'],
            'Rx_location':    user_loc,
            'RIS_location':   best_loc if best_loc is not None else np.nan,
            'RIS_orientation': best_ori if best_ori is not None else np.nan
        })

    LoS_Results = pd.DataFrame(los_results)
    Filtered_LoS_Results = LoS_Results.dropna(subset=['RIS_location']).reset_index(drop=True)
    return Filtered_LoS_Results


def LoS_RIS_Tx(
    Filtered_LoS_Results: pd.DataFrame,
    Coverage_user_phase1: pd.DataFrame,
    ply_folder: str
) -> pd.DataFrame:
    """
    For each RIS in Filtered_LoS_Results, find which transmitters (from Coverage_user_phase1)
    have line-of-sight (LoS) to it using the environment meshes in ply_folder.
    Returns the DataFrame with an expanded 'LoS_Tx_indices' column.
    """

    def parse_tx_location(location):
        if isinstance(location, str):
            location = location.strip()
            try:
                return tuple(map(float, ast.literal_eval(location)))
            except (ValueError, SyntaxError):
                parts = location.replace('[','').replace(']','').split()
                return tuple(map(float, parts))
        return location

    def load_meshes(folder):
        meshes = []
        for f in os.listdir(folder):
            if f.endswith(".ply"):
                meshes.append(trimesh.load(os.path.join(folder, f)))
        return meshes

    def parse_location(location):
        if isinstance(location, str):
            location = location.strip()
            try:
                return tuple(map(float, ast.literal_eval(location)))
            except (ValueError, SyntaxError):
                parts = location.replace('[','').replace(']','').split()
                return tuple(map(float, parts))
        return location

    def is_in_los(ris_position, tx_position, meshes):
        origin = np.array(parse_location(ris_position), dtype=np.float64)
        target = np.array(parse_location(tx_position), dtype=np.float64)
        direction = target - origin
        length = np.linalg.norm(direction)
        if length == 0:
            return False
        direction /= length
        for mesh in meshes:
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=[origin], ray_directions=[direction]
            )
            if len(locs) > 0:
                dists = np.linalg.norm(locs - origin, axis=1)
                if np.any(dists < length):
                    return False
        return True

    # Prepare unique Tx list
    unique_Tx_data = Coverage_user_phase1[['Tx_index', 'Tx_location']].drop_duplicates().reset_index(drop=True)
    unique_Tx_data['Tx_location'] = unique_Tx_data['Tx_location'].apply(parse_tx_location)
    unique_Tx_list = unique_Tx_data.values.tolist()

    # Load environment meshes
    meshes = load_meshes(ply_folder)

    # Initialize the LoS_Tx_indices column
    Filtered_LoS_Results['LoS_Tx_indices'] = None

    # For each RIS, find all Tx in LoS
    for idx, ris_row in Filtered_LoS_Results.iterrows():
        ris_loc = ris_row['RIS_location']
        los_txs = []
        for tx_index, tx_loc in unique_Tx_list:
            if is_in_los(ris_loc, tx_loc, meshes):
                los_txs.append(tx_index)
        Filtered_LoS_Results.at[idx, 'LoS_Tx_indices'] = los_txs if los_txs else np.nan

    # Explode to one row per (RIS, Tx) pair
    Filtered_LoS_Results = Filtered_LoS_Results.explode('LoS_Tx_indices').reset_index(drop=True)
    return Filtered_LoS_Results




def LoS_precoder_updated(
    scene,
    file_path: str,
    latitudes,
    longitudes,
    BS_height,
    bearing_deg,
    downtilt,
    num_rows: int,
    num_cols: int,
    gob,
    parse_RIS_orientation_deg,
    lat_lon_to_cartesian,
    output_file_path: str = "Final_Filtered_LoS_Results_precoder_updated.csv"
):
    """
    Reads Final_Filtered_LoS_Results, de-duplicates by RIS and Tx, then for each pair
    tests all precoders to find the best RIS coverage. Saves updated CSV.
    """
    # Load data
    Final_Filtered_LoS_Results = pd.read_csv(file_path)

    # Convert RIS_location to tuple
    Final_Filtered_LoS_Results['RIS_location'] = Final_Filtered_LoS_Results['RIS_location'].apply(
        lambda x: tuple(eval(x)) if isinstance(x, str) else tuple(x)
    )

    # Drop duplicates
    Unique_RIS_Tx = Final_Filtered_LoS_Results.drop_duplicates(subset=['RIS_location', 'Tx_indice'])
    Unique_RIS_Tx = Unique_RIS_Tx.reset_index(drop=True)

    print('===================================')
    print('Updating precoder for deployed RIS in dataframe...')
    print('===================================')

    # Remove existing transmitters
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)

    BS_coverage_dB = []

    Choosing_best_precoder = False

    if Choosing_best_precoder == True:

        for idx, row in tqdm(Unique_RIS_Tx.iterrows(),
                             total=len(Unique_RIS_Tx),
                             desc="Precoder Optimization"):

            # Extract RIS deployment location and orientation from the DataFrame
            ris_deployment_location = row['RIS_location']
            RIS_orientation_deg = row['RIS_orientation']
            Tx_index = row['Tx_indice']

            # Parse RIS_orientation_deg
            if isinstance(RIS_orientation_deg, str):
                RIS_orientation_deg = parse_RIS_orientation_deg(RIS_orientation_deg)
            elif isinstance(RIS_orientation_deg, list):
                # Extract numerical values, including from tensors
                parsed_orientation = []
                for elem in RIS_orientation_deg:
                    if isinstance(elem, tf.Tensor):
                        parsed_orientation.append(elem.numpy())
                    else:
                        parsed_orientation.append(float(elem))
                RIS_orientation_deg = parsed_orientation

            # Convert RIS orientation to radians
            ris_orientation_rad = RIS_orientation_deg

            ris_orientation_rad[1] -= math.radians(90)

            BS_coverage_dB = []

            lat = latitudes[Tx_index]
            lon = longitudes[Tx_index]
            height = BS_height[Tx_index]
            bearing = bearing_deg[Tx_index]
            tilt = downtilt[Tx_index]

            x, y = lat_lon_to_cartesian(lat, lon)
            tx_position = [x, y, height]  # Using the actual height from the dataset
            tx_name = f"tx{Tx_index}"  # Generate a unique name for each transmitter
            tx_orientation = [bearing * (math.pi / 180), (tilt * math.pi / 180), 0]  # Incorporating bearing and downtilt into the orientation

            # Create and add the transmitter to the scene
            tx = Transmitter(name=tx_name, position=tx_position, orientation=tx_orientation)
            scene.add(tx)

            # Initialize a list to collect coverage maps for all precoders for this BS
            coverage_list_dB_bs = []

            # Nested loop for precoder configuration
            for row in range(num_rows):
                for col in tqdm(range(num_cols), desc=f"Processing precoder cols for row {num_rows}"):
                    precoding_vec = gob[row, col, :]  # Select the precoding vector

                    # Compute the coverage map for the current BS and precoder
                    cm_RIS_power = scene.coverage_map(num_samples=int(5e6),
                                                      max_depth=3,
                                                      los=True,
                                                      reflection=True,
                                                      diffraction=False,
                                                      scattering=False,
                                                      ris=False,
                                                      combining_vec=None,
                                                      precoding_vec=precoding_vec,
                                                      cm_cell_size=[6, 6],
                                                      cm_orientation=ris_orientation_rad,
                                                      cm_center=ris_deployment_location,
                                                      cm_size=[6, 6])

                    Coverage_tensor = cm_RIS_power.path_gain
                    gain_dB_single_precoder = 10 * tf.math.log(Coverage_tensor) / tf.math.log(10.0)

                    # Append the coverage map for this precoder
                    coverage_list_dB_bs.append(gain_dB_single_precoder)

            # Check if the coverage_list_dB_bs is empty before stacking
            if not coverage_list_dB_bs:
                # Create a placeholder tensor with -inf if no coverage maps exist
                gain_dB_single_BS = tf.fill([1, num_rows, num_cols], float('-inf'))
            else:
                # Combine coverage maps across all precoders for this BS
                gain_dB_single_BS = tf.stack(coverage_list_dB_bs, axis=0)  # Shape: [num_rows, num_cols, ...]
                gain_dB_single_BS = tf.reshape(gain_dB_single_BS, [1, num_rows, num_cols])  # Final shape: [1, num_rows, num_cols]

            # Append to global coverage lists
            BS_coverage_dB.append(gain_dB_single_BS)

            # Remove transmitter to start again with a new one
            scene.remove(tx_name)

            if not BS_coverage_dB:
                # Create a placeholder tensor with -inf if no BS coverage data exists
                gain_dB = tf.fill([1, num_rows, num_cols], float('-inf'))
            else:
                # Concatenate the results to form the final tensor
                gain_dB = tf.concat(BS_coverage_dB, axis=0)

            # Flatten gain_dB to find the strongest value and corresponding indices
            flat_gain_dB = tf.reshape(gain_dB, [-1])  # Flatten to 1D tensor

            # Find the maximum gain and its index
            max_gain = tf.reduce_max(flat_gain_dB)  # Strongest value
            max_index = tf.argmax(flat_gain_dB).numpy()  # Index of the strongest value

            # Convert the flat index back to (BS, row, column) indices
            precoder_row = gain_dB.shape[1]
            precoder_col = gain_dB.shape[2]

            best_row = (max_index % (precoder_row * precoder_col)) // precoder_col  # Row index
            best_col = max_index % precoder_col  # Column index

            # Prepare the result tuple for the best server
            best_server = (int(best_row), int(best_col))

            # Update the DataFrame with the results
            Unique_RIS_Tx.loc[idx, 'Best_RIS_power'] = max_gain.numpy()
            Unique_RIS_Tx.at[idx, 'Best_RIS_server'] = f"({best_server[0]}, {best_server[1]})"

        Unique_RIS_Tx.to_csv(output_file_path, index=False)

    return Unique_RIS_Tx



def LoS_data_RIS_UE(
    file_path_precoder_updated: str,
    file_path_UE_RIS: str,
    output_file: str = "Final_LoS_UE_RIS.csv"
) -> pd.DataFrame:
    """
    Post-process the per-RIS best-precoder data and the LoS results,
    to assign each UE its final RIS precoder.

    Parameters
    ----------
    file_path_precoder_updated : str
        CSV with columns ['RIS_location','Tx_indice','Best_RIS_power','Best_RIS_server',…]
    file_path_UE_RIS : str
        CSV with columns ['Rx_index','Rx_location','RIS_location','RIS_orientation',…]
    output_file : str, optional
        Where to write the final UE→RIS precoder assignments, by default "Final_LoS_UE_RIS.csv"

    Returns
    -------
    pd.DataFrame
        The filtered UE table with a new 'Precoder' column.
    """
    # read in
    LoS_RIS_Update_Precoder = pd.read_csv(file_path_precoder_updated)
    LoS_UE_RIS                  = pd.read_csv(file_path_UE_RIS)

    # ensure RIS_location is hashable tuple
    LoS_RIS_Update_Precoder['RIS_location'] = LoS_RIS_Update_Precoder['RIS_location'].apply(
        lambda x: tuple(eval(x)) if isinstance(x, str) else tuple(x)
    )
    # pick best per RIS_location
    idx = LoS_RIS_Update_Precoder.groupby('RIS_location')['Best_RIS_power'].idxmax()
    LoS_RIS_Update_Precoder = LoS_RIS_Update_Precoder.loc[idx].reset_index(drop=True)

    # build mapping RIS_location→Best_RIS_server
    ris_server_mapping = LoS_RIS_Update_Precoder.set_index('RIS_location')['Best_RIS_server'].to_dict()

    # align types in LoS_UE_RIS
    LoS_UE_RIS['RIS_location'] = LoS_UE_RIS['RIS_location'].apply(
        lambda x: tuple(eval(x)) if isinstance(x, str) else tuple(x)
    )
    # map in
    LoS_UE_RIS['Precoder'] = LoS_UE_RIS['RIS_location'].map(ris_server_mapping)

    # now drop any rows where we didn’t find a valid (RIS_location, Tx_indice) pair
    valid_pairs = set(
        LoS_RIS_Update_Precoder[['RIS_location','Tx_indice']]
        .itertuples(index=False, name=None)
    )
    LoS_UE_RIS = LoS_UE_RIS[
        LoS_UE_RIS.apply(
            lambda row: (row['RIS_location'], row['Tx_indice']) in valid_pairs,
            axis=1
        )
    ].reset_index(drop=True)

    LoS_UE_RIS.to_csv(output_file, index=False)
    return LoS_UE_RIS


def Re_assocciation_RSRP_RIS(
    scene,
    LoS_UE_RIS: pd.DataFrame,
    filtered_data: pd.DataFrame,
    latitudes,
    longitudes,
    BS_height,
    bearing_deg,
    downtilt,
    gob,
    output_file: str = "Coverage_users_RIS750_cluster_BRICH_10GHz_precoder_LoS_06_05_2025.csv"
) -> pd.DataFrame:
    """
    For each UE in LoS_UE_RIS, deploy its RIS, shoot a coverage map with its assigned precoder,
    and save the per-UE RSS results.
    """

    def to_db(x):
        return 10 * tf.math.log(x) / tf.math.log(10.0)

    results_df_coverage = []

    # Clear scene
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)
    for ris_name in list(scene.ris.keys()):
        scene.remove(ris_name)

    tx_indices = LoS_UE_RIS['Tx_indice'].unique()
    bearing_deg_list = list(filtered_data.iloc[:, 45])
    downtilt_list     = list(filtered_data.iloc[:, 43])

    for tx_index in tqdm(tx_indices, desc="Computing Coverage map with LoS_UE_RIS"):
        df_tx = LoS_UE_RIS[LoS_UE_RIS['Tx_indice']==tx_index].sort_values('Rx_indice')
        for _, row in df_tx.iterrows():
            # parse positions
            ris_loc = tuple(map(float, ast.literal_eval(row['RIS_location'])))
            if not (isinstance(ris_loc, (list,tuple,np.ndarray)) and len(ris_loc)==3):
                continue
            ris_loc = np.array(ris_loc, dtype=np.float32)

            Tx_loc = tuple(map(float, row['Tx_location'].replace('[','').replace(']','').split()))
            Tx_loc = np.array(Tx_loc, dtype=np.float32)
            Rx_loc = tuple(map(float, ast.literal_eval(row['Rx_location'])))
            Rx_loc = np.array(Rx_loc, dtype=np.float32)

            try:
                b_deg = bearing_deg_list[tx_index]
                d_deg = downtilt_list[tx_index]
            except IndexError:
                print(f"Bearing/downtilt missing for Tx {tx_index}, skipping")
                continue

            tx_orientation = [b_deg*np.pi/180, d_deg*np.pi/180, 0]
            tx = Transmitter(name=f"tx{tx_index}", position=Tx_loc.tolist(), orientation=tx_orientation)
            scene.add(tx)

            ris_orientation = tuple(map(float, ast.literal_eval(row['RIS_orientation'])))
            ris_orientation = np.array(ris_orientation, dtype=np.float32)
            ris = RIS(name=f"ris{tx_index}",
                      position=tf.Variable(ris_loc, dtype=tf.float32),
                      orientation=ris_orientation,
                      num_rows=750, num_cols=750, num_modes=1)
            scene.add(ris)

            # phase-grade
            ris.phase_gradient_reflector(Tx_loc, Rx_loc)

            # pick precoder
            pc_idx = eval(row['Precoder'])
            precoding_vec = gob[pc_idx[0], pc_idx[1], :]

            # coverage
            cm = scene.coverage_map(num_samples=int(1e7),
                                    max_depth=4, los=True,
                                    reflection=True, diffraction=False,
                                    scattering=False, ris=True,
                                    combining_vec=None,
                                    precoding_vec=precoding_vec,
                                    cm_cell_size=[2,2],
                                    cm_orientation=[0,0,0],
                                    cm_center=Rx_loc.tolist(),
                                    cm_size=[2,2])
            power = cm.path_gain[0,0,0]
            power_db = to_db(power) + 8.85

            results_df_coverage.append({
                'Tx_index':         tx_index,
                'Tx_location':      Tx_loc.tolist(),
                'Rx_index':         row['Rx_indice'],
                'Rx_location':      Rx_loc.tolist(),
                'RIS_name':         f"ris{tx_index}",
                'RIS_position':     ris_loc.tolist(),
                'RIS_orientation':  ris_orientation.tolist(),
                'RSS_without_RIS':  row['RSS_without_RIS'],
                'RSS_dB_RIS':       power_db.numpy(),
                'Precoder':         pc_idx,
                'point_A':          Tx_loc.tolist(),
                'point_B':          Rx_loc.tolist(),
            })

            scene.remove(f"tx{tx_index}")
            scene.remove(f"ris{tx_index}")

    df = pd.DataFrame(results_df_coverage)
    df.to_csv(output_file, index=False)
    return df

def RIS_UE_RSRP_Data(
    final_data: pd.DataFrame,
    coverage_users_los_csv: str,
    output_file: str = "Final_Coverage_results_RSRP_RIS260_07_05_2025.csv"
) -> pd.DataFrame:
    """
    Take a Final_data DataFrame and a CSV of Coverage_users_LoS,
    compute Final_RSS and update LoS_RSS and related columns, then save.
    """
    # Load
    Coverage_users_LoS = pd.read_csv(coverage_users_los_csv)

    Coverage_users_LoS.rename(columns={'Precoder': 'Precoder_index'}, inplace=True)
    Coverage_users_LoS.rename(columns={'RSS_dB_RIS': 'RSS_with_RIS'},    inplace=True)

    # Function to apply the logic
    def choose_rss(row):
        if row['RSS_with_RIS'] > row['RSS_without_RIS']:
            return row['RSS_with_RIS']
        elif row['RSS_with_RIS'] == -np.inf:
            return row['RSS_without_RIS']
        elif row['RSS_with_RIS'] < row['RSS_without_RIS']:
            return row['RSS_without_RIS']
        else:  # Covers the case where RSS_with_RIS == RSS_without_RIS
            return row['RSS_with_RIS']

    Coverage_users_LoS['Final_RSS'] = Coverage_users_LoS.apply(choose_rss, axis=1)

    # Step 1: Create mappings
    los_rss_mapping        = dict(zip(Coverage_users_LoS['Rx_index'], Coverage_users_LoS['Final_RSS']))
    tx_index_mapping       = dict(zip(Coverage_users_LoS['Rx_index'], Coverage_users_LoS['Tx_index']))
    tx_location_mapping    = dict(zip(Coverage_users_LoS['Rx_index'], Coverage_users_LoS['Tx_location']))
    ris_position_mapping   = dict(zip(Coverage_users_LoS['Rx_index'], Coverage_users_LoS['RIS_position']))
    ris_orientation_mapping= dict(zip(Coverage_users_LoS['Rx_index'], Coverage_users_LoS['RIS_orientation']))
    precoder_mapping       = dict(zip(Coverage_users_LoS['Rx_index'], Coverage_users_LoS['Precoder_index']))
    point_a_dict_p2        = dict(zip(Coverage_users_LoS['Rx_index'], Coverage_users_LoS['point_A']))

    # Step 2: Initialize LoS_RSS
    final_data['LoS_RSS'] = final_data['phase2_all_path']

    # Step 3: Update per-Rx_index
    for idx, row in final_data.iterrows():
        rx = row['Rx_index']
        if rx in los_rss_mapping:
            fr = los_rss_mapping[rx]
            p2 = row['phase2_all_path']
            final_data.at[idx, 'LoS_RSS'] = max(fr, p2)
            if fr > p2:
                final_data.at[idx, 'Tx_index']      = tx_index_mapping.get(rx, row['Tx_index'])
                final_data.at[idx, 'Tx_location']   = tx_location_mapping.get(rx, row['Tx_location'])
                final_data.at[idx, 'RIS_position']  = ris_position_mapping.get(rx, row['RIS_position'])
                final_data.at[idx, 'RIS_orientation']= ris_orientation_mapping.get(rx, row['RIS_orientation'])
                final_data.at[idx, 'Precoder_index'] = precoder_mapping.get(rx, row['Precoder_index'])
                final_data.at[idx, 'Point_A']        = point_a_dict_p2.get(rx, row['Point_A'])

    # Step 4: move LoS_RSS to last column
    cols = [c for c in final_data.columns if c!='LoS_RSS'] + ['LoS_RSS']
    final_data = final_data[cols]

    # Formatter
    def format_with_commas(val):
        if isinstance(val, str):
            clean_val = val.strip('[]')
            parts = re.split(r'[,\s]+', clean_val)
            try:
                parts = [str(float(p)) for p in parts if p]
                return f"[{', '.join(parts)}]"
            except ValueError:
                return val
        return val

    final_data['RIS_orientation'] = final_data['RIS_orientation'].apply(format_with_commas)

    # Save
    final_data.to_csv(output_file, index=False)
    return final_data












