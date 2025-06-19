# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


# all_ray_alg.py

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from tqdm import tqdm
from sionna.rt import Transmitter, RIS

def All_Ray_RIS_Opt(
    scene,
    df_paths_centroids_BIRCH: pd.DataFrame,
    df_paths_centroids_BIRCH2: pd.DataFrame,
    gob: np.ndarray,
    filtered_data: pd.DataFrame,
    txPower_dBm: float,
    output_csv_path: str
) -> pd.DataFrame:
    """
    Compute RIS-assisted coverage for full and second-reflection path DataFrames,
    merge results, save to CSV, and return the final DataFrame.

    Inputs:
      - scene: Sionna RT scene object
      - df_paths_centroids_BIRCH: first DataFrame with columns
          ['Tx_indice','Rx_indice','Tx_location','Rx_location',
           'Precoder','RIS_deployment_location','RIS_orientation',
           'point_A','point_B','RSS_rx']
      - df_paths_centroids_BIRCH2: second DataFrame (based on df2) with same columns
      - gob: precoding codebook array [num_rows, num_cols, ...]
      - filtered_data: original BS DataFrame (for bearing & downtilt at cols 45 & 43)
      - txPower_dBm: transmit power in dBm to add to RIS gain
      - output_csv_path: path to write merged results CSV

    Returns:
      - final_results_df: merged pd.DataFrame
    """
    def to_db(x):
        return 10 * tf.math.log(x) / tf.math.log(10.)

    results_df_coverage = []
    results_df2_coverage = []

    # clear scene
    for name in list(scene.transmitters.keys()):
        scene.remove(name)
    for name in list(scene.ris.keys()):
        scene.remove(name)

    # prepare bearing & downtilt lists
    bearing_deg_list = list(filtered_data.iloc[:,45])
    downtilt_list    = list(filtered_data.iloc[:,43])

    # 1) process df_paths_centroids_BIRCH
    for tx_index in tqdm(df_paths_centroids_BIRCH['Tx_indice'].unique(),
                          desc="All Ray Algorithm RIS deployment"):
        df_tx = df_paths_centroids_BIRCH[
            df_paths_centroids_BIRCH['Tx_indice']==tx_index
        ].sort_values('Rx_indice')
        for _, row in df_tx.iterrows():
            loc = row['RIS_deployment_location']
            if (loc is None
                or not isinstance(loc, (list,tuple,np.ndarray))
                or len(loc)!=3
                or np.isnan(loc).any()):
                continue
            Tx_i = row['Tx_indice']
            Rx_i = row['Rx_indice']
            Tx_loc = row['Tx_location']
            Rx_loc = row['Rx_location']
            r,c    = row['Precoder']
            prec   = gob[r,c,:]
            try:
                bd = bearing_deg_list[Tx_i]; dt = downtilt_list[Tx_i]
            except IndexError:
                continue
            tx_or = [bd*math.pi/180, dt*math.pi/180, 0]
            tx_name = f"tx{Tx_i}"
            scene.add(Transmitter(name=tx_name,
                                  position=Tx_loc,
                                  orientation=tx_or))
            ris_name = f"ris{Tx_i}"
            ris_or = [row['RIS_orientation']*math.pi/180, 0, 0]
            scene.add(RIS(name=ris_name,
                          position=tf.Variable(loc, dtype=tf.float32),
                          orientation=ris_or,
                          num_rows=260, num_cols=260, num_modes=1))
            scene.ris[ris_name].phase_gradient_reflector(
                row['point_A'], row['point_B'])
            cm = scene.coverage_map(
                num_samples=int(10e6), max_depth=4,
                los=True, reflection=True,
                diffraction=False, scattering=False,
                ris=True, combining_vec=None,
                precoding_vec=prec,
                cm_cell_size=[2,2],
                cm_orientation=[0,0,0],
                cm_center=Rx_loc,
                cm_size=[2,2]
            )
            p = cm.path_gain[0,0,0]
            p_db = to_db(p) + txPower_dBm
            results_df_coverage.append({
                'Tx_index': Tx_i,
                'Tx_location': Tx_loc,
                'Rx_index': Rx_i,
                'Rx_location': Rx_loc,
                'Precoder': (r,c),
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

    df_res1 = pd.DataFrame(results_df_coverage)

    # 2) process df_paths_centroids_BIRCH2
    for tx_index in tqdm(df_paths_centroids_BIRCH2['Tx_indice'].unique(),
                          desc="df_paths_centroids_BIRCH2"):
        df_tx2 = df_paths_centroids_BIRCH2[
            df_paths_centroids_BIRCH2['Tx_indice']==tx_index
        ].sort_values('Rx_indice')
        for _, row in df_tx2.iterrows():
            loc2 = row['RIS_deployment_location']
            if (loc2 is None
                or not isinstance(loc2, (list,tuple,np.ndarray))
                or len(loc2)!=3
                or np.isnan(loc2).any()):
                continue
            Tx_i = row['Tx_indice']
            Rx_i = row['Rx_indice']
            Tx_loc = row['Tx_location']
            Rx_loc = row['Rx_location']
            r,c    = row['Precoder']
            prec   = gob[r,c,:]
            try:
                bd = bearing_deg_list[Tx_i]; dt = downtilt_list[Tx_i]
            except IndexError:
                continue
            tx_or = [bd*math.pi/180, dt*math.pi/180, 0]
            tx_name = f"tx{Tx_i}"
            scene.add(Transmitter(name=tx_name,
                                  position=Tx_loc,
                                  orientation=tx_or))
            ris_name = f"ris{Tx_i}"
            ris_or = [row['RIS_orientation']*math.pi/180, 0, 0]
            scene.add(RIS(name=ris_name,
                          position=tf.Variable(loc2, dtype=tf.float32),
                          orientation=ris_or,
                          num_rows=260, num_cols=260, num_modes=1))
            scene.ris[ris_name].phase_gradient_reflector(
                row['point_A'], Rx_loc)
            cm2 = scene.coverage_map(
                num_samples=int(10e6), max_depth=4,
                los=True, reflection=True,
                diffraction=False, scattering=False,
                ris=True, combining_vec=None,
                precoding_vec=prec,
                cm_cell_size=[2,2],
                cm_orientation=[0,0,0],
                cm_center=Rx_loc,
                cm_size=[2,2]
            )
            p2 = cm2.path_gain[0,0,0]
            p2_db = to_db(p2) + txPower_dBm
            results_df2_coverage.append({
                'Tx_index': Tx_i,
                'Rx_index': Rx_i,
                'Precoder': (r,c),
                'RIS_position_df2': loc2,
                'RIS_orientation_df2': ris_or,
                'RSS_RIS_df2': p2_db.numpy(),
                'point_A': row['point_A'],
                'point_B': row['point_B']
            })
            scene.remove(tx_name)
            scene.remove(ris_name)

    df_res2 = pd.DataFrame(results_df2_coverage)

    # 3) merge & save
    final_results_df = pd.merge(
        df_res1, df_res2,
        on=['Tx_index','Rx_index'],
        how='outer'
    )
    final_results_df.to_csv(output_csv_path, index=False)
    return final_results_df
