# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import math
import tensorflow as tf
from tqdm import tqdm
import utils
from sionna.rt import Transmitter

def run_coverage_maps(scene,
                      latitudes,
                      longitudes,
                      BS_height,
                      bearing_deg,
                      downtilt,
                      gob,
                      num_rows,
                      num_cols,
                      max_ray_bouncing,
                      ray_reflection,
                      ray_diffraction,
                      ray_los,
                      rt_ris,
                      ray_scattering,
                      UE_grid,
                      num_ray_shooting,
                      run_maps=False):
    """
    Run Sionna coverage maps for all transmitters and save results.
    """
    tx_orientations = {}
    coverage_list_dB = []
    coverage_location_list = []

    if run_maps:
        for index, (lat, lon, height, bearing, tilt) in enumerate(tqdm(zip(latitudes, longitudes, BS_height, bearing_deg, downtilt)), start=1):
            
            x, y = utils.lat_lon_to_cartesian(lat, lon)
            tx_position = [x, y, height]
            tx_name = f"tx{index}"
            tx_orientation = [bearing * (math.pi/180), tilt * (math.pi/180), 0]
            tx_orientations[tx_name] = tx_orientation

            tx = Transmitter(name=tx_name, position=tx_position, orientation=tx_orientation)
            scene.add(tx)

            coverage_list_dB_bs = []

            for row in range(num_rows):
                for col in range(num_cols):
                    precoding_vec = gob[row, col, :]
                    cm = scene.coverage_map(max_depth = max_ray_bouncing,
                                            reflection=ray_reflection,
                                            diffraction=ray_diffraction,
                                            los=ray_los,
                                            ris=rt_ris,
                                            scattering=ray_scattering,
                                            cm_cell_size=UE_grid,
                                            combining_vec=None,
                                            precoding_vec=precoding_vec,
                                            num_samples=num_ray_shooting)

                    coverage_location = cm.cell_centers
                    Coverage_tensor = cm.path_gain
                    gain_dB_single_precoder = 10 * tf.math.log(Coverage_tensor) / tf.math.log(10.0)
                    coverage_list_dB_bs.append(gain_dB_single_precoder)

            gain_dB_single_BS = tf.stack(coverage_list_dB_bs, axis=0)
            gain_dB_single_BS = tf.reshape(gain_dB_single_BS, [1, num_rows, num_cols, 813, 880])

            coverage_list_dB.append(gain_dB_single_BS)
            coverage_location_list.append(coverage_location)

            scene.remove(tx_name)

    return coverage_list_dB, coverage_location_list, tx_orientations


