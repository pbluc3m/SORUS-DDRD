# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

# ray_info.py

import pickle
import math
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from utils import lat_lon_to_cartesian


def ray_generation(
    scene,
    chosen_poor_centroids_BIRCH,
    latitudes,
    longitudes,
    BS_height,
    bearing_deg,
    downtilt,
    xml_file,
    num_rows,
    num_cols,
    frequency,
    ref,
    dif,
    scatt,
    num_samples,
    max_depth
):
    

    # Remove existing transmitters
    for tx_name in list(scene.transmitters.keys()):
        scene.remove(tx_name)
    # Remove existing RIS
    for ris_name in list(scene.ris.keys()):
        scene.remove(ris_name)

    path_results_centroids_BIRCH = {}
    path_types_results_centroids_BIRCH = {}

    # Group users by their serving Tx
    tx_user_map = {}
    for entry in chosen_poor_centroids_BIRCH:
        sid = entry['serverID_users']
        tx_user_map.setdefault(sid, []).append(entry)

    # Loop over each Tx index and its users
    for tx_index, users in tx_user_map.items():
        scene_name = f"scene{tx_index}"
        scene = load_scene(xml_file)

        scene.tx_array = PlanarArray(
            num_rows=num_rows, num_cols=num_cols,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="tr38901", polarization="V"
        )
        scene.rx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"
        )

        scene.frequency = frequency
        scene.synthetic_array = True

        # Add the Tx
        tx_added = False
        for idx, (lat, lon, h, bear, tilt) in enumerate(
            zip(latitudes, longitudes, BS_height, bearing_deg, downtilt)
        ):
            if idx == tx_index:
                x, y = lat_lon_to_cartesian(lat, lon)
                tx_name = f"tx{tx_index}"
                tx = Transmitter(
                    name=tx_name,
                    position=[x, y, h],
                    orientation=[bear*math.pi/180,
                                 tilt*math.pi/180, 0]
                )
                scene.add(tx)
                tx_added = True
                print(f"Added transmitter: {tx_name} to {scene_name} at position {[x,y,h]}")
                break
        if not tx_added:
            raise ValueError(f"Transmitter Tx{tx_index} could not be added to the scene {scene_name}.")

        # Add all receivers for this Tx
        for ue in users:
            rx_name = f"Rx{ue['serverID_users']}_{ue['user_index']}"
            rx = Receiver(
                name=rx_name,
                position=ue['location_users'],
                orientation=[0,0,0]
            )
            scene.add(rx)
            print(f"Added receiver: {rx_name} to {scene_name} at position {ue['location_users']}")

        # Compute multi-user paths
        print(f"Computing paths for Tx{tx_index} in {scene_name} and its users: {[u['user_index'] for u in users]}")
        paths = scene.compute_paths(
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=False,
            max_depth=max_depth,
            num_samples=num_samples
        )

        vertices = paths.vertices
        objects  = paths.objects
        a, tau    = paths.cir(reflection=True, diffraction=False)

        path_results_centroids_BIRCH[f"a{tx_index}"]        = a
        path_results_centroids_BIRCH[f"tau{tx_index}"]      = tau
        path_results_centroids_BIRCH[f"vertices{tx_index}"] = vertices
        path_results_centroids_BIRCH[f"objects{tx_index}"]  = objects

        # Compute per-user path types
        for ue in users:
            for existing_rx in scene.receivers:
                scene.remove(existing_rx)
            rx_name = f"Rx{ue['serverID_users']}_{ue['user_index']}"
            rx = Receiver(
                name=rx_name,
                position=ue['location_users'],
                orientation=[0,0,0]
            )
            scene.add(rx)
            print(f"Computing paths for Tx{tx_index} and receiver {rx_name} individually")
            pt = scene.compute_paths(
                reflection=ref,
                diffraction=dif,
                scattering=scatt,
                ris=False,
                max_depth=max_depth,
                num_samples=num_samples
            ).types
            path_types_results_centroids_BIRCH[f"Tx{tx_index}_Rx{ue['user_index']}"] = pt

        scene.remove(tx_name)


    return path_results_centroids_BIRCH, path_types_results_centroids_BIRCH
