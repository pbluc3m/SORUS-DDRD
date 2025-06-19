# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

# -*- coding: utf-8 -*-
"""

@author: Sina Beyraghi (Telefonica Research)

Email: mohammadsina.beyraghi@telefonica.com

This is the main script to run the simulator
"""
###############################################################################
##########################################
# Required Packages
##########################################

import numpy as np
from scipy.io import savemat
import math
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import ast
import re
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import sionna 
from sionna.rt import Camera, Antenna, compute_gain,visualize, tr38901_pattern
from sionna.rt import load_scene, Transmitter, Receiver, RIS, PlanarArray, \
                      r_hat, normalize, Camera, Antenna, compute_gain,visualize, \
                      tr38901_pattern
# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.mimo.precoding import normalize_precoding_power, grid_of_beams_dft
import pickle
import torch
import random
from tqdm import tqdm
import sklearn
from sklearn.cluster import KMeans
import seaborn as sns

##############################################################################

##########################################
# Importing required classes
##########################################
import cpu_gpu_config
import utils
import coverage_map
import outdoor_disc
import plotting 
import post_processing
import RIS_orientation
import clustering
import RayInfo 
import reflection_alg_data 
import strongest_ray_alg 
import All_ray_alg 
import RIS_Re_Assocciation 


###############################################################################

##########################################
# System Intialization
##########################################

np.random.seed(41)
torch.manual_seed(41)
tf.random.set_seed(41)
random.seed(41)


cpu = True
gpu = False

# ------- Network Configuration -------
use_real_dataset = True     # Set to False to use manual BS lists

# ------- Indoor Outdoor UE discrimination -----
Sionna_run_maps = False      # if False, we do not compute coverage map for all BS
                            # and we use the results that we generated before. Therfore
                            # in the first time we must compute it. 
outdoor_UE_disc_phase1 = False     # This approach is based on RSRP (If an user 
                                    # get -inf from all BS, it is indoor)
outdoor_UE_disc_phase2 = False     #This approach serves as a complement to Approach 1, 
                                    #which relies on building data to extract outdoor users.

# ------- Network KPIs Plotting without RIS ------
heatmap_plot = False
network_RSRP_cdf_plot = False

# ------- Clustering outage UEs -------
clustring_UE = False     # Using BIRCH algorithm clustering to group nearby outage UEs

# ------- Applying Reflection-Based Algorithms for RIS location deployment --------
sionna_path_generation = False     #This is for generating the rays for centroids
Strongest_ray_RIS_location = False  # Deploying RIS on reflection location of 
                                    # Strongest rays (Evaluating RIS locations based on centroids)
precoder_optimization_strongest_ray = False        # After deploying RISs, we need to select the best beam from serving BS                  
RSRP_RIS_UEs_Computation_Strongest_Ray = False        # Computing RSRP for each UE of each cluster with RIS
All_ray_RIS_location = False         # Deploying RIS on reflection location of 
                                    # All rays (Evaluating RIS locations based on centroids)
precoder_optimization_all_ray = False       # After deploying RISs, we need to select the best beam from serving BS
RSRP_RIS_UEs_Computation_All_Ray = False        # Computing RSRP for each UE of each cluster with RIS

sionna_path_generation_recluster = False
RE_Cluster_Strongest_ray_RIS_location = False
precoder_optimization_strongest_ray_Re_Cluster = False
RSRP_RIS_UEs_Computation_Strongest_Ray_Re_Cluster = False
Re_Cluster_All_ray_RIS_location = False
precoder_optimization_all_ray_Re_cluster = False
RSRP_RIS_UEs_Computation_All_Ray_Re_Cluster = False

LoS_link_UE_RIS = False
LoS_link_RIS_Tx = False
Re_Assocciation_RIS_precoder_update = False
re_assocciation_RSRP_RIS = False

# -------- Plotting the CDF of Algorithm and RIS units number effectiveness -------
Alg_plot_RIS_ref == False
RIS_unit_plot_ref == False

# ------- Applying Scattering-Based Algorithm for RIS location deployment --------
sionna_path_generation_scattering = False
scattering_RIS_location = False
precoder_optimization_scattering = False        # After deploying RISs, we need to select the best beam from serving BS                  
RSRP_RIS_UEs_Computation_scattering = False        # Computing RSRP for each UE of each cluster with RIS

sionna_path_generation_recluster_scattering = False
scattering_RIS_location_recluster = False
precoder_optimization_scattering_recluster = False        # After deploying RISs, we need to select the best beam from serving BS                  
RSRP_RIS_UEs_Computation_scattering_recluster = False        # Computing RSRP for each UE of each cluster with RIS

LoS_link_UE_RIS_scatter = False
LoS_link_RIS_Tx_scatter = False
Re_Assocciation_RIS_precoder_update_scatter = False
re_assocciation_RSRP_RIS_scatter = False


###############################################################################

##########################################
# CPU/GPU configuration
##########################################

if gpu:
    cpu_gpu_config.configure_hardware("gpu")
elif cpu:
    cpu_gpu_config.configure_hardware("cpu")

###############################################################################

print('===================================')
print('System Parameters ...')
print('===================================')

Carrier_freq = 3.5e9    #It is in GHz
txPower_dBm = 13.85     # This is per sub-carrier
outage_threshold = -100    #this is the threshold for extracting outage users
ant_row_elem = 4        #Number of Tx element in row
ant_col_elem = 8        #Number of Tx element in column
ris_row_elem = 100      #Number of RIS element in row
ris_col_elem = 100      #Number of RIS element in column
ant_ver_spacing = 0.5      
ant_hor_spacing = 0.5 
ris_ver_spacing = 0.5      
ris_hor_spacing = 0.5 

ray_reflection = True      
ray_diffraction = False
ray_los = True
rt_ris = False
ray_scattering = False
UE_grid = (2, 2)          #Grid size of coverage map cells in m
num_ray_shooting = int(10e6)        #Number of shooting rays from all BSs
num_ray_shooting_loc = int(10e6)
max_ray_bouncing = int(4)

# The area of the scene that we want griding UEs (Masking)
min_UE_x_loc = -630
max_UE_x_loc = 360
min_UE_y_loc = -580
max_UE_y_loc = 370

meshes_filepath = 'C:\\PhD\\Sionna\\Precoders_RT_Channel\\meshes'


clustering_plotting = False
###############################################################################

print('===================================')
print('Importing BS parameters/creating BS parametrs...')
print('===================================')

'''
If you have the rwal-world BS coordinate information, make it as a .csv file
and import it. The second part is for filtering BSs based on specific area 
regarding desired area section.

If you do not have real-world BS coordinate information, choose some specific 
lat/long based on your desired area and provide the required list which are:
    latitudes, longitudes, BS_height, bearing_deg, downtilt, transmitter_power,
    Beam_width_horizontal
    
These parameter should be as a list

'''
# Choose one of the two options Real_data/manual
if use_real_dataset:
    # Load real-world BS coordinates from .csv and filter by area
    file_path = 'C:\\PhD\\Telefonica\\Gio\\try\\Test_Large_scale_2300_London.csv'
    data = pd.read_csv(file_path)
    latitudes = data.iloc[:, 62]
    longitudes = data.iloc[:, 63]

    # Define the geographic bounding box
    max_lat = 51.5200
    min_lat = 51.5080
    max_long = -0.1250
    min_long = -0.1450

    # Filter the dataset based on bounding box
    mask = (latitudes <= max_lat) & (latitudes >= min_lat) & (longitudes <= max_long) & (longitudes >= min_long)
    filtered_data = data[mask].reset_index(drop=True)

    latitudes = filtered_data.iloc[:, 62]
    longitudes = filtered_data.iloc[:, 63]
    BS_height = filtered_data.iloc[:, 41]
    BS_height.loc[[18, 19, 20]] = 65.3
    bearing_deg = filtered_data.iloc[:, 45]
    downtilt = filtered_data.iloc[:, 43]
    transmitter_power = filtered_data.iloc[:, 46]
    Beam_width_horizontal = filtered_data.iloc[:, 39]

else:
    # Manually define BS parameters as lists
    latitudes = [51.5120, 51.5105]
    longitudes = [-0.1350, -0.1400]
    BS_height = [30.0, 32.5]
    bearing_deg = [45.0, 135.0]
    downtilt = [10.0, 8.0]
    transmitter_power = [43.0, 43.0]
    Beam_width_horizontal = [65.0, 65.0]

###############################################################################

##########################################
# lat/long to Cartesian 
##########################################

x_coords, y_coords = utils.lat_lon_to_cartesian(latitudes, longitudes)

###############################################################################

print('===================================')
print('Loading the Scene, Antenna, Frequency ...')
print('===================================')

'''
To simulate the desired environment, you must generate an .xml file 
describing the scene's site-specific geometry. To generate this .xml please
watch the video in through the link in below from NVIDIA Sionna RT:
    
    https://www.youtube.com/watch?v=7xHLDxUaQ7c

'''
file_path_xml = r"C:\\PhD\\Sionna\\Precoders_RT_Channel\\precoder_RT_Channel.xml"
scene = load_scene(r"C:\\PhD\\Sionna\\Precoders_RT_Channel\\precoder_RT_Channel.xml")

# Configure antenna array for all transmitters and receivers (as needed)
scene.tx_array = PlanarArray(num_rows=ant_row_elem, num_cols=ant_col_elem, vertical_spacing=ant_ver_spacing, horizontal_spacing=ant_hor_spacing, pattern="tr38901", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")

# Operating freq. in Hz, implicitly updates RadioMaterials
scene.frequency = Carrier_freq 
scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

###############################################################################
##########################################
# DFT codebook Precoder 
##########################################

num_rows = ant_row_elem
num_cols = ant_col_elem
# Generate the Grid of Beams (GoB) precoders
gob = grid_of_beams_dft(num_rows, num_cols)

###############################################################################

print('===================================')
print('Cells Configuration & Computing coverage map...')
print('===================================')

if Sionna_run_maps:
    
    coverage_list_dB, coverage_location_list, tx_orientations = coverage_map.run_coverage_maps(
            scene=scene, latitudes=latitudes, longitudes=longitudes, BS_height=BS_height,
            bearing_deg=bearing_deg, downtilt=downtilt, gob=gob, num_rows=num_rows,
            num_cols=num_cols, max_ray_bouncing=max_ray_bouncing, ray_reflection=ray_reflection,
            ray_diffraction=ray_diffraction, ray_los=ray_los, rt_ris=rt_ris, ray_scattering=ray_scattering,
            UE_grid=UE_grid, num_ray_shooting=num_ray_shooting, run_maps=True)
    
##########################################
# If Sionna_run_maps == True, we can save the results to avoid recomputing
##########################################


# Assuming gain_dB is a TensorFlow tensor
if Sionna_run_maps == True:
    # Change the combined list to a tensor of all BSs coverage map among the first dim
    gain_dB = tf.concat(coverage_list_dB, axis=0)
    with open("C:...\RIS_Simulator_JSAC\Coverage_map_simulation\2025_04_30_CoverageMaps_BSs_Loop_gain_dB_model_with_precoder_3_5GHz.pkl", 'wb') as f:
        pickle.dump(gain_dB, f)
        
    with open('...\RIS_Simulator_JSAC\Coverage_map_simulation\coverage_location_list_3_5.pkl', 'wb') as f:
        pickle.dump(coverage_location_list, f)  # Saving using pickle
        
##########################################
# If Sionna_run_maps == False, we can use the saved results to avoid recomputing
##########################################
                
# Load the tensor
if Sionna_run_maps == False:
    with open('2025_04_30_CoverageMaps_BSs_Loop_gain_dB_model_with_precoder_3_5GHz.pkl', 'rb') as f:
        gain_dB = pickle.load(f)
        
    with open('coverage_location_list_3_5.pkl', 'rb') as f:
        coverage_location_list = pickle.load(f)
###############################################################################

print('===================================')
print('Extracting outdoor UEs...')
print('===================================')

if outdoor_UE_disc_phase1:
    outdoor_loc_gain = outdoor_disc.extract_outdoor_loc_gain(
        coverage_location_list,
        gain_dB,
        min_UE_x_loc = min_UE_x_loc,
        max_UE_x_loc = max_UE_x_loc,
        min_UE_y_loc = min_UE_y_loc,
        max_UE_y_loc = max_UE_y_loc
    )

    #### We save it to avoid running this part of the code again (Time consuming)
    file_path = (
        r"C:\PhD\Sionna\Coverage_improvement_RIS_Sionna\Simulations_with_precoders\\" +
        r"RIS_Simulator_JSAC\outdoor_indoor_discrimination\\" +
        r"outdoor_loc_gain_3_5Ghz_with_precoder.pt")
    torch.save(outdoor_loc_gain, file_path)

###############################################

### Loading the outdoor UEs from phase1
outdoor_loc_gain = torch.load(r"C:\PhD\Sionna\Coverage_improvement_RIS_Sionna\Simulations_with_precoders\\" +
                              r"RIS_Simulator_JSAC\outdoor_indoor_discrimination\\" +
                              r"outdoor_loc_gain_3_5Ghz_with_precoder.pt")

### Applying Phase2 for extracting outdoor UEs (Building-based)
if outdoor_UE_disc_phase2:
    outdoor_data_tensor, inside_outside_tensor = outdoor_disc.get_outdoor_users(
        outdoor_loc_gain,
        ply_folder = meshes_filepath,
        threshold=0.0,
        verbose=True
    )
    
    # Save the resulting tensors
    outdoor_save_path = "C:\PhD\Sionna\Coverage_improvement_RIS_Sionna\Simulations_with_precoders\RIS_Simulator_JSAC\outdoor_indoor_discrimination\outdoor_users_tensor_3_5GHz_with_precoder.pt"
    inside_outside_save_path = "C:\PhD\Sionna\Coverage_improvement_RIS_Sionna\Simulations_with_precoders\RIS_Simulator_JSAC\outdoor_indoor_discrimination\inside_outside_tensor_3_5GHz_test.pt"
    torch.save(outdoor_data_tensor, outdoor_save_path)
    torch.save(inside_outside_tensor, inside_outside_save_path)    
        
###############################################################################

print('===================================')
print('Calculating the received power ...')
print('===================================')

### Loading the outdoor UEs from phase2
outdoor_users = torch.load('outdoor_users_tensor_3_5GHz_with_precoder_heatmap.pt')

Coverage_channel_sionna_torch = outdoor_users
Channel_large_RT_dB = Coverage_channel_sionna_torch
Channel_large_RT_full = Channel_large_RT_dB
Channel_large_RT_location = Channel_large_RT_full[:, :, :, :, :2]
Channel_large_RT_dB = Channel_large_RT_dB[..., 3]
Channel_large_RT_dB = Channel_large_RT_dB.unsqueeze(-1)
Channel_large_RT = 10 ** (Channel_large_RT_dB / 10.0)
Channel_large_RT = torch.sqrt(Channel_large_RT)
txPower_linear = 10 ** ((txPower_dBm - 30) / 10.0)
rx_dB = txPower_dBm + Channel_large_RT_dB

###############################################################################

print('===================================')
print('Users association...')
print('===================================')

#Build BS list
baseStations = utils.construct_bs_dict(x_coords, y_coords, BS_height, transmitter_power,
    bearing_deg, downtilt, Beam_width_horizontal, latitudes, longitudes)

#Build User list
user_params_list = utils.construct_user_dict(Channel_large_RT_full)

# User association
baseStations, user_params_list, bestRSSs_per_user = utils.associate_users(
    rx_dB, baseStations, user_params_list)
###############################################################################

##########################################
# Network Heatmap Plotting
##########################################

if heatmap_plot:
    plotting.plot_heatmap(user_params_list, min_UE_x_loc=min_UE_x_loc, max_UE_x_loc=max_UE_x_loc,
                          min_UE_y_loc=min_UE_y_loc, max_UE_y_loc=max_UE_y_loc)

##########################################
# RSRP CDF Plotting
##########################################

if network_RSRP_cdf_plot:
    plotting.plot_rsrp_cdf(bestRSSs_per_user, filter_min=-135, threshold=-100,
                           title='Outdoor RSRP network 3.5GHz with DFT')
###############################################################################
print('===================================')
print('Extracting outage UEs...')
print('===================================')

Chosen_poor_coverage = post_processing.choose_poor_coverage(
    baseStations, user_params_list, min_rss = -130, max_rss = outage_threshold)

chosen_poor_coverage_Dataframe = pd.DataFrame(Chosen_poor_coverage)
###############################################################################

print('===================================')
print('Clustering outage UEs...')
print('===================================')

if clustring_UE:
  chosen_poor_coverage_Dataframe_BIRCH = clustering.perform_birch_clustering(
      chosen_poor_coverage_Dataframe, threshold=15, n_clusters=None, plot=clustering_plotting)
###############################################################################

##########################################
# Finding and assigning the nearest tiles to the original centroid location
##########################################

(nearest_tile_indices, nearest_tile_locations, pathgain_nearest_tiles,
 second_min_indices, second_min_locations, pathgain_second_tiles,
 third_min_indices, third_min_locations, pathgain_third_tiles,
 fourth_min_indices, fourth_min_locations, pathgain_fourth_tiles) = clustering.extract_centroid_tiles(
    outdoor_users, chosen_poor_coverage_Dataframe_BIRCH, rx_dB)

     
(combined_near_tiles_indices, combined_near_tiles_locations,combined_near_tiles_pathgains) = post_processing.combine_near_tiles(
    nearest_tile_indices, second_min_indices, third_min_indices, fourth_min_indices,
    nearest_tile_locations, second_min_locations, third_min_locations, fourth_min_locations,
    pathgain_nearest_tiles, pathgain_second_tiles, pathgain_third_tiles, pathgain_fourth_tiles
)

###############################################################################

#Build BS_centroid list
baseStations_centoids_BRICH = utils.construct_bs_dict(
    x_coords, y_coords, BS_height, transmitter_power, bearing_deg, downtilt,
    Beam_width_horizontal, latitudes, longitudes)

#Build UE_centroid list
centroids_BRICH_params_list = utils.create_user_params_list(combined_near_tiles_locations)

#centroid association
centroids_BRICH_params_list, baseStations_centoids_BRICH = utils.association_centroid(
    combined_near_tiles_pathgains, centroids_BRICH_params_list, baseStations_centoids_BRICH
)
###############################################################################

Chosen_poor_centroids_BIRCH = post_processing.create_chosen_poor_coverage_centroid(baseStations_centoids_BRICH, centroids_BRICH_params_list)

###############################################################################

print('===================================')
print('Ray Tracing for centroids ...')
print('===================================')

if sionna_path_generation:
    path_results_centroids_BIRCH, path_types_results_centroids_BIRCH = RayInfo.ray_generation(
        scene, Chosen_poor_centroids_BIRCH, latitudes, longitudes, BS_height,
        bearing_deg, downtilt, file_path_xml, num_rows, num_cols, Carrier_freq,
        ray_reflection, ray_diffraction, ray_scattering, num_ray_shooting_loc, max_ray_bouncing)
    
    # Save dictionaries to a file
    with open('path_results_centroids_BIRCH_cluster_3_5GHz_cluster_T15.pkl', 'wb') as f:
        pickle.dump(path_results_centroids_BIRCH, f)
    
    with open('path_types_results_centroids_BIRCH_cluster_3_5GHz_cluster_T15.pkl', 'wb') as f:
        pickle.dump(path_types_results_centroids_BIRCH, f)

# load the ray information that already saved
if sionna_path_generation == False:
    # Load the dictionaries from the file
    with open('path_results_centroids_BIRCH_cluster_3_5GHz_cluster_T15.pkl', 'rb') as f:
        path_results_centroids_BIRCH = pickle.load(f)

    with open('path_types_results_centroids_BIRCH_cluster_3_5GHz_cluster_T15.pkl', 'rb') as f:
        path_types_results_centroids_BIRCH = pickle.load(f)
        
###############################################################################

print('===================================')
print('Strongest Ray Data Prep (Reflection-Based) ...')
print('===================================')

##########################################
# Finding the strongest ray for each centroid
##########################################
best_path_reflections_centroids, Centroids_all_vertices, Centroids_all_objects_filtered, Centroids_all_a, Centroids_all_objects = post_processing.ray_funcs(
    path_results_centroids_BIRCH, path_types_results_centroids_BIRCH)

##########################################
# Dataframe prepration for strongest ray Algorithm
##########################################
df_centroids_BIRCH = reflection_alg_data.Data_Strong_ray(
    Chosen_poor_centroids_BIRCH, best_path_reflections_centroids, Centroids_all_vertices)
df_centroids_BIRCH = reflection_alg_data.update_rx_indice(df_centroids_BIRCH, nearest_tile_indices)

df_centroids_BIRCH_filtered = reflection_alg_data.Choose_centroid_with_ray(df_centroids_BIRCH)
df_centroids_BIRCH_fitered2 = reflection_alg_data.Data2_Strong_ray(df_centroids_BIRCH_filtered)
df_centroids_BIRCH_fitered3 = reflection_alg_data.Data3_Strong_ray(df_centroids_BIRCH_fitered2)
df_centroids_BIRCH_fitered4 = reflection_alg_data.Data4_Strong_ray(df_centroids_BIRCH_fitered3)
###############################################################################

print('===================================')
print('Starting Strongest Ray Algorithm (Reflection-Based) ...')
print('===================================')

##########################################
# Deploying RIS on reflection locations and evaluating the centroid point RSRP
##########################################

# For the first time we need to simulate it.
if Strongest_ray_RIS_location:
    final_results_df_centroids = strongest_ray_alg.Strongest_Ray_RIS_Opt(
        scene,
        df_centroids_BIRCH_filtered, df_centroids_BIRCH_fitered2,
        df_centroids_BIRCH_fitered3, df_centroids_BIRCH_fitered4,
        filtered_data, gob, txPower_dBm, ris_row_elem, ris_col_elem,
        ray_reflection, ray_diffraction, ray_scattering, UE_grid,
        num_ray_shooting, max_ray_bouncing)

    final_results_df_centroids.to_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_cluster_T15_12_05_2025.csv', index=False)

# If we simulated it before and saved the results, we can load it
if Strongest_ray_RIS_location == False:
    final_results_df_centroids = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_cluster_T15_12_05_2025.csv')


##########################################
# Dataframe prepration for BS precoder optimization Strongest ray algorithm
##########################################

Best_RIS_RSS_Configuration_BIRCH_Centroids = reflection_alg_data.Data_prep_precoder(
    final_results_df_centroids, df_centroids_BIRCH_filtered)

Best_RIS_RSS_Configuration_BIRCH_Centroids.to_csv('Best_RIS_RSS_T15_3_5GHz_before_precoder.csv', index=False)

###############################################################################
##########################################
# Beam Optimization for Strongest Ray Selection Algorithm
##########################################

if precoder_optimization_strongest_ray:
    Centroids_precoder_update = strongest_ray_alg.BS_Beam_Opt(
        scene,'Best_RIS_RSS_T15_3_5GHz_before_precoder.csv',
        latitudes, longitudes, BS_height, bearing_deg, downtilt, num_rows, num_cols,
        gob, txPower_dBm, 'Centroids_precoder_update_T15_3_5GHz.csv')

if precoder_optimization_strongest_ray == False:
    Centroid_data_with_updated_precoder = pd.read_csv('Centroids_precoder_update_T15_3_5GHz.csv')

##########################################
# Updating the precoder data in the dataframe
##########################################

Centroid_data_with_updated_precoder['Precoder'] = Centroid_data_with_updated_precoder.apply(lambda row: row['Best_RIS_server'] if row['Best_RIS_power'] != -np.inf else row['Precoder'], axis=1)
output_file_path = "Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T15_Updated_precoder_12_05_2025.csv"  # Specify the desired output file path
Centroid_data_with_updated_precoder.to_csv(output_file_path, index=False)

###############################################################################
##########################################
# Computing RSRP for outage UEs with RIS
##########################################

Best_RIS_RSS_Configuration_BIRCH_Centroids = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T15_Updated_precoder_12_05_2025.csv')
Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'] = Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'].apply(post_processing.convert_bracket_to_parenthesis)

if RSRP_RIS_UEs_Computation_Strongest_Ray:
    results_cluster_df_centroids = strongest_ray_alg.RSRP_RIS_UE(
        scene, Best_RIS_RSS_Configuration_BIRCH_Centroids, filtered_data,
        gob, chosen_poor_coverage_Dataframe_BIRCH, txPower_dBm,
        'Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_12_05_2025.csv'
    )

if RSRP_RIS_UEs_Computation_Strongest_Ray == False:
    results_cluster_df_centroids = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_12_05_2025.csv')

results_cluster_df_centroids['Final_RSS'] = results_cluster_df_centroids.apply(post_processing.choose_rss, axis=1)

print('===================================')
print('Strongest Ray Algorithm (Reflection-Based) Finished ...')
print('===================================')
###############################################################################

print('===================================')
print('All Ray Data Prep (Reflection-Based) ...')
print('===================================')

proc_vertices, proc_objects, updated_vertices, a_db_list, all_path_details_centroids = reflection_alg_data.PreData_All_Ray(
    Centroids_all_vertices, Centroids_all_objects, Centroids_all_a, Centroids_all_objects_filtered
)

df_paths_centroids_BIRCH = reflection_alg_data.Data_All_Ray(Chosen_poor_centroids_BIRCH, 
                                                            all_path_details_centroids, 
                                                            Centroids_all_vertices)
df_paths_centroids_BIRCH2 = reflection_alg_data.Data2_All_Ray(df_paths_centroids_BIRCH)

###############################################################################

print('===================================')
print('Starting All Ray Algorithm (Reflection-Based) ...')
print('===================================')

##########################################
# Deploying RIS on first and second reflection locations of ALL Rays and evaluating 
# the centroid point RSRP
##########################################

if All_ray_RIS_location:
    final_results = All_ray_alg.All_Ray_RIS_Opt(
        scene,
        df_paths_centroids_BIRCH,
        df_paths_centroids_BIRCH2,
        gob,
        filtered_data,
        txPower_dBm,
        'Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_T15_all_path_13_05_2025.csv'
    )

if All_ray_RIS_location == False:
    final_results_df_centroids_all_paths = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_T15_all_path_13_05_2025.csv')

final_results_df_centroids_all_paths = reflection_alg_data.update_rx_indice(final_results_df_centroids_all_paths, nearest_tile_indices)
###############################################################################

##########################################
# Dataframe prepration for BS precoder optimization All Ray algorithm
##########################################

filtered_improved_centroids_all_path_results, filtered_No_improved_centroids_all_path_results,clusters_to_keep_phase1 = reflection_alg_data.All_ray_filtering_data(
    results_cluster_df_centroids,
    final_results_df_centroids_all_paths)

Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path = reflection_alg_data.Data_prep_precoder_all_ray(
    filtered_improved_centroids_all_path_results)

# clusters_to_keep_phase1 was produced earlier by All_ray_filtering_data
df_final_selected, filtered_clusters_to_keep_phase1 = reflection_alg_data.Select_RIS_All_Ray(
    Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path, clusters_to_keep_phase1,
    'Best_RIS_RSS_Configuration_BIRCH_Centroids_3_5GHz_T15_all_path.csv')

###############################################################################

##########################################
# Beam Optimization for All Ray Selection Algorithm
##########################################

if precoder_optimization_all_ray:
    Centroids_precoder_update = strongest_ray_alg.BS_Beam_Opt(
        scene,'Best_RIS_RSS_Configuration_BIRCH_Centroids_3_5GHz_T15_all_path.csv',
        latitudes, longitudes, BS_height, bearing_deg, downtilt, num_rows, num_cols,
        gob, txPower_dBm, 'Centroids_precoder_update_T15_3_5GHz_all_path.csv')

if precoder_optimization_all_ray == False:
    Centroid_data_with_updated_precoder = pd.read_csv('Centroids_precoder_update_T15_3_5GHz_all_path.csv')

###############################################################################

##########################################
# Updating the precoder data in the dataframe
##########################################

Centroid_data_with_updated_precoder['Precoder'] = Centroid_data_with_updated_precoder.apply(lambda row: row['Best_RIS_server'] if row['Best_RIS_power'] != -np.inf else row['Precoder'], axis=1)
output_file_path = "Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T15_Updated_precoder_all_path_14_05_2025.csv"  # Specify the desired output file path
Centroid_data_with_updated_precoder.to_csv(output_file_path, index=False)

###############################################################################

##########################################
# Computing RSRP for outage UEs with RIS
##########################################

Best_RIS_RSS_Configuration_BIRCH_Centroids = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T15_Updated_precoder_all_path_14_05_2025.csv')
Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'] = Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'].apply(post_processing.convert_bracket_to_parenthesis)

if RSRP_RIS_UEs_Computation_All_Ray:
    results_cluster_df_centroids = strongest_ray_alg.RSRP_RIS_UE(
        scene, Best_RIS_RSS_Configuration_BIRCH_Centroids, filtered_data,
        gob, chosen_poor_coverage_Dataframe_BIRCH, txPower_dBm,
        'Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_all_path_14_05_2025.csv'
    )

if RSRP_RIS_UEs_Computation_All_Ray == False:
    results_cluster_df_centroids = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_all_path_14_05_2025.csv')

results_cluster_df_centroids['Final_RSS'] = results_cluster_df_centroids.apply(post_processing.choose_rss, axis=1)

print('===================================')
print('All Ray Algorithm (Reflection-Based) Finished ...')
print('===================================')
###############################################################################

print('===================================')
print('Preparing data for Re-clustering ...')
print('===================================')

results_cluster_df_centroids = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_12_05_2025.csv')
results_cluster_df_centroids['Final_RSS'] = results_cluster_df_centroids.apply(post_processing.choose_rss, axis=1)


df_recluster = reflection_alg_data.Data_Recluster(
    results_cluster_df_centroids,
    final_results_df_centroids_all_paths,
    'Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_all_path_14_05_2025.csv'
)
###############################################################################

print('===================================')
print('Re-clustering remained outage UEs ...')
print('===================================')

chosen_poor_coverage_Dataframe = pd.read_csv('Dataframe_prepared_3_5GHz_T15_clustering_again.csv')
chosen_poor_coverage_Dataframe = chosen_poor_coverage_Dataframe.rename(columns={"Location_users": "location_users"})

chosen_poor_coverage_Dataframe_BIRCH = clustering.perform_birch_clustering(
    chosen_poor_coverage_Dataframe, threshold=10, n_clusters=None, plot=clustering_plotting)

###############################################################################

##########################################
# Finding and assigning the nearest tiles to the original centroid location
##########################################

(nearest_tile_indices, nearest_tile_locations, pathgain_nearest_tiles,
 second_min_indices, second_min_locations, pathgain_second_tiles,
 third_min_indices, third_min_locations, pathgain_third_tiles,
 fourth_min_indices, fourth_min_locations, pathgain_fourth_tiles) = clustering.extract_centroid_tiles(
    outdoor_users, chosen_poor_coverage_Dataframe_BIRCH, rx_dB)

     
(combined_near_tiles_indices, combined_near_tiles_locations,combined_near_tiles_pathgains) = post_processing.combine_near_tiles(
    nearest_tile_indices, second_min_indices, third_min_indices, fourth_min_indices,
    nearest_tile_locations, second_min_locations, third_min_locations, fourth_min_locations,
    pathgain_nearest_tiles, pathgain_second_tiles, pathgain_third_tiles, pathgain_fourth_tiles
)
###############################################################################
# RE-CLUSTERING
#Build BS_centroid list
baseStations_centoids_BRICH = utils.construct_bs_dict(
    x_coords, y_coords, BS_height, transmitter_power, bearing_deg, downtilt,
    Beam_width_horizontal, latitudes, longitudes)

#Build UE_centroid list
centroids_BRICH_params_list = utils.create_user_params_list(combined_near_tiles_locations)

#centroid association
centroids_BRICH_params_list, baseStations_centoids_BRICH = utils.association_centroid(
    combined_near_tiles_pathgains, centroids_BRICH_params_list, baseStations_centoids_BRICH
)
###############################################################################

Chosen_poor_centroids_BIRCH = post_processing.create_chosen_poor_coverage_centroid(baseStations_centoids_BRICH, centroids_BRICH_params_list)

###############################################################################

print('===================================')
print('Ray Tracing for RE-Clustered centroids ...')
print('===================================')

if sionna_path_generation_recluster:
    path_results_centroids_BIRCH, path_types_results_centroids_BIRCH = RayInfo.ray_generation(
        scene, Chosen_poor_centroids_BIRCH, latitudes, longitudes, BS_height,
        bearing_deg, downtilt, file_path_xml, num_rows, num_cols, Carrier_freq,
        ray_reflection, ray_diffraction, ray_scattering, num_ray_shooting_loc, max_ray_bouncing)
    
    # Save dictionaries to a file
    with open('path_results_centroids_BIRCH_cluster_3_5GHz_cluster_T10.pkl', 'wb') as f:
        pickle.dump(path_results_centroids_BIRCH, f)
    
    with open('path_types_results_centroids_BIRCH_cluster_3_5GHz_cluster_T10.pkl', 'wb') as f:
        pickle.dump(path_types_results_centroids_BIRCH, f)

# load the ray information that already saved
if sionna_path_generation_recluster == False:
    # Load the dictionaries from the file
    with open('path_results_centroids_BIRCH_cluster_3_5GHz_cluster_T10.pkl', 'rb') as f:
        path_results_centroids_BIRCH = pickle.load(f)

    with open('path_types_results_centroids_BIRCH_cluster_3_5GHz_cluster_T10.pkl', 'rb') as f:
        path_types_results_centroids_BIRCH = pickle.load(f)
        
###############################################################################

print('===================================')
print('RE-Cluster Strongest Ray Data Prep (Reflection-Based) ...')
print('===================================')

##########################################
# Finding the strongest ray for each re-clustered centroid
##########################################
best_path_reflections_centroids, Centroids_all_vertices, Centroids_all_objects_filtered, Centroids_all_a, Centroids_all_objects = post_processing.ray_funcs(
    path_results_centroids_BIRCH, path_types_results_centroids_BIRCH)

##########################################
# Dataframe prepration for re-cluster strongest ray Algorithm
##########################################
df_centroids_BIRCH = reflection_alg_data.Data_Strong_ray(
    Chosen_poor_centroids_BIRCH, best_path_reflections_centroids, Centroids_all_vertices)
df_centroids_BIRCH = reflection_alg_data.update_rx_indice(df_centroids_BIRCH, nearest_tile_indices)

df_centroids_BIRCH_filtered = reflection_alg_data.Choose_centroid_with_ray(df_centroids_BIRCH)
df_centroids_BIRCH_fitered2 = reflection_alg_data.Data2_Strong_ray(df_centroids_BIRCH_filtered)
df_centroids_BIRCH_fitered3 = reflection_alg_data.Data3_Strong_ray(df_centroids_BIRCH_fitered2)
df_centroids_BIRCH_fitered4 = reflection_alg_data.Data4_Strong_ray(df_centroids_BIRCH_fitered3)
###############################################################################

print('===================================')
print('Starting RE-Cluster Strongest Ray Algorithm (Reflection-Based) ...')
print('===================================')

##########################################
# Deploying RIS on reflection locations and evaluating the centroid point RSRP
##########################################

# For the first time we need to simulate it.
if RE_Cluster_Strongest_ray_RIS_location:
    final_results_df_centroids = strongest_ray_alg.Strongest_Ray_RIS_Opt(
        scene,
        df_centroids_BIRCH_filtered, df_centroids_BIRCH_fitered2,
        df_centroids_BIRCH_fitered3, df_centroids_BIRCH_fitered4,
        filtered_data, gob, txPower_dBm, ris_row_elem, ris_col_elem,
        ray_reflection, ray_diffraction, ray_scattering, UE_grid,
        num_ray_shooting, max_ray_bouncing)

    final_results_df_centroids.to_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_REcluster_T10_14_05_2025.csv', index=False)

# If we simulated it before and saved the results, we can load it
if RE_Cluster_Strongest_ray_RIS_location == False:
    final_results_df_centroids = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_REcluster_T10_14_05_2025.csv')


##########################################
# Dataframe prepration for BS precoder optimization Strongest ray algorithm
##########################################

Best_RIS_RSS_Configuration_BIRCH_Centroids = reflection_alg_data.Data_prep_precoder(
    final_results_df_centroids, df_centroids_BIRCH_filtered)

Best_RIS_RSS_Configuration_BIRCH_Centroids.to_csv('Best_RIS_RSS_T10_3_5GHz_before_precoder.csv', index=False)
###############################################################################
##########################################
# Beam Optimization for RE-Cluster Strongest Ray Selection Algorithm
##########################################

if precoder_optimization_strongest_ray_Re_Cluster:
    Centroids_precoder_update = strongest_ray_alg.BS_Beam_Opt(
        scene,'Best_RIS_RSS_T10_3_5GHz_before_precoder.csv',
        latitudes, longitudes, BS_height, bearing_deg, downtilt, num_rows, num_cols,
        gob, txPower_dBm, 'Centroids_precoder_update_T10_3_5GHz.csv')

if precoder_optimization_strongest_ray_Re_Cluster == False:
    Centroid_data_with_updated_precoder = pd.read_csv('Centroids_precoder_update_T10_3_5GHz.csv')

##########################################
# Updating the precoder data in the dataframe
##########################################

Centroid_data_with_updated_precoder['Precoder'] = Centroid_data_with_updated_precoder.apply(lambda row: row['Best_RIS_server'] if row['Best_RIS_power'] != -np.inf else row['Precoder'], axis=1)
output_file_path = "Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T10_Updated_precoder_15_05_2025.csv"  # Specify the desired output file path
Centroid_data_with_updated_precoder.to_csv(output_file_path, index=False)
###############################################################################
##########################################
# Computing RSRP for outage UEs with RIS
##########################################

Best_RIS_RSS_Configuration_BIRCH_Centroids = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T10_Updated_precoder_15_05_2025.csv')
Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'] = Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'].apply(post_processing.convert_bracket_to_parenthesis)

if RSRP_RIS_UEs_Computation_Strongest_Ray_Re_Cluster:
    results_cluster_df_centroids = strongest_ray_alg.RSRP_RIS_UE(
        scene, Best_RIS_RSS_Configuration_BIRCH_Centroids, filtered_data,
        gob, chosen_poor_coverage_Dataframe_BIRCH, txPower_dBm,
        'Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T10_15_05_2025.csv'
    )

if RSRP_RIS_UEs_Computation_Strongest_Ray_Re_Cluster == False:
    results_cluster_df_centroids = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T10_15_05_2025.csv')

results_cluster_df_centroids['Final_RSS'] = results_cluster_df_centroids.apply(post_processing.choose_rss, axis=1)

print('===================================')
print('Re-Cluster Strongest Ray Algorithm (Reflection-Based) Finished ...')
print('===================================')
###############################################################################
print('===================================')
print('All Ray Data Prep (Reflection-Based) ...')
print('===================================')

proc_vertices, proc_objects, updated_vertices, a_db_list, all_path_details_centroids = reflection_alg_data.PreData_All_Ray(
    Centroids_all_vertices, Centroids_all_objects, Centroids_all_a, Centroids_all_objects_filtered
)

df_paths_centroids_BIRCH = reflection_alg_data.Data_All_Ray(Chosen_poor_centroids_BIRCH, 
                                                            all_path_details_centroids, 
                                                            Centroids_all_vertices)
df_paths_centroids_BIRCH2 = reflection_alg_data.Data2_All_Ray(df_paths_centroids_BIRCH)

###############################################################################

print('===================================')
print('Starting Re-Cluster All Ray Algorithm (Reflection-Based) ...')
print('===================================')

##########################################
# Deploying RIS on first and second reflection locations of ALL Rays and evaluating 
# the centroid point RSRP
##########################################

if Re_Cluster_All_ray_RIS_location:
    final_results = All_ray_alg.All_Ray_RIS_Opt(
        scene,
        df_paths_centroids_BIRCH,
        df_paths_centroids_BIRCH2,
        gob,
        filtered_data,
        txPower_dBm,
        'Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_T10_all_path_15_05_2025.csv'
    )

if Re_Cluster_All_ray_RIS_location == False:
    final_results_df_centroids_all_paths = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_precoder_T10_all_path_15_05_2025.csv')

final_results_df_centroids_all_paths = reflection_alg_data.update_rx_indice(final_results_df_centroids_all_paths, nearest_tile_indices)
###############################################################################

##########################################
# Dataframe prepration for BS precoder optimization All Ray algorithm
##########################################

filtered_improved_centroids_all_path_results, filtered_No_improved_centroids_all_path_results,clusters_to_keep_phase1 = reflection_alg_data.All_ray_filtering_data(
    results_cluster_df_centroids,
    final_results_df_centroids_all_paths)

Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path = reflection_alg_data.Data_prep_precoder_all_ray(
    filtered_improved_centroids_all_path_results)

# clusters_to_keep_phase1 was produced earlier by All_ray_filtering_data
df_final_selected, filtered_clusters_to_keep_phase1 = reflection_alg_data.Select_RIS_All_Ray(
    Best_RIS_RSS_Configuration_BIRCH_Centroids_all_path, clusters_to_keep_phase1,
    'Best_RIS_RSS_Configuration_BIRCH_Centroids_3_5GHz_T10_all_path.csv')

###############################################################################

##########################################
# Beam Optimization for All Ray Selection Algorithm
##########################################

if precoder_optimization_all_ray_Re_cluster:
    Centroids_precoder_update = strongest_ray_alg.BS_Beam_Opt(
        scene,'Best_RIS_RSS_Configuration_BIRCH_Centroids_3_5GHz_T10_all_path.csv',
        latitudes, longitudes, BS_height, bearing_deg, downtilt, num_rows, num_cols,
        gob, txPower_dBm, 'Centroids_precoder_update_T10_3_5GHz_all_path.csv')

if precoder_optimization_all_ray_Re_cluster == False:
    Centroid_data_with_updated_precoder = pd.read_csv('Centroids_precoder_update_T10_3_5GHz_all_path.csv')

###############################################################################

##########################################
# Updating the precoder data in the dataframe
##########################################

Centroid_data_with_updated_precoder['Precoder'] = Centroid_data_with_updated_precoder.apply(lambda row: row['Best_RIS_server'] if row['Best_RIS_power'] != -np.inf else row['Precoder'], axis=1)
output_file_path = "Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T10_Updated_precoder_all_path_15_05_2025.csv"  # Specify the desired output file path
Centroid_data_with_updated_precoder.to_csv(output_file_path, index=False)

###############################################################################

##########################################
# Computing RSRP for outage UEs with RIS
##########################################

Best_RIS_RSS_Configuration_BIRCH_Centroids = pd.read_csv('Coverage_centroids_RIS260_cluster_BRICH_3_5GHz_T10_Updated_precoder_all_path_15_05_2025.csv')
Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'] = Best_RIS_RSS_Configuration_BIRCH_Centroids['Point_A'].apply(post_processing.convert_bracket_to_parenthesis)

if RSRP_RIS_UEs_Computation_All_Ray_Re_Cluster:
    results_cluster_df_centroids = strongest_ray_alg.RSRP_RIS_UE(
        scene, Best_RIS_RSS_Configuration_BIRCH_Centroids, filtered_data,
        gob, chosen_poor_coverage_Dataframe_BIRCH, txPower_dBm,
        'Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T10_all_path_15_05_2025.csv'
    )

if RSRP_RIS_UEs_Computation_All_Ray_Re_Cluster == False:
    results_cluster_df_centroids = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T10_all_path_15_05_2025.csv')

results_cluster_df_centroids['Final_RSS'] = results_cluster_df_centroids.apply(post_processing.choose_rss, axis=1)

print('===================================')
print('Re-Cluster All Ray Algorithm (Reflection-Based) Finished ...')
print('===================================')
###############################################################################

print('===================================')
print('Final Data Prepration ...')
print('===================================')


Coverage_user_phase1             = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_12_05_2025.csv')
Coverage_user_phase1_all_path    = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T15_all_path_14_05_2025.csv')
Coverage_user_phase2             = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T10_15_05_2025.csv')
Coverage_user_phase2_all_path    = pd.read_csv('Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_T10_all_path_15_05_2025.csv')

Final_data, unique_RIS_list, Worst_UEs_Data = post_processing.Fina_Dataframe(
    Coverage_user_phase1, Coverage_user_phase1_all_path,
    Coverage_user_phase2, Coverage_user_phase2_all_path,
    'clusters_removed_phase1.npy')
###############################################################################
print('===================================')
print('RIS Re-Association ...')
print('===================================')

##########################################
# Finding LoS link between remained outage UEs and deployed RISs
##########################################

if LoS_link_UE_RIS:
    Filtered_LoS_Results = RIS_Re_Assocciation.LoS_UE_RIS(
        Worst_UEs_Data,
        unique_RIS_list,
        r"C:\PhD\Sionna\Precoders_RT_Channel\meshes")

    # Define the file path for saving
    csv_filename = "Filtered_LoS_Results.csv"
    
    # Save the filtered DataFrame as a CSV file
    Filtered_LoS_Results.to_csv(csv_filename, index=False)

##########################################
# Finding LoS link between chosen RISs and deployed Transmitters
##########################################

# Load precomputed LOS filter results
Filtered_LoS_Results = pd.read_csv('Filtered_LoS_Results.csv')

if LoS_link_RIS_Tx:
    # Compute which Tx are in LoS with each RIS
    Filtered_LoS_Results = RIS_Re_Assocciation.LoS_RIS_Tx(
        Filtered_LoS_Results,
        Coverage_user_phase1,
        meshes_filepath
    )
    
    csv_filename = "Filtered_LoS_Results.csv"
    Filtered_LoS_Results.to_csv(csv_filename, index=False)
    Final_Filtered_LoS_Results = Filtered_LoS_Results.dropna(subset=['LoS_Tx_indices']).reset_index(drop=True)

    # Define the file path for saving
    csv_filename = "Final_Filtered_LoS_Results.csv"
    Final_Filtered_LoS_Results.to_csv(csv_filename, index=False)

Final_Filtered_LoS_Results = pd.read_csv('Final_Filtered_LoS_Results.csv')
Final_Filtered_LoS_Results.rename(columns={'LoS_Tx_indices': 'Tx_indice'}, inplace=True)
Final_Filtered_LoS_Results.rename(columns={'Rx_index': 'Rx_indice'}, inplace=True)
tx_location_mapping = dict(zip(Coverage_user_phase1['Tx_index'], Coverage_user_phase1['Tx_location']))
Final_Filtered_LoS_Results['Tx_location'] = Final_Filtered_LoS_Results['Tx_indice'].map(tx_location_mapping)
rss_mapping = dict(zip(Final_data['Rx_index'], Final_data['RSS_without_RIS']))
Final_Filtered_LoS_Results['RSS_without_RIS'] = Final_Filtered_LoS_Results['Rx_indice'].map(rss_mapping)
###############################################################################

##########################################
# Re-assocciation precoder updated
##########################################

csv_filename = "Final_Filtered_LoS_Results.csv"
Final_Filtered_LoS_Results.to_csv(csv_filename, index=False)

if Re_Assocciation_RIS_precoder_update:
    updated_df = RIS_Re_Assocciation.LoS_precoder_updated(
        scene=scene,
        file_path='Final_Filtered_LoS_Results.csv',
        latitudes=latitudes,
        longitudes=longitudes,
        BS_height=BS_height,
        bearing_deg=bearing_deg,
        downtilt=downtilt,
        num_rows=ant_row_elem,
        num_cols=ant_col_elem,
        gob=gob,
        parse_RIS_orientation_deg = post_processing.parse_RIS_orientation_deg,
        lat_lon_to_cartesian=utils.lat_lon_to_cartesian,
        output_file="Final_Filtered_LoS_Results_precoder_updated.csv"
    )

###############################################################################

##########################################
# Data Prepration for computing Tx-RIS-UE with Re-assocciation
##########################################

final_los_ue_ris = RIS_Re_Assocciation.LoS_data_RIS_UE(
    file_path_precoder_updated='Final_Filtered_LoS_Results_precoder_updated.csv',
    file_path_UE_RIS='Final_Filtered_LoS_Results.csv',
    output_file='Final_LoS_UE_RIS.csv')

###############################################################################
##########################################
# Computing the RSRP of remained outage UEs with Re-assocciation RIS method
##########################################

LoS_UE_RIS = pd.read_csv('Final_LoS_UE_RIS.csv')

if re_assocciation_RSRP_RIS:
    df_loS_rsrp = RIS_Re_Assocciation.Re_assocciation_RSRP_RIS(
        scene=scene, LoS_UE_RIS=final_los_ue_ris, filtered_data=filtered_data,
        latitudes=latitudes, longitudes=longitudes, BS_height=BS_height,
        bearing_deg=bearing_deg, downtilt=downtilt, gob=gob,
        output_file="Coverage_users_RIS750_cluster_BRICH_10GHz_precoder_LoS_06_05_2025.csv"
    )

###############################################################################
##########################################
# Preparing Final Data
##########################################

Final_data = RIS_Re_Assocciation.RIS_UE_RSRP_Data(
    final_data=Final_data,
    coverage_users_los_csv="Coverage_users_RIS260_cluster_BRICH_3_5GHz_precoder_LoS_16_05_2025.csv",
    output_file="Final_Coverage_results_RSRP_RIS260_07_05_2025.csv")

###############################################################################
##########################################
# Plotting the effectiveness of algorithms
##########################################
if Alg_plot_RIS_ref:
  plotting.plot_Alg_effect_RIS(Final_data)

##########################################
# Plotting the effectiveness of RIS units
##########################################
if RIS_unit_plot_ref:
  plotting.plot_RIS_units_effect(Final_data)

###############################################################################











