# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.lines import Line2D


def plot_heatmap(user_params_list,
                 min_UE_x_loc,
                 max_UE_x_loc,
                 min_UE_y_loc,
                 max_UE_y_loc,
                 vmin=-130,
                 vmax=-50,
                 num_ticks=5):
    """
    Plot a heatmap of RSS_dBm values for users.

    Parameters:
    - user_params_list: list of dicts with keys 'x', 'y', 'RSS_dBm'
    - min_UE_x_loc, max_UE_x_loc: float bounds for x-axis ticks
    - min_UE_y_loc, max_UE_y_loc: float bounds for y-axis ticks
    - vmin, vmax: colorbar limits
    - num_ticks: number of ticks on each axis
    """
    print('===================================')
    print('plotting Heatmap...')
    print('===================================')

    # Convert to DataFrame
    df = pd.DataFrame(user_params_list)

    # Pivot to heatmap format
    heatmap_data = df.pivot_table(index='y', columns='x', values='RSS_dBm')

    # Create figure
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        heatmap_data,
        cmap='viridis',
        annot=False,
        cbar=True,
        vmin=vmin,
        vmax=vmax
    )

    # Configure colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel('RSRP (dBm)', fontsize=20, rotation=270, labelpad=20)

    # Axis limits and ticks
    x_ticks = np.linspace(min_UE_x_loc, max_UE_x_loc, num_ticks, dtype=int)
    y_ticks = np.linspace(max_UE_y_loc, min_UE_y_loc, num_ticks, dtype=int)

    ax.set_xlim(0, heatmap_data.shape[1] - 1)
    ax.set_ylim(heatmap_data.shape[0] - 1, 0)

    ax.set_xticks(np.linspace(0, heatmap_data.shape[1] - 1, num_ticks, dtype=int))
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(np.linspace(0, heatmap_data.shape[0] - 1, num_ticks, dtype=int))
    ax.set_yticklabels(y_ticks)

    # Labels and fonts
    plt.xlabel('X Coordinate', fontsize=20)
    plt.ylabel('Y Coordinate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()
    
def plot_rsrp_cdf(bestRSSs_per_user,
                  filter_min=-135,
                  threshold=-100,
                  title='Outdoor RSRP network',
                  figsize=(14, 10),
                  fontsize=20):
    """
    Plot the CDF of received power and compute outage percentage.

    Parameters:
    - bestRSSs_per_user: 1D array-like or torch.Tensor of RSRP values (dBm)
    - filter_min: minimum RSRP value to include in CDF
    - threshold: RSRP threshold for outage calculation
    - title: plot title
    - figsize: figure size tuple
    - fontsize: label and tick fontsize
    """
    print('===================================')
    print('Plotting the CDF received power of all users...')
    print('===================================')

    # Convert to NumPy array
    if isinstance(bestRSSs_per_user, torch.Tensor):
        rp = bestRSSs_per_user.cpu().numpy().flatten()
    else:
        rp = np.array(bestRSSs_per_user).flatten()

    # Filter values
    rp = rp[rp > filter_min]

    # Sort and compute CDF
    sorted_power = np.sort(rp)
    cdf = np.arange(1, len(sorted_power) + 1) / len(sorted_power)

    # Plot CDF
    plt.figure(figsize=figsize)
    plt.plot(sorted_power, cdf, marker='.', linestyle='none')
    plt.xlabel('RSRP (dBm)', fontsize=fontsize)
    plt.ylabel('CDF', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title, fontsize=fontsize - 4)
    plt.grid(True)
    plt.show()

    # Outage calculation
    below_thresh = np.sum(sorted_power < threshold)
    total = sorted_power.size
    pct_below = (below_thresh / total) * 100 if total > 0 else 0

    print(f"Number of users below {threshold} dBm: {below_thresh}")
    print(f"Percentage of users below {threshold} dBm: {pct_below:.2f}%")    
    
def plot_Alg_effect_RIS(Final_data):
    """
    Plot the CDF of RSRP under various RIS algorithms.
    
    Parameters:
    - Final_data (pd.DataFrame): Must contain columns
      ['RSS_without_RIS','phase1','phase1_all_path','phase2','phase2_all_path','LoS_RSS'].
    """
    # Define the columns to plot
    columns_to_plot = [
        'RSS_without_RIS',
        'phase1',
        'phase1_all_path',
        'phase2',
        'phase2_all_path',
        'LoS_RSS'
    ]

    # Define a mapping for the labels
    label_mapping = {
        'RSS_without_RIS':    'No RIS',
        'phase1':             'Strongest ray, T=15',
        'phase1_all_path':    'All-ray, T=15',
        'phase2':             'Strongest Ray, Re-T=10',
        'phase2_all_path':    'All-ray, Re-T=10',
        'LoS_RSS':            'RIS Re-assoc'
    }

    # Define different markers
    line_styles = ['-x', '-v', '-s', '-o', '-D', '-*']

    plt.figure(figsize=(14, 10))

    for i, column in enumerate(columns_to_plot):
        data = Final_data[column].dropna()
        sns.kdeplot(
            data,
            cumulative=True,
            label=label_mapping[column],
            linewidth=2,
            bw_adjust=0.5,
            linestyle='-',
            marker=line_styles[i][-1],
            markevery=0.05,
        )

    plt.xlabel('RSRP (dBm)', fontsize=28)
    plt.ylabel('CDF', fontsize=28)
    plt.xticks(np.arange(-130, -49, 10), fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlim(left=-140, right=-50)
    plt.grid(True)
    plt.legend(fontsize=26)
    plt.tight_layout()
    plt.show()    
    
    
def plot_RIS_units_effect(Final_data):
    """
    Plot the CDF comparing No RIS baseline to increasing numbers of RIS units.
    
    Parameters:
    - Final_data (pd.DataFrame): Must contain 'Final_Cluster_ID', 'LoS_RSS', and 'RSS_without_RIS'.
    """
    # Count and sort clusters by size
    cluster_user_counts = Final_data.groupby('Final_Cluster_ID').size()
    sorted_clusters     = cluster_user_counts.sort_values(ascending=False).index
    total_clusters      = len(sorted_clusters)

    # Build cumulative bins every 50 clusters up to total_clusters
    cut_points = list(range(50, total_clusters, 50)) + [total_clusters]
    cluster_bins = [sorted_clusters[:c] for c in cut_points]

    def get_cdf_data(selected_clusters):
        plot_data = Final_data.copy()
        plot_data['CDF_values'] = np.where(
            plot_data['Final_Cluster_ID'].isin(selected_clusters),
            plot_data['LoS_RSS'],
            plot_data['RSS_without_RIS']
        )
        return plot_data['CDF_values']

    plt.figure(figsize=(14, 10))
    colors  = sns.color_palette("tab10", len(cluster_bins) + 1)
    markers = ['x','o','v','s','D','^','*','P','<','>']
    legend_handles = []

    # Baseline: No RIS
    baseline = Final_data['RSS_without_RIS'].dropna().sort_values()
    y_vals   = np.linspace(0, 1, len(baseline))
    sns.kdeplot(baseline, cumulative=True, bw_adjust=0.5,
                linestyle='dashed', linewidth=2, color='black')
    plt.plot(baseline[::len(baseline)//20], y_vals[::len(baseline)//20],
             marker='x', linestyle='None', color='black')
    legend_handles.append(
        Line2D([0],[0], color='black', linestyle='dashed', marker='x',
               label='No RIS', linewidth=2)
    )

    # Plot each bin
    for i, cluster_bin in enumerate(cluster_bins):
        data = get_cdf_data(cluster_bin).dropna().sort_values()
        y    = np.linspace(0, 1, len(data))
        color  = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        sns.kdeplot(data, cumulative=True, bw_adjust=0.5,
                    linewidth=2, color=color)
        plt.plot(data[::len(data)//20], y[::len(data)//20],
                 marker=marker, linestyle='None', color=color)
        legend_handles.append(
            Line2D([0],[0], color=color, linestyle='-',
                   marker=marker, label=f'{len(cluster_bin)} RIS Units',
                   linewidth=2)
        )

    plt.xlabel('RSRP (dBm)', fontsize=28)
    plt.ylabel('CDF', fontsize=28)
    plt.xticks(np.arange(-130, -49, 10), fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlim(-140, -50)
    plt.grid(True)
    plt.legend(handles=legend_handles, fontsize=28)
    plt.tight_layout()
    plt.show()







    