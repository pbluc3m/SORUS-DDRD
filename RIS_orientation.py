# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import tensorflow as tf
import numpy as np

# The compute_azimuth_elevation function to calculate azimuth and elevation angles
def compute_azimuth_elevation(p1, p2):
    '''
    Computes the azimuth and elevation angles from point p1 to point p2.

    Args:
        p1: TensorFlow tensor of shape (..., 3) - the starting point
        p2: TensorFlow tensor of shape (..., 3) - the ending point

    Returns:
        azimuth_deg: Azimuth angle in degrees
        elevation_deg: Elevation angle in degrees
    '''
    dx = p2[..., 0] - p1[..., 0]
    dy = p2[..., 1] - p1[..., 1]
    dz = p2[..., 2] - p1[..., 2]

    distance = tf.sqrt(dx**2 + dy**2 + dz**2)

    azimuth_rad = tf.atan2(dy, dx)
    elevation_rad = tf.asin(dz / distance)

    azimuth_deg = azimuth_rad * 180 / np.pi
    elevation_deg = elevation_rad * 180 / np.pi

    return azimuth_deg, elevation_deg