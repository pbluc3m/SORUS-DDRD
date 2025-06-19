# Copyright (c) 2025 Sina Beyraghi
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import tensorflow as tf

def configure_hardware(device: str = "gpu"):
    """
    Configure the hardware environment (CPU or GPU) for the simulation.

    Parameters:
        device (str): "cpu" or "gpu" (default is "gpu").
    """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

    if device.lower() == "cpu":
        # Hides GPUs from TensorFlow
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[INFO] Running simulation on CPU.")
    elif device.lower() == "gpu":
        # Set GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"[INFO] Running simulation on GPU: {gpus[0].name}")
        else:
            print("[WARNING] No GPU found. Falling back to CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        raise ValueError("Invalid device choice. Use 'cpu' or 'gpu'.")

    # Reduce TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    
    

    
    
    
    
    
    
    
    
    