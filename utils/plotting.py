import os
import glob
import random
import numpy as np
import rioxarray as rio
import matplotlib.pyplot as plt

def find_index_in_inferences(tile_id, input_files):
    for index, file_name in enumerate(input_files):
        if tile_id in file_name:
            return index
    
    raise Exception("File not found")

def plot_rgb_agb_gedi(tile_id, predict_input_dir, input_file_names, inference_results, test_label_dir, biome):
    """Plotting routine

        Args:
            tile_id (str): Patch idenfier. If several files in predict_input_dir match the pattern, 
                a random one will be selected.
            predict_input_dir (str): Path to directory with geotiff test images.
            input_file_names (list): List of names of input files aligned with inference_results
            inference_results (list): List of inference outputs
            test_label_dir (str): Path to directory with GEDI test labels.
            biome: Name of the biome the tile belongs to. Used in figure title.
    """
    input_files = glob.glob(predict_input_dir + f'/{tile_id}*.tif')
    if len(input_files) != 1:
        return(f'None or more than one files found matching tile_id pattern: {tile_id}')
    input_file = input_files[0]
    patch_pattern = '_'.join(input_file.split('/')[-1].split('_')[:2])
    output_index = find_index_in_inferences(tile_id, input_file_names)
    output_inference = inference_results[output_index]
    label_file = test_label_dir + f'/{patch_pattern}_tile_label.tif'
    
    rgb_input = rio.open_rasterio(input_file).sel(band=[4, 3, 2]).transpose("y", "x", "band").to_numpy()
    rgb_input = rgb_input/rgb_input.max() # Normalization of RGB bands
    agb_pred = output_inference.detach().cpu().numpy()
    agb_gedi = rio.open_rasterio(label_file).to_numpy()[0]
    agb_gedi[agb_gedi==-1] = np.nan
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    ax[0].imshow(rgb_input)
    im = ax[1].imshow(agb_pred, cmap='Greens')
    ax[1].imshow(agb_gedi, interpolation='none')
    plt.colorbar(im, ax=ax.ravel().tolist(), label='Above-ground biomass (Mg/ha)')
    ax[0].set_title('Input HLS image (RGB bands)')
    ax[1].set_title('Predicted AGB overlaid by GEDI points')
    plt.suptitle(f'Biome: {biome}')