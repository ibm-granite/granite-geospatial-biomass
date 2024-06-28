import rasterio as rio
import numpy as np
import pandas as pd
import os

def calc_mean_std(img_folder):
    """Calculate data mean and standard deviation

        Args:
            img_folder (str): Path to directory with geotiff data for which to calculate the scalers.
        Returns:
            Pandas Dataframe with mean and std for all available bands in input data. 
    """
    img_lst = []

    for img_file in os.listdir(img_folder):
        if '.tif' in img_file:
            imm = rio.open(os.path.join(img_folder,img_file)).read().astype('float32')
            img_lst.append(imm)

    img_lst_arr = np.array(img_lst)
    img_lst_arr[img_lst_arr==-9999] = np.nan
    mean = np.nanmean(img_lst_arr, axis=(0,2,3))
    std = np.nanstd(img_lst_arr, axis=(0,2,3))

    return( pd.DataFrame({'mean': mean,'std': std}) )
