"https://towardsdatascience.com/visualising-the-rgb-channels-of-satellite-images-with-python-6d541af1f98d"
import matplotlib
matplotlib.use("TkAgg")
from pyramids.dataset import Dataset
from osgeo import gdal

import numpy as np
import matplotlib.pyplot as plt

import glob
#%%
#Load paths
paths = glob.glob("examples/medium/SWED/test/images/*")
#Load first image
img = gdal.Open(paths[0]).ReadAsArray()
# np.save("examples/medium/s2a.npy", img)
dataset = Dataset.read_file(paths[0])
img.shape #(12,256,256)
dataset.shape
dataset.read_array()
#
#Get RGB image
# RGB 3, 2, 1
rgb = img[[3,2,1]].transpose(1,2,0)

# Pixel range
print(rgb.min(),rgb.max()) #150 8,600
#%%
"""
The sentinel-2 images have a maximum reflectance value of 10000. Although pixels can 
occasionally have values higher than this, 
we can ignore these large values when visualising the RGB channels. So, we scale images by 
dividing by 10000 and clipping them between 0 and 1. 

This ensures all pixel values will be between 0 and 1.
"""
# Scale image
rgb = np.clip(rgb/10000, 0, 1)
#%%
#Display RGB image
plt.imshow(rgb)
#%%
#Display histograms of pixel intesity for each band
fig, axs = plt.subplots(1,3,figsize=(18,5))
fig.patch.set_facecolor('xkcd:white')

labels = ['Red','Green','Blue']
for i, ax in enumerate(axs):
    ax.hist(rgb[:,:,i].flatten(),bins=100)
    ax.set_title(labels[i],size=20,fontweight="bold")
    ax.set_xlabel("Pixel Value",size=15)
    ax.set_ylabel("Frequency",size =15)
#%%
# Clip RGB image to 0.3
rgb = np.clip(rgb,0,0.3)/0.3

plt.imshow(rgb)
#%%
def visualise_rgb(img, clip=[0.3, 0.3, 0.3], display=True):
    """Visulaise RGB image with given clip values and return image"""

    # Scale image
    img = np.clip(img / 10000, 0, 1)

    # Get RGB channels
    rgb = img[[3, 2, 1]]

    # clip rgb values
    rgb[0] = np.clip(rgb[0], 0, clip[0]) / clip[0]
    rgb[1] = np.clip(rgb[1], 0, clip[1]) / clip[1]
    rgb[2] = np.clip(rgb[2], 0, clip[2]) / clip[2]

    rgb = rgb.transpose(1, 2, 0)

    if display:

        # Display histograms of pixel intesity with given clip values
        fig, axs = plt.subplots(1, 4, figsize=(22, 5))
        fig.patch.set_facecolor('xkcd:white')

        labels = ['Red', 'Green', 'Blue']
        for i, ax in enumerate(axs[0:3]):
            ax.hist(img[3 - i].flatten(), bins=100)
            ax.set_title(labels[i], size=20, fontweight="bold")
            ax.axvline(clip[i], color="red", linestyle="--")
            ax.set_yticks([])

        # Display RGB image
        axs[3].imshow(rgb)
        axs[3].set_title("RGB", size=20, fontweight="bold")
        axs[3].set_xticks([])
        axs[3].set_yticks([])

    return rgb
#%%

img = gdal.Open(paths[0]).ReadAsArray()
rgb = visualise_rgb(img,[0.3,0.3,0.3])
#%% Adjusting brightness
"""
Different images will have different optimal cutoffs. In fact, the cover image of this article was created using 3 
different values. 
The function above makes it easy to adjust them. 
In Figure 5, you can see how adjusting the cutoffs will change the brightness.
"""
rgb_1 = visualise_rgb(img,[0.15,0.15,0.15],display=True)
rgb_2 = visualise_rgb(img,[0.3,0.3,0.3],display=True)
rgb_3 = visualise_rgb(img,[0.45,0.45,0.45],display=True)
#%%
rgb_1 = visualise_rgb(img,[0.2,0.3,0.3],display=True)
rgb_2 = visualise_rgb(img,[0.3,0.3,0.2],display=True)
rgb_3 = visualise_rgb(img,[0.3,0.2,0.3],display=True)