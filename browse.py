from __future__ import print_function
import ROOT
from ROOT import TChain
from larcv import larcv
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import math

from larcv import larcv

# A utility function to compute the 2D (X,Y) range to zoom-in so that it avoids showing zero region of an image.
def get_view_range(image2d):
    nz_pixels=np.where(image2d>0.0)
    ylim = (np.min(nz_pixels[0])-5,np.max(nz_pixels[0])+5)
    xlim = (np.min(nz_pixels[1])-5,np.max(nz_pixels[1])+5)
    # Adjust for allowed image range
    ylim = (np.max((ylim[0],0)), np.min((ylim[1],image2d.shape[1]-1)))
    xlim = (np.max((xlim[0],0)), np.min((xlim[1],image2d.shape[0]-1)))
    return (xlim,ylim)

def show_event(entry=-1):
    # Create TChain for data image
    chain_image2d = ROOT.TChain('image2d_data_tree')
    chain_image2d.AddFile('data/test_10k.root')
    # Create TChain for label image
    chain_label2d = ROOT.TChain('image2d_segment_tree')
    chain_label2d.AddFile('data/test_10k.root')
    
    if entry < 0:
        entry = np.random.randint(0,chain_label2d.GetEntries())

    chain_label2d.GetEntry(entry)
    chain_image2d.GetEntry(entry)

    # Let's grab a specific projection (1st one)
    image2d = larcv.as_ndarray(chain_image2d.image2d_data_branch.as_vector().front())
    label2d = larcv.as_ndarray(chain_label2d.image2d_segment_branch.as_vector().front())

    # Get image range to focus
    xlim, ylim = get_view_range(image2d)
    
    # Dump images
    fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(18,12), facecolor='w')
    ax0.imshow(image2d, interpolation='none', cmap='jet', origin='lower')
    ax1.imshow(label2d, interpolation='none', cmap='jet', origin='lower',vmin=0., vmax=3.1)
    ax0.set_title('Data',fontsize=20,fontname='Georgia',fontweight='bold')
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax1.set_title('Label',fontsize=20,fontname='Georgia',fontweight='bold')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    plt.show()
    
    return (np.array(image2d), np.array(label2d))

#the lines commented out should be included to see the scatter plot of the shower produced
def plot_best_fit(image_array):
    weights = np.array(image_array)
    x = np.where(weights>0)[1]
    y = np.where(weights>0)[0]
    #plt.plot(x, y,'.')
    size = len(image_array) * len(image_array[0])

    y = np.zeros((len(image_array), len(image_array[0])))
    for i in range(len(np.where(weights>0)[0])):
	y[np.where(weights>0)[0][i]][np.where(weights>0)[1][i]] = np.where(weights>0)[0][i]
    y = y.reshape(size)
    x = np.array(range(len(image_array)) * len(image_array[0]))
    weights = weights.reshape((size))
    b, m = polyfit(x, y, 1, w=weights)
    angle = math.atan(m) * 180/math.pi
    #plt.plot(x, b+m*x, '-')
    return b,m, angle
# Let's look at one specific event entry
ENTRY = 783
image2d, label2d = show_event(ENTRY)


unique_values, unique_counts = np.unique(label2d, return_counts=True)
print('Label values:',unique_values)
print('Label counts:',unique_counts)

categories = ['Background','Shower','Track']

fig, axes = plt.subplots(1, len(unique_values), figsize=(18,12), facecolor='w')
xlim,ylim = get_view_range(image2d)

for index, value in enumerate(unique_values):
    ax = axes[index]
    mask = (label2d == value)
    Image = image2d*mask
    ax.imshow(Image, interpolation='none', cmap='jet', origin='lower')
    print("Event", categories[index])
    print("{0:.2f} MeV".format(np.sum((Image)/100)))
    ax.set_title(categories[index],fontsize=20,fontname='Georgia',fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if index==1:
       showerImage = np.array(Image)
       b, m, angle=plot_best_fit(showerImage)
       ax.set_title("{0} {1:.2f} deg".format(categories[index], angle),fontsize=20,fontname='Georgia',fontweight='bold')
       print("Angle of the shower: {0:.2f}".format(angle))
       x = np.array(range(0,len(showerImage[0])))
       y = x*m + b
       ax.plot(x, y)
plt.show()
