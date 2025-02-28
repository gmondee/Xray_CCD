from astropy.io import fits
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import lmfit
from importlib import reload
import GrantCRR
reload(GrantCRR)
from scipy.ndimage import gaussian_filter
from GrantCRR import findContours, getEventEnergies

mpl.use('QtAgg')
mpl.rcParams["image.interpolation"] = "none"
plt.ion()
makePlots=False
def open_fits(file_loc):
    hdu_list = fits.open(file_loc)
    return hdu_list[0].data

def combine_fits(file_loc_list):
    combined_fits = open_fits(file_loc_list[0]) #first file
    for file_loc in file_loc_list[1:]:
        combined_fits = np.concatenate((combined_fits, open_fits(file_loc)), axis=0)
    return combined_fits

def combine_fits_after_opened(data_list):
    #stacks exposures to be analyzed together in HunterCrr(). Assumes each frame has equal exposure time.
    allFrames = data_list[0]
    for frame in data_list[1:]:
        allFrames = np.concatenate((allFrames, frame), axis=0)
    return allFrames


def HunterCrr(data):
    # Plot histogram of intensity values
    # f, ax = plt.subplots()
    # #plt.hist(data.ravel(), bins=1000, label ='hist of data')
    # #plt.yscale('log')
    # #plt.legend()
    # # Image without any corrections
    # f, ax = plt.subplots()
    # plt.figure()
    # plt.imshow(np.sum(data.T, axis=0), cmap='jet', vmin = 900, vmax = 1100)
    # plt.colorbar()
    # plt.title('raw no corrections, 120s')

    # Try and remove the worst of cosmic rays, could be improved using Grant's contour finding. replace points >1500 with the median
    data[data > 1700] = np.median(data[data < 975]) #2/25/2025: ~900 background and up to ~3300 ADU photons in strongest pixel
    data[:,:,1312:1490]=np.median(data[data < 975]) ### remove the bright column
    data[:,0:160,:]=np.median(data[data < 975]) ### remove bright row along the edge of the sensor
    dataPreCRR=np.copy(data)
    #Grant CRR, acts on individual frames
    for i, frame in enumerate(data):
        med = np.median(frame[frame<975]) #median
        stderr = np.std(frame[frame<975]) #stdev of the noise
        frame_out, events_rot, lowLevelDisc, contours = findContours(frame, median = med, stderr = stderr, plot=False)
        COM_pixels_all, pts_all, energies_all, energy_bins_all, sizes_all, frame_CRR = getEventEnergies(frame, events_rot, CRR=True, plot=makePlots, 
                                                                                                        median=med, MAX_CLUSTER_ENERGY=2500, MAX_CLUSTER_SIZE=11,
                                                                                                        MAX_PIXEL_ENERGY=2000)
        data[i]=frame_CRR


    # # Plot to show how mean includes signal while median (mostly) doesn't
    # f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # plt.sca(axs[0])
    # plt.imshow(np.mean(data, axis=0).T, vmin=850, vmax=1100, label = 'mean')
    # plt.title('mean')
    # plt.sca(axs[1])
    # plt.imshow(np.median(data, axis=0).T, vmin=850, vmax=1100, label ='median')
    # plt.title('median')
    # meads =np.median(data, axis=0)
    combined = np.mean(data, axis=0) - np.median(data, axis=0)

    # print(combined.shape)
    # # New histogram that shows noise gaussian close to 0
    # f, ax = plt.subplots()
    # plt.hist(combined.ravel(), bins=1024, label = 'mean-median hist')
    # plt.title('mean- median')
    # plt.yscale('log')

    # f, ax = plt.subplots()
    # plt.imshow(combined.T, vmin=0, vmax=10, aspect='auto', cmap='inferno', label = 'transposed')
    # plt.colorbar()
    # plt.title('transposed')
    # # Remove background to help denoise image
    bkg_removed = combined.copy()
    bkg_removed[bkg_removed < -100] = 0
    # f, ax = plt.subplots()
    # plt.imshow(bkg_removed.T, vmin=0, vmax=10, aspect='auto', cmap='inferno', label = 'bkg removed')
    # plt.colorbar()
    # plt.title('bkg removed')

    # Shift rows to align image
    tilted = np.zeros((2200, 2048))
    L, W = bkg_removed.shape

    tilt_dir = 'CW' # CW or CCW
    tilt_amt = 17.0 # positive float
    for i, col in enumerate(bkg_removed.T):
        offset = int((i * tilt_amt) // W)
        if tilt_dir == 'CW':
            tilted[-(offset+L+1):-(offset+1), i] = col
        else:
            tilted[offset:offset+L, i] = col

    # Final plot and spectra
    f, (ax1, ax2) = plt.subplots(2,1)
    ax1.imshow(tilted.T, vmin=0, vmax=10, aspect='auto', label = 'tilt corrected')
    plt.legend()
    plt.title('tilt corrected')
    #f, ax = plt.subplots()
    ax2.plot(np.sum(tilted, axis=1), label = 'collapsed')
    plt.title('collapsed')
    plt.show()
    plt.tight_layout()


    #binned spectra
    collapsed = np.sum(tilted, axis=1)
    # binSize = 6 #in pixels
    # L, W = tilted.shape
    # collapsedBinned = []
    # collapsedBinEdges = [0]

    # # i=1
    # _binCounts = 0.
    # for pixel in collapsed:
    #     _binCounts += pixel
    #     i+=1
    #     if i>binSize:
    #         i=1
    #         collapsedBinned.append(_binCounts)
    #         collapsedBinEdges.append(collapsedBinEdges[-1]+i)
    #         _binCounts=0
    # # plt.figure()
    # plt.gca().plot(collapsedBinned, label='binned')
    return np.array(collapsed)


windowWidth = 7 #must be odd

##### R030
data_R030 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\02282025\15min\R030_8000eV_30mA_ArInjec.fits")
R030 = HunterCrr(data_R030)
L=len(R030)
collapsedAvgR030 = list(np.zeros(int((windowWidth-1)/2))) #make two empty entries to 

for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR030.append(np.sum([R030[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR030 = np.array(collapsedAvgR030)
smoothR030 = gaussian_filter(R030, sigma=2)

##### R031
data_R031 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\02282025\15min\R031_8000eV_30mA_ArInjec.fits")
R031 = HunterCrr(data_R031)
L=len(R031)
collapsedAvgR031 = list(np.zeros(int((windowWidth-1)/2))) #make two empty entries to 

for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR031.append(np.sum([R031[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR031 = np.array(collapsedAvgR031)
smoothR031 = gaussian_filter(R031, sigma=2)

##### CRR analysis using all available data
listOfData = [data_R030, data_R031]
allFramesInOne = combine_fits_after_opened(listOfData)
allCollapsed02282025 = HunterCrr(allFramesInOne)
L=len(allCollapsed02282025)
collapsedAvgAll = list(np.zeros(int((windowWidth-1)/2)))
for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgAll.append(np.sum([allCollapsed02282025[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgAll = np.array(collapsedAvgAll)
allCollapsedSmooth = gaussian_filter(allCollapsed02282025, sigma=2)

##### Storing data for plots
collapsedData = {"R030":{"data":R030,"avg":collapsedAvgR030,"smooth":smoothR030},
                 "R031":{"data":R031,"avg":collapsedAvgR031,"smooth":smoothR031},
                 "All" :{"data":allCollapsed02282025,"avg":collapsedAvgAll,"smooth":allCollapsedSmooth}}
##### Plots

plt.figure()
plt.plot(np.sum([collapsedData[key]["avg"] for key in collapsedData.keys() if key != "All"], axis=0))
plt.title(f'running average coadded, {windowWidth} pixel width')

plt.figure()
plt.plot(np.sum([collapsedData[key]["data"] for key in collapsedData.keys() if key != "All"], axis=0), label='coadded, no bin/avg')
#plt.plot(R022, label='R022')
plt.legend()

f, axes = plt.subplots(nrows=len(collapsedData.keys())+1, ncols=1, sharex=True, sharey=True)
plt.suptitle("Smoothed")
for i, Run in enumerate(collapsedData.keys()):
    axes[i].plot(collapsedData[Run]["smooth"], label=Run)
    axes[i].legend()
axes[-1].plot(np.sum([collapsedData[key]["smooth"] for key in collapsedData.keys() if key != "All"], axis=0), label='Added after analysis')
axes[-1].legend()
plt.tight_layout()

f, axes = plt.subplots(nrows=len(collapsedData.keys())+1, ncols=1, sharex=True, sharey=True)
plt.suptitle("Collapsed")
for i, Run in enumerate(collapsedData.keys()):
    axes[i].plot(collapsedData[Run]["data"], label=Run)
    axes[i].legend()
axes[-1].plot(np.sum([collapsedData[key]["data"] for key in collapsedData.keys() if key != "All"], axis=0), label='Added after analysis')
axes[-1].legend()
plt.tight_layout()

