from astropy.io import fits
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import lmfit
from scipy.ndimage import gaussian_filter


mpl.use('QtAgg')
plt.ion()
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
    # plt.imshow(data[0].T, cmap='jet', vmin = 900, vmax = 1100)
    # plt.colorbar()
    # plt.title('raw no corrections, 120s')

    # Try and remove the worst of cosmic rays, could be improved using Grant's contour finding. replace points >1500 with the median
    data[data > 4600] = np.median(data[data < 975]) #2/25/2025: ~900 background and up to ~3300 ADU photons in strongest pixel

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
    tilt_amt = 23.0 # positive float
    for i, col in enumerate(bkg_removed.T):
        offset = int((i * tilt_amt) // W)
        if tilt_dir == 'CW':
            tilted[-(offset+L+1):-(offset+1), i] = col
        else:
            tilted[offset:offset+L, i] = col

    # Final plot and spectra
    # f, (ax1, ax2) = plt.subplots(2,1)
    # ax1.imshow(tilted.T, vmin=0, vmax=10, aspect='auto', label = 'tilt corrected')
    # plt.legend()
    # plt.title('tilt corrected')
    # #f, ax = plt.subplots()
    # ax2.plot(np.sum(tilted, axis=1), label = 'collapsed')
    # plt.title('collapsed')
    # plt.show()
    # plt.tight_layout()


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
##### R019
data_R019 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R019_8000eV_35mA_ArInject_01.fits")
R019 = HunterCrr(data_R019)
L=len(R019)
collapsedAvgR019 = list(np.zeros(int((windowWidth-1)/2))) #make two empty entries to 

for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR019.append(np.sum([R019[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR019 = np.array(collapsedAvgR019)
smoothR019 = gaussian_filter(R019, sigma=2)
##### R021
data_R021 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R021_8000eV_35mA_ArInject_01.fits")
R021 = HunterCrr(data_R021)
L=len(R021)
collapsedAvgR021 = list(np.zeros(int((windowWidth-1)/2))) #make two empty entries to 

for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR021.append(np.sum([R021[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR021 = np.array(collapsedAvgR021)
smoothR021 = gaussian_filter(R021, sigma=2)

##### R022
R022_files = [fr"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R022\R022_8000eV_30mA_ArInject_02_00{x:02}.fits" for x in range(0,13)] #0000-0012
data_R022 = combine_fits(R022_files)
R022 = HunterCrr(data_R022)
L=len(R022)
collapsedAvgR022 = list(np.zeros(int((windowWidth-1)/2)))
for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR022.append(np.sum([R022[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR022 = np.array(collapsedAvgR022)
smoothR022 = gaussian_filter(R022, sigma=2)

##### R023
data_R023 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R023_8000eV_35mA_ArInject_all.fits")
R023 = HunterCrr(data_R023)
L=len(R023)
collapsedAvgR023 = list(np.zeros(int((windowWidth-1)/2)))

for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR023.append(np.sum([R023[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR023 = np.array(collapsedAvgR023)
smoothR023 = gaussian_filter(R023, sigma=2)

##### R024
data_R024 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R024_8000eV_35mA_ArInject_all.fits")
R024 = HunterCrr(data_R024)
L=len(R024)
collapsedAvgR024 = list(np.zeros(int((windowWidth-1)/2)))

for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR024.append(np.sum([R024[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR024 = np.array(collapsedAvgR024)
smoothR024 = gaussian_filter(R024, sigma=2)

##### CRR analysis using all available data
listOfData = [data_R023, data_R024]#[data_R019, data_R021, data_R022, data_R023, data_R024]
allFramesInOne = combine_fits_after_opened(listOfData)
allCollapsed = HunterCrr(allFramesInOne)
L=len(allCollapsed)
collapsedAvgAll = list(np.zeros(int((windowWidth-1)/2)))
for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgAll.append(np.sum([allCollapsed[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgAll = np.array(collapsedAvgAll)
allCollapsedSmooth = gaussian_filter(allCollapsed, sigma=2)

##### Storing data for plots
collapsedData = {"R019":{"data":R019,"avg":collapsedAvgR019,"smooth":smoothR019},
                 "R021":{"data":R021,"avg":collapsedAvgR021,"smooth":smoothR021},
                 "R022":{"data":R022,"avg":collapsedAvgR022,"smooth":smoothR022},
                 "R023":{"data":R023,"avg":collapsedAvgR023,"smooth":smoothR023},
                 "R024":{"data":R024,"avg":collapsedAvgR024,"smooth":smoothR024},
                 "All" :{"data":allCollapsed,"avg":collapsedAvgAll,"smooth":allCollapsedSmooth}}
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

