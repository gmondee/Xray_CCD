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


def HunterCrr(data, do_Grant_CRR=True):
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
    if do_Grant_CRR:
        for i, frame in enumerate(data):
            med = np.median(frame[frame<975]) #median
            stderr = np.std(frame[frame<975]) #stdev of the noise
            frame_out, events_rot, lowLevelDisc, contours = findContours(frame, median = med, stderr = stderr, plot=False, LLD_sigma=5)
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
    print(f"{len(data)=}")
    if len(data)>2:
        combined = np.mean(data, axis=0) - np.median(data, axis=0)
    else:
        combined = np.mean(data, axis=0) - np.median(data)

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
Sigma = 3

##### R037
R037_files = [fr"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R037\R037_5000eV_35mA_ArInject_{x:02}.fits" for x in range(0,2)] #00-01
data_R037 = combine_fits(R037_files)
R037 = HunterCrr(data_R037)
L=len(R037)
collapsedAvgR037 = list(np.zeros(int((windowWidth-1)/2)))
for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR037.append(np.sum([R037[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR037 = np.array(collapsedAvgR037)
smoothR037 = gaussian_filter(R037, sigma=Sigma)

##### R037
# data_R037 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R037\R037_5000eV_35mA_ArInject_01.fits")
# R037 = HunterCrr(data_R037, do_Grant_CRR=True)
# L=len(R037)
# collapsedAvgR037 = list(np.zeros(int((windowWidth-1)/2))) #make two empty entries to 

# for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
#     collapsedAvgR037.append(np.sum([R037[windowInd] for windowInd in window.astype(int)])/windowWidth)
# collapsedAvgR037 = np.array(collapsedAvgR037)
# smoothR037 = gaussian_filter(R037, sigma=Sigma)




##### CRR analysis using all available data
# listOfData = [data_R037]
# allFramesInOne = combine_fits_after_opened(listOfData)
# allCollapsed03112025 = HunterCrr(allFramesInOne)
# L=len(allCollapsed03112025)
# collapsedAvgAll = list(np.zeros(int((windowWidth-1)/2)))
# for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
#     collapsedAvgAll.append(np.sum([allCollapsed03112025[windowInd] for windowInd in window.astype(int)])/windowWidth)
# collapsedAvgAll = np.array(collapsedAvgAll)
# allCollapsedSmooth = gaussian_filter(allCollapsed03112025, sigma=Sigma)

##### Storing data for plots
collapsedData = {"R037":{"data":R037,"avg":collapsedAvgR037,"smooth":smoothR037},
                 "All":{"data":R037,"avg":collapsedAvgR037,"smooth":smoothR037}}
                 #"All" :{"data":allCollapsed03112025,"avg":collapsedAvgAll,"smooth":allCollapsedSmooth}}
##### Plots

plt.figure()
plt.plot(np.sum([collapsedData[key]["avg"] for key in collapsedData.keys() if key != "All"], axis=0))
plt.title(f'running average coadded, {windowWidth} pixel width')

plt.figure()
plt.plot(np.sum([collapsedData[key]["data"] for key in collapsedData.keys() if key != "All"], axis=0), label='coadded, no bin/avg')
#plt.plot(R037, label='R037')
plt.legend()

f, axes = plt.subplots(nrows=len(collapsedData.keys())+1, ncols=1, sharex=True, sharey=True)
plt.suptitle("Smoothed")
for i, Run in enumerate(collapsedData.keys()):
    axes[i].plot(collapsedData[Run]["smooth"], label=Run)
    axes[i].legend()
axes[-1].plot(np.sum([collapsedData[key]["smooth"] for key in collapsedData.keys() if key != "All"], axis=0), label='Added after analysis')
axes[-1].legend()
plt.tight_layout()

plt.figure()
ax = plt.gca()
ax.plot(collapsedData["All"]["smooth"])
plt.ylabel("Intensity per pixel row (Arbs)")
plt.xlabel("Pixel row")
plt.title("All data, smoothed")
plt.tight_layout()

if False:
    i=np.linspace(0,2199, num=2200)
    import csv
    rows = zip(i, collapsedData["All"]["smooth"])
    with open("03112025_R037.txt", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Pixel", "Intensity"])
        for row in rows:
            writer.writerow(row)


f, axes = plt.subplots(nrows=len(collapsedData.keys())+1, ncols=1, sharex=True, sharey=True)
plt.suptitle("Collapsed")
for i, Run in enumerate(collapsedData.keys()):
    axes[i].plot(collapsedData[Run]["data"], label=Run)
    axes[i].legend()
axes[-1].plot(np.sum([collapsedData[key]["data"] for key in collapsedData.keys() if key != "All"], axis=0), label='Added after analysis')
axes[-1].legend()
plt.tight_layout()