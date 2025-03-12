"""
CRR for x-ray crystal spectrometer
Written by Grant Mondeel, with code contributions from Hunter Staiger

Before proceeding:
You will need the .fits files. Ideally, each frame should be in the same .fits file, which is one of the
settings for saving files in Andor Solis. Otherwise, you can use the combine_fits() function to combine the frames by passing in
a list of file paths to each frame.
For Hunter's background removal to work, you need at least three frames. Otherwise, you will get a large background signal caused by
dead pixels. 
For the script to function, you need to place the GrantCRR.py file in the same folder as this file.

Navigating this script:
There are only a few lines that you might need to change in this script, and they are located around line 140.
Scroll past the four functions at the top of the script until you find a comment that looks like 
    ##### R049
Just above this, you will see some constants controlling how the data is processed after the CRR process. The defaults should be good enough.
Below the "##### R049" comment, we see a file path pointing to the R049 .fits file. In the next paragraph, I'll suggest a way to edit
this script to analyze any .fits file and possibly save the data to a .csv file.

How to use:
To make a script that analyses a fits file, we will do three steps.
(Optional:) make a copy of this script (File -> Save As...) and rename it to something related to the run number.
(1:)    We want to replace the previous run, R049, with the new one, R0XX. There are many variables that refer to this
            number, and to keep the names consistent, I simply replace R049 with R0XX. Specifically, I use Ctrl+f to search
            for R049, and I replace all instances with R0XX. This can be done all at once on Visual Studio Code.
(2:)    Next, we will specify the .fits file location. There is a call to the open_fits function with a file path already,
            and you will simply swap this out with the location of the R0XX .fits file. 
            As mentioned above, this is much easier if all frames are stored in a single .fits file, but I have provided an 
            example of how to combine thoseframes if they are stored separately.
(3:)    The script is now ready to run. Several plots should pop up. Most notably, the coadded spectrum is shown three ways:
            coadded, coadded+moving average, coadded+smoothed. 

Saving data:
I save the data in .csv format as rows of [pixel, intensity]. This part is disabled by default and is stored at the very
bottom of the script, and it looks like:
if False: #export data to a .csv file
    i=np.linspace(0,2199, num=2200)
    ... etc

You can set False to True and rerun the script, or simply paste this part into your python environment and run it after the main script.
The data will be saved in the current working directory, which might be wherever this script is on your computer.
"""



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

def open_fits(file_loc):
    hdu_list = fits.open(file_loc)
    return hdu_list[0].data

def combine_fits(file_loc_list):
    #used to paste individual frames together
    #file_loc_list is a list of paths of each frame to combine
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
            COM_pixels_all, pts_all, energies_all, energy_bins_all, sizes_all, frame_CRR = getEventEnergies(frame, events_rot, CRR=True, plot=makeCrrPlots, 
                                                                                                            median=med, MAX_CLUSTER_ENERGY=2500, MAX_CLUSTER_SIZE=11,
                                                                                                            MAX_PIXEL_ENERGY=2000)
            data[i]=frame_CRR

    if len(data)>2:
        combined = np.mean(data, axis=0) - np.median(data, axis=0)
    else:
        combined = np.mean(data, axis=0) - np.median(data)


    # # Remove background to help denoise image
    bkg_removed = combined.copy()
    bkg_removed[bkg_removed < -100] = 0

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
    plt.title('coadded frames with tilt correction')
    #f, ax = plt.subplots()
    ax2.plot(np.sum(tilted, axis=1), label = 'collapsed')
    plt.title('coadded frames with tilt correction, collapsed')
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

"""Here are some constants used to smooth the data. The defaults should be sufficient"""
windowWidth = 7 #must be odd. used for the sliding average.
Sigma = 3 #sigma setting for gaussian  blurring
makeCrrPlots=False #enables/disables plots showing what CRR is doing

"""Change this part below to analyze your new data, as described at the top of this file."""
##### R049
data_R049 = open_fits(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R049_xrayCal.fits")
R049 = HunterCrr(data_R049, do_Grant_CRR=True)
L=len(R049)
collapsedAvgR049 = list(np.zeros(int((windowWidth-1)/2))) #make two empty entries to 

for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
    collapsedAvgR049.append(np.sum([R049[windowInd] for windowInd in window.astype(int)])/windowWidth)
collapsedAvgR049 = np.array(collapsedAvgR049)
smoothR049 = gaussian_filter(R049, sigma=Sigma)

"""Example of how to combine frames saved individually into one:
    You need to change R037_files to a list of your new .fits files. In my case, the file paths only differ by the last two
    numbers, so I used a list comprehension. You can simply replace this with a list of strings, where each string is
    a path to one of the frames. Then, you would follow the analysis steps above, picking back up at L=len(R049)."""
##### R037
# R037_files = [fr"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Data\R037\R037_5000eV_35mA_ArInject_{x:02}.fits" for x in range(0,2)] #files ending in ...ArInject_00-01
# data_R037 = combine_fits(R037_files)
# R037 = HunterCrr(data_R037)

##### CRR analysis using all available data if there is more than one file to process
# listOfData = [data_R049]
# allFramesInOne = combine_fits_after_opened(listOfData)
# allCollapsed03122025 = HunterCrr(allFramesInOne)
# L=len(allCollapsed03122025)
# collapsedAvgAll = list(np.zeros(int((windowWidth-1)/2)))
# for window in [np.linspace(a,a+windowWidth-1,num=windowWidth) for a in np.linspace(0,L-windowWidth,num=L-windowWidth+1)]:
#     collapsedAvgAll.append(np.sum([allCollapsed03122025[windowInd] for windowInd in window.astype(int)])/windowWidth)
# collapsedAvgAll = np.array(collapsedAvgAll)
# allCollapsedSmooth = gaussian_filter(allCollapsed03122025, sigma=Sigma)

##### Storing data for plots
collapsedData = {"R049":{"data":R049,"avg":collapsedAvgR049,"smooth":smoothR049},
                 "All":{"data":R049,"avg":collapsedAvgR049,"smooth":smoothR049}}
                 #"All" :{"data":allCollapsed03122025,"avg":collapsedAvgAll,"smooth":allCollapsedSmooth}}
##### Plots
#plot all frames coadded with a moving average filter applied
plt.figure()
plt.plot(np.sum([collapsedData[key]["avg"] for key in collapsedData.keys() if key != "All"], axis=0))
plt.title(f'running average of coadded frames, {windowWidth} pixel width')

#plot all frames coadded
plt.figure()
plt.plot(np.sum([collapsedData[key]["data"] for key in collapsedData.keys() if key != "All"], axis=0))
plt.title("Coadded frames")
plt.legend()

if len(collapsedData) > 2: #if more than one .fits file is being analyzed, we can compare them here.
    #plot all coadded and smoothed spectra
    f, axes = plt.subplots(nrows=len(collapsedData.keys())+1, ncols=1, sharex=True, sharey=True)
    plt.suptitle("Smoothed spectra")
    for i, Run in enumerate(collapsedData.keys()):
        axes[i].plot(collapsedData[Run]["smooth"], label=Run)
        axes[i].legend()
    axes[-1].plot(np.sum([collapsedData[key]["smooth"] for key in collapsedData.keys() if key != "All"], axis=0), label='Added after analysis')
    axes[-1].legend()
    plt.tight_layout()

    #plot all coadded spectra
    f, axes = plt.subplots(nrows=len(collapsedData.keys())+1, ncols=1, sharex=True, sharey=True)
    plt.suptitle("Coadded spectra")
    for i, Run in enumerate(collapsedData.keys()):
        axes[i].plot(collapsedData[Run]["data"], label=Run)
        axes[i].legend()
    axes[-1].plot(np.sum([collapsedData[key]["data"] for key in collapsedData.keys() if key != "All"], axis=0), label='Added after analysis')
    axes[-1].legend()
    plt.tight_layout()

#plot all frames coadded and smoothed
plt.figure()
ax = plt.gca()
ax.plot(collapsedData["All"]["smooth"])
plt.ylabel("Intensity per pixel row (Arbs)")
plt.xlabel("Pixel row")
plt.title("All frames coadded and smoothed")
plt.tight_layout()

if False: #export data to a .csv file
    i=np.linspace(0,2199, num=2200)
    import csv
    rows = zip(i, collapsedData["All"]["smooth"])
    with open("03122025_R049.txt", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Pixel", "Intensity"])
        for row in rows:
            writer.writerow(row)