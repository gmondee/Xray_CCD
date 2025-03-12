import matplotlib as mpl
mpl.use('QtAgg')
mpl.rcParams["image.interpolation"] = "none"
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import lmfit
import pathlib

spectraPath = pathlib.Path(r"C:\Users\Grant Mondeel\Box\CfA\Xray Crystal\Xray_CCD")
spectraFiles = list(spectraPath.glob("0*.txt"))

numSpectra = len(spectraFiles)
fig = plt.figure()
ax = plt.gca()
verticalBuffer = 3000
xs = np.linspace(0,2199, num=2200)
for i, spectrumFile in enumerate(spectraFiles):
    spectrum= pd.read_csv(spectrumFile, delimiter=",")
    data = np.array(spectrum["Intensity"].to_list())
    ax.plot(xs, data-i*verticalBuffer, label=str(spectraFiles[i]).split('\\')[-1][:-4])
plt.legend()
