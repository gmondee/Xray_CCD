from MCAutil import analyzeSpectrum
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import csv

mcaFiles = [r"C:\Users\Grant Mondeel\Box\CfA\MeVVA SDD\Data\A_9720beamE_02_19_25_delay1p998.mca", 
            r"C:\Users\Grant Mondeel\Box\CfA\MeVVA SDD\Data\B_9720beamE_02_19_25_delay1p995.mca",
            r"C:\Users\Grant Mondeel\Box\CfA\MeVVA SDD\Data\D_9720beamE_02_19_25_delay1p995_LongExposure.mca"]

cathodes = []
spectra = []
summedCalibrated = np.zeros(2048)
summedBins = np.zeros(2048)
for f in mcaFiles:
    cathodes.append(analyzeSpectrum(f))
    spectra.append(cathodes[-1].getCalibratedNormalizedData())
    summedCalibrated += np.array(spectra[-1][1])
    summedBins = spectra[-1][0]

plt.figure()
plt.plot(summedBins,gaussian_filter(summedCalibrated, sigma=3))

plt.figure()
for spect in spectra:
    plt.plot(summedBins, gaussian_filter(spect[1], sigma=2))
plt.xlabel("Energy (eV)")
plt.ylabel("Relative intensity (Arbs)")
plt.legend(["Ti", "Va", "Fe"])

rows = zip(summedBins, gaussian_filter(spectra[0][1], sigma=2), gaussian_filter(spectra[1][1], sigma=2), gaussian_filter(spectra[2][1], sigma=2))
with open("CathodeSpectra.txt", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Energy", "Ti", "Va", "Fe"])
    for row in rows:
        writer.writerow(row)