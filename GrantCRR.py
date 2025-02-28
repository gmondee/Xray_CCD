import numpy as np
import matplotlib.pyplot as plt
import cv2

def findContours(im, median=None, stderr=None, plot=True):
  ### Make a binary image to find the contours. For the CCD camera, cosmic rays and x-rays both bleed into multiple pixels.
  if median==None:
    #Find a rough median value for the background
    median=np.median(im)
      
  if stderr==None:
    #find the deviation in the noise using counts up to 20% higher than the median in the image
    #nonZeroIm = im[im!=0]
    stderr=np.std(im[im<=(np.mean(im)/2)])  #try to find the stderr of the noise only by assuming it is within 10% of the median
  print(f'{median=}, {stderr=}')
  binary_image = np.ndarray(im.shape, np.uint8) #create a blank image with the same shape as our image
  binary_image[:,:]=1                 #set all values to 1
  lowLevelDisc = max(5*stderr, 5)
  print(f'{lowLevelDisc=}')
  binary_image[im<=(lowLevelDisc+median)]=0  #cut out background within a few sigma of the median
  #im=im-(lowLevelDisc)  #shift all counts down by the low level discriminator
  #im[binary_image<1]=0 #delete bg pixels
  contours, hierarchy = cv2.findContours(binary_image,cv2.RETR_LIST ,cv2.CHAIN_APPROX_NONE)
  if plot==True:
    plt.figure()
    plt.imshow(im, vmin=0, vmax=median+10*stderr)
    plt.colorbar()
    plt.title("image")
    plt.figure()
    plt.imshow(im)
    plt.imshow(cv2.drawContours(np.copy(im-median-stderr), contours, -1, (255,255,255), -1), vmin=0, vmax=255)
    plt.title("Contours")
    plt.colorbar()

  # Large events have pixels inside of the contour that we need to add in.
  # events will be a list containing "events", 
  # e.g. for a 2-pixel event: [ [[x1,y1], energy1], [[x2,y2], energy2]] ]
  events = []

  # For each list of contour points...
  for i in range(len(contours)):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros(im.shape)
    cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

    # Access the image pixels and create a 1D numpy array.Then, add the pixel coordinates and ADU value to the list of pixels
    pts = np.where(cimg == 255)

    pxs = [np.transpose([pts[1], pts[0]]), im[pts[0], pts[1]]]
    events.append([px for px in zip(pxs[0], pxs[1])]) 
  #im=np.clip(im+(lowLevelDisc), 0, 100000) #undo the subtraction but subtract some noise level
  return im, events, lowLevelDisc, contours

def getEventEnergies(im, events, median, CRR=True, plot=False, MAX_CLUSTER_SIZE=8, MAX_CLUSTER_ENERGY = 1700, MAX_PIXEL_ENERGY = 1500):
  #   MAX_CLUSTER_SIZE = 8 #in pixels
  #   MAX_CLUSTER_ENERGY = 1700#2400 #in units of ADU after a background has already been subtracted. How should we set this?   
  #   MAX_PIXEL_ENERGY = 1500#2000#.65*MAX_CLUSTER_ENERGY  #800
  ### Combine events into single pixels placed at the center-of-mass with the combined energy. We've already subtracted out the median background.
  # COM is defined as (m1*x1 + m2*x2 + ... mn*xn)/(m1 + m2 + ... + mn) where "m" is replaced with the ADU value
  COM_pixels = []
  for event in events:
    total_energy = 0.0
    COM_x_arr = [] #x_n, m_n
    COM_y_arr = [] #y_n, m_n
    COM_energies_arr = []
    for pixel in event:
      energy = pixel[1] #in ADU
      total_energy = total_energy+np.clip(energy-median,0,1000000)
      COM_x_arr.append(pixel[0][0])
      COM_y_arr.append(pixel[0][1])
      COM_energies_arr.append(energy)
    ### make a new pixel at the COM
    COM_x = np.dot(COM_x_arr,COM_energies_arr)/sum(COM_energies_arr)
    COM_y = np.dot(COM_y_arr,COM_energies_arr)/sum(COM_energies_arr)
    if CRR==True:
      if (total_energy>MAX_CLUSTER_ENERGY or len(event)>MAX_CLUSTER_SIZE) or any(np.array(COM_energies_arr)>MAX_PIXEL_ENERGY): ## if above either threshold, set its pixels to 0
        for pixel in event:
          im[pixel[0][1], pixel[0][0]] = median #pixel[0] is [x y]
        total_energy = MAX_CLUSTER_ENERGY+1 #flag it in the case of MAX_PIXEL_ENERGY being exceeded
    COM_pixels.append([[COM_x, COM_y], total_energy, len(event)])

  # # Generate a flat image of the center-of-mass pixels (no energy information)
  # plt.figure()
  # pts = np.array([pixel[0] for pixel in COM_pixels])
  # plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts])
  # plt.gca().invert_yaxis()
  if plot == True:
    # Generate a histogram of the photon/CR energies
    energies = np.array([pixel[1] for pixel in COM_pixels])
    plt.figure()
    energy_bins = np.linspace(0,max(int(np.median(energies)*5),2000), num=int(len(energies)/3))
    plt.hist(energies, energy_bins)
    plt.xlabel("ADU")
    plt.ylabel(f"Counts per {int(np.median(energies)*5)/int(len(energies)/3)} ADU bin")
    plt.title("Cluster energy histogram")

    #size hist, add hists from other files
    # Generate a histogram of the event sizes
    sizes = np.array([pixel[2] for pixel in COM_pixels])
    plt.figure()
    size_bins = np.linspace(0, 50, 51)
    plt.hist(sizes, size_bins)
    plt.xlabel("Cluster size [Pixels]")
    plt.ylabel("Number of clusters")
    plt.title("Cluster size histogram")

    # Generate a 2D plot with COM pixels and their energies
    plt.figure()
    pts = np.array([pixel[0] for pixel in COM_pixels])
    plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], c=energies)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title("Cluster energies")

    # Generate a 2D plot with COM pixels and their sizes
    plt.figure()
    pts = np.array([pixel[0] for pixel in COM_pixels])
    plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], c=sizes)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title("Cluster sizes")

  ###Cluster CRR
  if CRR==True:
    ###Remove very large or very energetic clusters
    COM_pixels_CRR = [px_crr for px_crr in COM_pixels if (px_crr[2]<=MAX_CLUSTER_SIZE and px_crr[1]<=MAX_CLUSTER_ENERGY)]
    ##repopulate the pts, energies, energy_bins, and sizes lists now that we've cut pixels out
    pts = [pixel[0] for pixel in COM_pixels_CRR]
    energies = [pixel[1] for pixel in COM_pixels_CRR]
    energy_bins = np.linspace(0,max(int(np.median(energies)*5),2000), num=int(len(energies)/3))
    sizes = [pixel[2] for pixel in COM_pixels_CRR]
      

    # Generate a 2D plot with COM pixels showing which are removed by CRR
    if plot == True:
      plt.figure()
      pts_before = np.array([pixel[0] for pixel in COM_pixels])
      plt.scatter([pt[0] for pt in pts_before], [pt[1] for pt in pts_before], c='r')
      pts_crr = np.array([pixel[0] for pixel in COM_pixels_CRR])
      plt.scatter([pt[0] for pt in pts_crr], [pt[1] for pt in pts_crr], c='g')      
      plt.gca().invert_yaxis()
      plt.title("Removed pixels")

    return COM_pixels, pts, energies, energy_bins, sizes, im
  else:
    return COM_pixels, pts, energies, energy_bins, sizes, im