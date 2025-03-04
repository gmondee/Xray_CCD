import csv
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

class analyzeSpectrum:
    def __init__(self,filename):
        self.filename = filename
        self.dEnd = 0
        self.dStart = 0
        self.dRoi = 0
        self.dCali = 0
        self.graph = 0
        self.contents = []
        self.calibrationPoints = []
        self.regionsOfInterest = []
        self.exposureTime = 0
        self.calib = 0
        self.data = np.zeros(2048)
        self.bins = np.arange(0,len(self.data))
        self.energies = 0
        self.areas = []
        self.run()

    #Intakes the csv file puts it into an easy to work with list
    def getDataWMarkers(self):
        with open(self.filename, mode ='r')as file:
            csvFile = csv.reader(file)
            for line in csvFile:
                self.contents.append(line)
        for index,line in enumerate(self.contents):
            if '<DATA>' in line[0]:
                self.dStart = index
            elif '<END>' in line[0]:
                self.dEnd = index
            elif 'LABEL' in line[0]:
                self.dCali = index
            elif '<ROI>' in line[0]:
                self.dRoi = index
            elif 'LIVE_TIME' in line[0]:
                self.exposureTime = float(line[0].split(' ')[2])

    #Seperates the region of interest ranges
    def getROI(self):
        for point in range(self.dRoi+1,self.dStart):
            nums = self.contents[point][0].split(' ')
            self.regionsOfInterest.append([int(nums[0]),int(nums[1])])

    #Seperates the calibration points
    def getCalibrationPoints(self):
        if self.dRoi:
            caliEnd = self.dRoi
        else:
            caliEnd = self.dStart
        for point in range(self.dCali+1,caliEnd):
            nums = self.contents[point][0].split(' ')
            self.calibrationPoints.append([float(nums[0]),float(nums[1])])

    #Seperates the data from the graph
    def getData(self):
        for point in range(self.dStart+1,self.dEnd):
            nums = self.contents[point]
            self.data[point-self.dStart-1] = float(nums[0])

    #Returns calibrated data as (x,y)
    def getCalibratedData(self):
        return (self.calib(self.bins), self.data)
    
    def getCalibratedNormalizedData(self):
        (x, y) = self.getCalibratedData()
        y = np.array(y)/self.exposureTime
        return (x,y)

    #Creates the graph
    def createGraph(self):
        x = []
        y = []
        for q in self.calibrationPoints:
            y.append(self.data[int(q[0])])
            x.append(q[0])
        plt.figure()
        plt.plot(self.calib(self.bins),self.data,label='Data')
        plt.plot(self.calib(x),y,'r.',label='Calibration Points')
        plt.xlabel('Energy')
        plt.ylabel('Count')
        plt.title(self.filename.split('.')[0])
        plt.legend()

    #Draws the Graph
    def drawGraph(self):
        plt.show()

    #Creates a fit line using the calibration points to have an energy scale
    def getEnergyFit(self):
        x = []
        y = []
        for point in self.calibrationPoints:
            x.append(point[0])
            y.append(point[1])
        self.calib = np.poly1d(np.polyfit(x,y,2))

    #Calculates the areas with each region of interest
    def calcAreas(self):
        for q in self.regionsOfInterest:
            area = sum(self.data[q[0]:q[1]])
            self.areas.append(area)

    #Reports the areas of ROI, this is a seperate function so it can be done multiple times without slowing down the program to recalculate the areas
    def reportAreas(self):
        print('For',str(self.filename))
        for q in range(len(self.areas)):
            print('The region spanning from',round(self.calib(self.regionsOfInterest[q][0]),3),'through',
                  round(self.calib(self.regionsOfInterest[q][1]),3),'has an area of ',self.areas[q])
            print('\n')

    #Perform all of the functions
    def run(self):
        self.getDataWMarkers()
        if self.dRoi:
            self.getROI()
            self.getCalibrationPoints()
            self.getData()
            self.getEnergyFit()
            self.calcAreas()
        if self.dRoi:
            self.reportAreas()
        else:
            print('There were no saved regions of interest to report the area of for',self.filename,'\n')
        self.createGraph()

if __name__ == '__main__':
    files = ['YOUR_FILE_NAME.mca']
    graphs = []
    for index,filename in enumerate(files):
        graphs.append(analyzeSpectrum(filename))
        graphs[index].run()
    graphs[0].drawGraph()