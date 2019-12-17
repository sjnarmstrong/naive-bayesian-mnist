import numpy as np
import matplotlib.pyplot as plt
from os.path import join as appendFileToFolder
from os import linesep
import cv2
import json
from PartBFeatureTypes import classedFeature, trainedContinousFeature, continousFeature

filesTest=['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
files=['train-images-idx3-ubyte','train-labels-idx1-ubyte']

# code from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
#accessed 03/03/29017
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

with open(appendFileToFolder('mnist',files[0]), 'rb') as imageFile:
  with open(appendFileToFolder('mnist',files[1]), 'rb') as labelFile:  
      dt = np.dtype(np.int32)
      dt = dt.newbyteorder('>')
     
      
      magicNumber=np.frombuffer(labelFile.read(4), dtype=dt)
      numDigets=np.frombuffer(labelFile.read(4), dtype=dt)
      
      magicNumber=np.frombuffer(imageFile.read(4), dtype=dt)
      numDigets=np.frombuffer(imageFile.read(4), dtype=dt)
      dimention=np.frombuffer(imageFile.read(8), dtype=dt)##rows,cols
      
      
      pixelCount=dimention[0]*dimention[1]
      
      
      TrainImgs=np.fromfile(imageFile, dtype=np.uint8).reshape(np.append(-1,dimention))
      Trainlbls = np.fromfile(labelFile, dtype=np.int8)
      TrainBigImgs = []
      
      for image in TrainImgs:
        ret,threshImage = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
        
        im2, contours, hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        maxContour=max(contours,key=cv2.contourArea)
        
        rect = cv2.minAreaRect(maxContour)
        box = order_points(cv2.boxPoints(rect))
        dst = np.array([[0, 0],[31, 0],[31, 31],[0, 31]], dtype = "float32")
          
        M = cv2.getPerspectiveTransform(box, dst)
        bigImg = cv2.warpPerspective(image, M, (32, 32))
        TrainBigImgs.append(bigImg)
     
with open(appendFileToFolder('mnist',files[0]), 'rb') as imageFile:
  with open(appendFileToFolder('mnist',files[1]), 'rb') as labelFile:  
      dt = np.dtype(np.int32)
      dt = dt.newbyteorder('>')
     
      
      magicNumber=np.frombuffer(labelFile.read(4), dtype=dt)
      numDigets=np.frombuffer(labelFile.read(4), dtype=dt)
      
      magicNumber=np.frombuffer(imageFile.read(4), dtype=dt)
      numDigets=np.frombuffer(imageFile.read(4), dtype=dt)
      dimention=np.frombuffer(imageFile.read(8), dtype=dt)
      
      TestImgs=np.fromfile(imageFile, dtype=np.uint8).reshape(np.append(-1,dimention))
      Testlbls = np.fromfile(labelFile, dtype=np.int8)
      
      TestBigImgs = []
      
      for image in TrainImgs:
        ret,threshImage = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
        
        im2, contours, hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        maxContour=max(contours,key=cv2.contourArea)
        
        rect = cv2.minAreaRect(maxContour)
        box = order_points(cv2.boxPoints(rect))
        dst = np.array([[0, 0],[31, 0],[31, 31],[0, 31]], dtype = "float32")
          
        M = cv2.getPerspectiveTransform(box, dst)
        bigImg = cv2.warpPerspective(image, M, (32, 32))
        TestBigImgs.append(bigImg)
 


     
      
def doTest(hogDesccv,winStride,padding):
    locations = ((0,0),)
    hogDescriptors=[]
    
    for bigImg,label in zip(TrainBigImgs,Trainlbls):

        hist = hogDesccv.compute(bigImg,winStride,padding,locations)
    
        if len(hogDescriptors)==0:
            for val in hist:
                newHog=continousFeature()
                newHog.addTrainingObservation(val,label)
                hogDescriptors.append(newHog)
        else:
           for val,hog in zip(hist,hogDescriptors): 
                hog.addTrainingObservation(val,label)
    
    countMatrix=np.zeros((10,10))            
    for i in range(len(hogDescriptors)):
        hogDescriptors[i]=hogDescriptors[i].getTrainedFeature()
    
    
    for bigImg,label in zip(TestBigImgs,Testlbls):    
        posteriour=np.repeat(1/10,10)
        hist = hogDesccv.compute(bigImg,winStride,padding,locations)    
        
        for i,hogV in enumerate(zip(hogDescriptors,hist)):
            Hog,v=hogV
            posteriour*=Hog.getPclassesGivenFeature(v)
            
        predicted=np.argmax(posteriour)
        actualLabel=label
            
        countMatrix[predicted][actualLabel]+=1
    
    print("accuracy is: "+str(countMatrix.trace()/np.sum(countMatrix)*100)+"%")
 
for nbins in range(1,9):   
    print("________________________________________Test1")    
    winSize = (32,32)
    blockSize = (16,16)
    blockStride = (16,16)
    cellSize = (8,8)
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    hogDesccv = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
    winStride = (16,16)
    padding = (0,0)
    
    doTest(hogDesccv,winStride,padding)
    
for L2HysThreshold in np.arange(0.05,0.3,0.05):   
    print("________________________________________Test1")    
    winSize = (32,32)
    blockSize = (16,16)
    blockStride = (16,16)
    cellSize = (8,8)
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    nbins = 6
    gammaCorrection = 0
    nlevels = 64
    hogDesccv = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
    winStride = (16,16)
    padding = (0,0)
    
    doTest(hogDesccv,winStride,padding)

print("________________________________________Test1")    
winSize = (16,16)
blockSize = (16,16)
blockStride = (16,16)
cellSize = (8,8)
derivAperture = 1
winSigma = 4.
nbins=6
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
hogDesccv = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

winStride = (16,16)
padding = (0,0)

doTest(hogDesccv,winStride,padding)



print("________________________________________Test1")    
winSize = (16,16)
blockSize = (8,8)
blockStride = (8,8)
cellSize = (4,4)
derivAperture = 1
winSigma = 4.
nbins=6
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
hogDesccv = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

winStride = (4,4)
padding = (0,0)

doTest(hogDesccv,winStride,padding)




print("________________________________________Test1")    
winSize = (32,32)
blockSize = (32,32)
blockStride = (32,32)
cellSize = (32,32)
derivAperture = 1
winSigma = 4.
nbins=9
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
hogDesccv = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

winStride = (32,32)
padding = (0,0)

doTest(hogDesccv,winStride,padding)

