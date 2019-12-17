import numpy as np
import matplotlib.pyplot as plt
from os.path import join as appendFileToFolder
from os import linesep
import cv2
import json

files=['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
#files=['train-images-idx3-ubyte','train-labels-idx1-ubyte']

outFileName='IntermidiateTest.dat'
#outFileName='Intermidiate.dat'

winSize = (32,32)
blockSize = (16,16)
blockStride = (16,16)
#cellSize = (8,8)
cellSize = (16,16)
#nbins = 3
nbins=6
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

winStride = (16,16)
padding = (0,0)
locations = ((0,0),)


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
          with open(appendFileToFolder('meta',outFileName),'w') as outputFile:
              dt = np.dtype(np.int32)
              dt = dt.newbyteorder('>')
             
              
              magicNumber=np.frombuffer(labelFile.read(4), dtype=dt)
              numDigets=np.frombuffer(labelFile.read(4), dtype=dt)
              
              magicNumber=np.frombuffer(imageFile.read(4), dtype=dt)
              numDigets=np.frombuffer(imageFile.read(4), dtype=dt)
              dimention=np.frombuffer(imageFile.read(8), dtype=dt)##rows,cols
              
              outputFile.write(json.dumps({"Dimention":dimention.tolist()})+linesep)
              
              pixelCount=dimention[0]*dimention[1]
              
              digetsPerPercent=round(numDigets[0]/100)
              percentCounter=0
              percentCompleate=0
              
              while True:
              
                  image=np.fromfile(imageFile, dtype=np.uint8, count=pixelCount)
                  lbl = np.fromfile(labelFile, dtype=np.int8, count=1)
                  
                  if image.size < pixelCount:
                      break;
                  if lbl.size < 1:
                      break;
                      
                  lbl=int(lbl[0])
                      
                  image=image.reshape(dimention)
                  
                  ret,threshImage = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
                  im2, contours, hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                  
                  
                  outputDescriptor={"classLabel":lbl}
                  
                  maxContour=max(contours,key=cv2.contourArea)
                  
                  outputDescriptor["pixelCount"]=cv2.contourArea(maxContour)
                  outputDescriptor["arkLen"]=cv2.arcLength(maxContour, False)
                  #outputDescriptor["convexContours"]=cv2.isContourConvex(maxContour)
                  
                  rect = cv2.minAreaRect(maxContour)
                  box = cv2.boxPoints(rect)
                  #box = np.int0(box)
                  
                  lengthArray=np.array([np.linalg.norm(box[0]-box[1]),np.linalg.norm(box[1]-box[2]),
                                        np.linalg.norm(box[2]-box[3]),np.linalg.norm(box[3]-box[0])])
                                        
                  a=lengthArray[0]
                  b=lengthArray[1]
                  c=lengthArray[2]
                  d=lengthArray[3]
                  
                  p=np.linalg.norm(box[1]-box[3])
                  q=np.linalg.norm(box[0]-box[2])
                                        
                  outputDescriptor["minLen"]=float(np.max(lengthArray))
                  outputDescriptor["minWidth"]=float(np.min(lengthArray))
                  
                  holdAreasq=4*p*p*q*q-(b**2+d**2-a**2-c**2)**2
                  outputDescriptor["minArea"]=0.25*np.sqrt(holdAreasq)
                  
                  #MainContImg=np.zeros(threshImage.shape)
                  #cv2.drawContours(MainContImg, maxContour, -1, 255, cv2.FILLED)
                  
                  
                  rect = cv2.minAreaRect(maxContour)
                  box = order_points(cv2.boxPoints(rect))
                    
                  dst = np.array([[0, 0],[31, 0],[31, 31],[0, 31]], dtype = "float32")
                  
                  M = cv2.getPerspectiveTransform(box, dst)
                  bigImg = cv2.warpPerspective(image, M, (32, 32))
                  
                  moments=cv2.moments(bigImg, binaryImage=False)
                  
                  for k,v in moments.items():
                    if "mu" in k:
                        outputDescriptor["Moment"+k]=v
                    if k == "m00":
                        outputDescriptor["Momentmu00"]=v


                  HuMoments=cv2.HuMoments(moments)
                  
                  
                  outputDescriptor["HuMoment0"]=float(HuMoments[0,0])                 
                  outputDescriptor["HuMoment1"]=float(HuMoments[1,0])                  
                  outputDescriptor["HuMoment2"]=float(HuMoments[2,0]) 
                  outputDescriptor["HuMoment3"]=float(HuMoments[3,0])                   
                  outputDescriptor["HuMoment4"]=float(HuMoments[4,0])                   
                  outputDescriptor["HuMoment5"]=float(HuMoments[5,0])                  
                  outputDescriptor["HuMoment6"]=float(HuMoments[6,0]) 
                  
                  im2, invContours, hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                  for i,cnt in reversed(list(enumerate(invContours))):
                      cntarea=cv2.contourArea(cnt)
                      if cntarea>700 or cntarea < 2:
                          invContours.pop(i)
    
                  
                  outputDescriptor["numValidContours"]=len(invContours) 
                  
                  maxVal=0
                  TotalSum=0
                  for cont in invContours:
                      area=cv2.contourArea(cont)
                      TotalSum+=area
                      maxVal=maxVal if maxVal > area else area
                  outputDescriptor["areaEnclosed"]=TotalSum-maxVal
                  
                  
                  
                  lines = cv2.HoughLines(threshImage,1,np.pi/180,13)
                  imhold = cv2.cvtColor(threshImage, cv2.COLOR_GRAY2BGR)
                    
                  if lines is not None:
                    HLines=[]
                    VLines=[]
                    for i in range(len(lines)):
                        isvalid=True
                        
                        rho,theta = lines[i][0]
                        
                        a = np.cos(theta)
                        b = np.sin(theta)
                        xc = a*rho
                        yc = b*rho
                        
                        if abs(b)>abs(a):
                            pint = np.array([0,yc+(a*xc/b)])
                            pend = np.array([threshImage.shape[0],yc+(a*(xc-threshImage.shape[0])/b)])
                        else:
                            pint = np.array([xc+(b*yc/a),0])
                            pend = np.array([xc+(b*(yc-threshImage.shape[1])/a),threshImage.shape[1]])
                                    
                        
                        for pintF,pendF,thetaF,rhoF in HLines:
                            if (np.linalg.norm(pintF-pint)<7 and np.linalg.norm(pendF-pend)<7):
                                isvalid=False
                            if ((abs((thetaF-theta+(np.pi/2.0))%(np.pi)-(np.pi/2.0))<np.pi/5) and 
                                abs(rhoF-rho)<7):
                                isvalid=False
                        for pintF,pendF,thetaF,rhoF in VLines:
                            if (np.linalg.norm(pintF-pint)<7 and np.linalg.norm(pendF-pend)<7):
                                isvalid=False
                            if ((abs((thetaF-theta+(np.pi/2.0))%(np.pi)-(np.pi/2.0))<np.pi/5) and 
                                abs(rhoF-rho)<7):
                                isvalid=False
                                
                            
                        if isvalid:
                            if abs((theta+(np.pi/2.0))%(np.pi)-(np.pi/2.0))<(np.pi/4.0):
                                VLines.append((pint,pend,theta,rho))
                            else:
                                HLines.append((pint,pend,theta,rho))
                    outputDescriptor["HLines"]=len(VLines)
                    outputDescriptor["VLines"]=len(HLines)                               
                  else:
                    outputDescriptor["HLines"]=0
                    outputDescriptor["VLines"]=0

                  cornerImg = cv2.cornerHarris(255-image,2,1,0.14)
                  corners=np.argwhere(cornerImg > 0.02)
                  toBeDeleted=np.array([])
                  for i, corner in enumerate(corners):
                      if i not in toBeDeleted:
                          dist=np.linalg.norm(corners-corner,axis=1)
                          deleteIndexes=np.argwhere(dist<3).reshape(-1)
                          deleteIndexes = np.delete(deleteIndexes, np.argwhere(deleteIndexes==i))
                          toBeDeleted=np.append(toBeDeleted,deleteIndexes)
                    
                  corners=np.delete(corners,toBeDeleted,axis=0)

                  outputDescriptor["cornerCount"]=len(corners)
                  
                  
                  hist = hog.compute(bigImg,winStride,padding,locations)
                  
                  for i,hel in enumerate(hist):
                      outputDescriptor["HOG"+str(i)]=float(hel[0])
                  
                  outputFile.write(json.dumps(outputDescriptor)+linesep)
                  
                  
                  percentCounter+=1
                  if percentCounter >= digetsPerPercent:
                      percentCompleate+=1
                      print("Building features "+str(percentCompleate)+"% compleate")
                      percentCounter=0
#                  if holdAreasq <0:
#                      imhold = cv2.cvtColor(threshImage, cv2.COLOR_GRAY2BGR)
#                      cv2.drawContours(imhold,[box],0,(0,0,255),1)
#                      print(lbl)
#                      plt.imshow(imhold, vmin = 0, vmax = 255)
#                      plt.show()